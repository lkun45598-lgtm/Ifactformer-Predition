import argparse, json, os, pickle, shutil
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gba_data import (load_gba, train_stats, delta_std, fill_land, WindowDataset,
                      predict_next, masked_rel_l2_ch)
from libs.model import Model
from libs.utils import dict2namespace, count_params

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/gba.yml')
p.add_argument('--rollout', type=int, default=1, help='训练时自回归步数 K')
p.add_argument('--init', default=None)
p.add_argument('--norm', default=None, help='沿用上一阶段 norm.pkl（多步微调必须传）')
p.add_argument('--lr', type=float, default=None)
p.add_argument('--epochs', type=int, default=None)
p.add_argument('--batch_size', type=int, default=None, help='每卡 batch')
p.add_argument('--warmup', type=int, default=1, help='warmup epoch 数，之后余弦衰减')
p.add_argument('--residual', action='store_true', help='输出 = 当天值 + 增量*dstd')
p.add_argument('--amp', action='store_true')
p.add_argument('--hist_overlap', action='store_true')
p.add_argument('--tag', default='')
args = p.parse_args()

ddp = 'RANK' in os.environ
rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
local = int(os.environ.get('LOCAL_RANK', 0))
if ddp:
    dist.init_process_group('nccl'); torch.cuda.set_device(local)
device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
main = rank == 0
log_print = lambda *a, **k: print(*a, **k, flush=True) if main else None

cfg = yaml.safe_load(open(args.config)); config = dict2namespace(cfg)
tr = config.training
lr = args.lr or tr.lr
epochs = args.epochs or tr.epochs
bs = args.batch_size or tr.batch_size
tin, K = config.model.in_time_window, args.rollout
torch.manual_seed(1234 + rank); np.random.seed(1234)

save_dir = Path(config.log_dir) / f'K{K}{args.tag}'
if main:
    i = 0
    while save_dir.exists():
        i += 1; save_dir = Path(config.log_dir) / f'K{K}{args.tag}_{i}'
    save_dir.mkdir(parents=True)
    shutil.copyfile(args.config, save_dir / 'gba.yml')
    json.dump(vars(args), open(save_dir / 'args.json', 'w'))
    print(save_dir, 'world', world, flush=True)

data, valid, split = load_gba(config.data_dir)
if args.norm:
    norm = pickle.load(open(args.norm, 'rb'))
else:
    mean, std = train_stats(data, split)
    norm = {'mean': mean, 'std': std, 'dstd': delta_std(data, split)}
fill_land(data, norm['mean'])
if main:
    pickle.dump(norm, open(save_dir / 'norm.pkl', 'wb'))
norm_t = tuple(torch.tensor(norm[k], device=device).float() for k in ('mean', 'std', 'dstd'))
mask = torch.from_numpy(valid).to(device)[None, :, :, None]

train_ds = WindowDataset(data, split['train_start'], split['train_end'], tin, K)
val_ds = WindowDataset(data, split['val_start'], split['val_end'], tin, K, args.hist_overlap)
log_print('train/val windows', len(train_ds), len(val_ds), 'global batch', bs * world)
tsamp = DistributedSampler(train_ds, world, rank, shuffle=True, drop_last=True) if ddp else None
vsamp = DistributedSampler(val_ds, world, rank, shuffle=False) if ddp else None
train_loader = DataLoader(train_ds, batch_size=bs, sampler=tsamp, shuffle=tsamp is None,
                          drop_last=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=bs, sampler=vsamp, shuffle=False, num_workers=2)

H, W = data.shape[1:3]
L = getattr(config, 'pos_length', [1.0, 1.0])
pos_lst = [torch.linspace(0, L[0], H, device=device).unsqueeze(-1),
           torch.linspace(0, L[1], W, device=device).unsqueeze(-1)]

model = Model(config.model).to(device)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=device))
log_print('params', count_params(model))
net = DDP(model, device_ids=[local]) if ddp else model
opt = torch.optim.AdamW(model.parameters(), lr=lr)
steps_per_ep = len(train_loader); total = epochs * steps_per_ep; warm = args.warmup * steps_per_ep
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))) * 0.99 + 0.01)


def rollout_loss(x, y):
    x_cur, loss = x, 0.0
    for k in range(y.shape[1]):
        pred = predict_next(net, x_cur, norm_t, pos_lst, args.residual, args.amp, mask)
        loss = loss + masked_rel_l2_ch(pred, y[:, k], mask).mean()
        x_cur = torch.cat([x_cur[:, 1:], pred.unsqueeze(1)], dim=1)
    return loss / y.shape[1]


def allmean(v):
    t = torch.tensor([v], device=device)
    if ddp: dist.all_reduce(t); t /= world
    return t.item()


best, log = 1e10, {'train': [], 'val': []}
for ep in range(epochs):
    t0 = default_timer(); net.train(); tl = []
    if tsamp: tsamp.set_epoch(ep)
    for n, (x, y) in enumerate(train_loader):
        loss = rollout_loss(x.to(device), y.to(device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tl.append(loss.item())
        if n % 20 == 0:
            log_print(f'ep {ep} it {n}/{steps_per_ep} loss {loss.item():.4f}')
    net.eval(); vl = []
    with torch.no_grad():
        for x, y in val_loader:
            vl.append(rollout_loss(x.to(device), y.to(device)).item())
    tr_l, va_l = allmean(float(np.mean(tl))), allmean(float(np.mean(vl)))
    log['train'].append(tr_l); log['val'].append(va_l)
    if main:
        info = (f'Epoch {ep} | {default_timer()-t0:.0f}s | train {tr_l:.4f} | val {va_l:.4f} | '
                f'lr {opt.param_groups[0]["lr"]:.2e} | mem {torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
        print(info, flush=True)
        open(save_dir / 'training_epoch_log.txt', 'a').write(info + '\n')
        np.save(save_dir / 'loss_log.npy', log)
        if va_l < best:
            best = va_l; torch.save(model.state_dict(), save_dir / 'checkpoint_best.pth')
if main:
    torch.save(model.state_dict(), save_dir / 'checkpoint_final.pth')
    print('best val', best, save_dir)
if ddp: dist.destroy_process_group()
