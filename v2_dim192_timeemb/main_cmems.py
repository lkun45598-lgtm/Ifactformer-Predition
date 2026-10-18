import argparse, json, os, pickle, shutil
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.checkpoint
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cmems_data import CmemsWindows, compute_norm
from gba_data import predict_next, masked_rel_l2_ch
from libs.model import Model
from libs.utils import dict2namespace, count_params

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/cmems.yml')
p.add_argument('--rollout', type=int, default=1, help='训练时自回归步数 K')
p.add_argument('--stride', type=int, default=1, help='训练窗口间隔')
p.add_argument('--init', default=None)
p.add_argument('--norm', default=None, help='norm.pkl（compute_norm 的结果，多步微调必须沿用）')
p.add_argument('--lr', type=float, default=None)
p.add_argument('--epochs', type=int, default=None)
p.add_argument('--batch_size', type=int, default=None, help='每卡 batch')
p.add_argument('--warmup', type=float, default=1.0, help='warmup epoch 数')
p.add_argument('--residual', action='store_true')
p.add_argument('--amp', action='store_true')
p.add_argument('--ckpt_grad', action='store_true', help='对每个 rollout 步做梯度检查点(省显存, 约 +30% 时间)')
p.add_argument('--mem_frac', type=float, default=0.0, help='>0 时限制本进程显存占比（与他人共用显卡时用）')
p.add_argument('--max_steps', type=int, default=0, help='>0 时只跑这么多步（冒烟测试）')
p.add_argument('--start_epoch', type=int, default=0, help='从第几个 epoch 续训：学习率日程按 12 epoch 余弦的该位置接着走（需配合 --init；优化器动量不保存，故重新累积）')
p.add_argument('--rewarm', type=int, default=0, help='续训时对学习率再做这么多步的线性 warmup（动量清零后防止震荡）')
p.add_argument('--tag', default='')
args = p.parse_args()

ddp = 'RANK' in os.environ
rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1)); local = int(os.environ.get('LOCAL_RANK', 0))
if ddp:
    dist.init_process_group('nccl'); torch.cuda.set_device(local)
device = torch.device('cuda', local)
if args.mem_frac > 0: torch.cuda.set_per_process_memory_fraction(args.mem_frac, device)
main = rank == 0
log_print = lambda *a, **k: print(*a, **k, flush=True) if main else None

config = dict2namespace(yaml.safe_load(open(args.config))); tr = config.training
lr = args.lr or tr.lr; epochs = args.epochs or tr.epochs; bs = args.batch_size or tr.batch_size
tin, K = config.model.in_time_window, args.rollout
torch.manual_seed(1234 + rank); np.random.seed(1234)

save_dir = Path(config.log_dir) / f'K{K}{args.tag}'
if main:
    i = 0
    while save_dir.exists():
        i += 1; save_dir = Path(config.log_dir) / f'K{K}{args.tag}_{i}'
    save_dir.mkdir(parents=True)
    shutil.copyfile(args.config, save_dir / 'cmems.yml'); json.dump(vars(args), open(save_dir / 'args.json', 'w'))
    print(save_dir, 'world', world, flush=True)

root = config.data_dir
norm = pickle.load(open(args.norm, 'rb')) if args.norm else compute_norm(root)
if main: pickle.dump(norm, open(save_dir / 'norm.pkl', 'wb'))
norm_t = tuple(torch.tensor(norm[k], device=device).float() for k in ('mean', 'std', 'dstd'))
ocean = torch.from_numpy(~np.load(f'{root}/static/mask.npy')).to(device)[None, :, :, None]   # 陆地不反馈回输入

train_ds = CmemsWindows(root, 'train', tin, K, norm['mean'], args.stride)
val_ds = CmemsWindows(root, 'val', tin, K, norm['mean'], 1)
log_print('train/val windows', len(train_ds), len(val_ds), 'global batch', bs * world)
tsamp = DistributedSampler(train_ds, world, rank, shuffle=True, drop_last=True) if ddp else None
vsamp = DistributedSampler(val_ds, world, rank, shuffle=False) if ddp else None
train_loader = DataLoader(train_ds, batch_size=bs, sampler=tsamp, shuffle=tsamp is None, drop_last=True, num_workers=3, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=bs, sampler=vsamp, shuffle=False, num_workers=2)

H, W = np.load(f'{root}/static/mask.npy').shape
L = getattr(config, 'pos_length', [1.0, 1.0])
pos_lst = [torch.linspace(0, L[0], H, device=device).unsqueeze(-1), torch.linspace(0, L[1], W, device=device).unsqueeze(-1)]

model = Model(config.model).to(device)
if args.init: model.load_state_dict(torch.load(args.init, map_location=device))
log_print('params', count_params(model))
manual = ddp and (K > 1 or args.ckpt_grad)      # 多步/检查点: 不用 DDP 包装, 反向后手动 all-reduce 梯度(要求各进程同一 --init)
if manual: assert args.init, 'manual sync 需要所有进程从同一个 --init 开始'
net = DDP(model, device_ids=[local]) if (ddp and not manual) else model
opt = torch.optim.AdamW(model.parameters(), lr=lr)
steps_per_ep = len(train_loader); total = epochs * steps_per_ep; warm = max(int(args.warmup * steps_per_ep), 1)
off = args.start_epoch * steps_per_ep      # 续训：日程从第 start_epoch 个 epoch 的起点接着走
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s0: min(1.0, (s0 + 1) / args.rewarm if args.rewarm else 1.0) * (
        (s0 + off + 1) / warm if s0 + off < warm else
        0.5 * (1 + np.cos(np.pi * (s0 + off - warm) / max(total - warm, 1))) * 0.99 + 0.01))


def rollout_loss(x, y, vy, tf):
    x_cur, loss = x, 0.0
    for k in range(y.shape[1]):
        step = lambda xc: predict_next(net, xc, norm_t, pos_lst, args.residual, args.amp, ocean, tf[:, tin + k])
        pred = torch.utils.checkpoint.checkpoint(step, x_cur, use_reentrant=False) if (args.ckpt_grad and net.training) else step(x_cur)
        loss = loss + masked_rel_l2_ch(pred, y[:, k], vy[:, k]).mean()      # 只在原始有效点上算
        x_cur = torch.cat([x_cur[:, 1:], pred.unsqueeze(1)], dim=1)
    return loss / y.shape[1]


def allmean(v):
    t = torch.tensor([v], device=device)
    if ddp: dist.all_reduce(t); t /= world
    return t.item()


best, log = 1e10, {'train': [], 'val': []}
for ep in range(args.start_epoch, epochs):
    t0 = default_timer(); net.train(); tl = []
    if tsamp: tsamp.set_epoch(ep)
    for n, (x, y, vy, tf) in enumerate(train_loader):
        loss = rollout_loss(x.to(device, non_blocking=True), y.to(device, non_blocking=True), vy.to(device, non_blocking=True), tf.to(device))
        opt.zero_grad(); loss.backward()
        if manual:
            for q in model.parameters():
                if q.grad is not None: dist.all_reduce(q.grad); q.grad /= world
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tl.append(loss.item())
        if n % 20 == 0: log_print(f'ep {ep} it {n}/{steps_per_ep} loss {loss.item():.4f} mem {torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
        if args.max_steps and n + 1 >= args.max_steps: break
    if args.max_steps:
        log_print('smoke test done, losses', [round(v, 4) for v in tl]); break
    if manual:   # 校验手动梯度同步: 各进程参数应完全一致
        cs = torch.stack([q.detach().double().sum() for q in model.parameters()]).sum(); mx, mn = cs.clone(), cs.clone()
        dist.all_reduce(mx, op=dist.ReduceOp.MAX); dist.all_reduce(mn, op=dist.ReduceOp.MIN)
        log_print(f'param checksum spread across ranks: {(mx - mn).item():.3e}')
    net.eval(); vl = []
    with torch.no_grad():
        for x, y, vy, tf in val_loader:
            vl.append(rollout_loss(x.to(device), y.to(device), vy.to(device), tf.to(device)).item())
    tr_l, va_l = allmean(float(np.mean(tl))), allmean(float(np.mean(vl)))
    log['train'].append(tr_l); log['val'].append(va_l)
    if main:
        info = (f'Epoch {ep} | {default_timer()-t0:.0f}s | train {tr_l:.4f} | val {va_l:.4f} | '
                f'lr {opt.param_groups[0]["lr"]:.2e} | mem {torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
        print(info, flush=True); open(save_dir / 'training_epoch_log.txt', 'a').write(info + '\n')
        np.save(save_dir / 'loss_log.npy', log)
        if va_l < best: best = va_l; torch.save(model.state_dict(), save_dir / 'checkpoint_best.pth')
        torch.save(model.state_dict(), save_dir / 'checkpoint_last.pth')
if main and not args.max_steps:
    print('best val', best, save_dir)
if ddp: dist.destroy_process_group()
