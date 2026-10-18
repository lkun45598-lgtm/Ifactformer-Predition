"""真正的 20 天自回归评估 + persistence / 气候态基线，逐 lead time 的 masked 相对 L2"""
import argparse, os, pickle
import numpy as np
import torch, yaml

import json
from gba_data import load_gba, WindowDataset, predict_next, fill_land
from libs.model import Model
from libs.utils import dict2namespace

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, help='结果目录，含 checkpoint_best.pth / norm.pkl / gba.yml')
p.add_argument('--ckpt', default='checkpoint_best.pth')
p.add_argument('--split', default='test', choices=['val', 'test'])
p.add_argument('--steps', type=int, default=20)
p.add_argument('--batch_size', type=int, default=4)
p.add_argument('--hist_overlap', action='store_true')
p.add_argument('--stride', type=int, default=1, help='每隔 stride 个窗口评估一个')
p.add_argument('--amp', action='store_true')
p.add_argument('--gpu', default='0')
args = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = dict2namespace(yaml.safe_load(open(f'{args.run}/gba.yml')))
tin, S = config.model.in_time_window, args.steps
data, valid, split = load_gba(config.data_dir)
norm = pickle.load(open(f'{args.run}/norm.pkl', 'rb'))
norm_t = tuple(torch.tensor(norm[k], device=device).float() for k in ('mean', 'std', 'dstd'))
residual = json.load(open(f'{args.run}/args.json')).get('residual', False)
mean = norm_t[0]
fill_land(data, norm['mean'])
mask = torch.from_numpy(valid).to(device)[None, :, :, None]

a, b = split[f'{args.split}_start'], split[f'{args.split}_end']
ds = WindowDataset(data, a, b, tin, S, args.hist_overlap)
ds.starts = ds.starts[::args.stride]
print(args.split, 'windows', len(ds), '(stride %d)' % args.stride, flush=True)

doy_np = np.load(f'{config.data_dir}/doy.npy').astype(int) - 1
if os.path.exists(f'{config.data_dir}/clim.npz'):
    clim = np.transpose(np.load(f'{config.data_dir}/clim.npz')['clim'], (0, 2, 3, 1))   # (366,H,W,C)
else:  # 缺省：仅用 train 段按年积日平均
    tr = slice(split['train_start'], split['train_end'])
    clim = np.stack([data[tr][doy_np[tr] == d].mean(0) if (doy_np[tr] == d).any() else data[tr].mean(0)
                     for d in range(366)]).astype(np.float32)
clim = torch.from_numpy(clim).to(device)
doy = torch.from_numpy(doy_np).long().to(device)

H, W = data.shape[1:3]
L = getattr(config, 'pos_length', [1.0, 1.0])
pos_lst = [torch.linspace(0, L[0], H, device=device).unsqueeze(-1), torch.linspace(0, L[1], W, device=device).unsqueeze(-1)]
model = Model(config.model).to(device)
model.load_state_dict(torch.load(f'{args.run}/{args.ckpt}', map_location=device)); model.eval()


def metrics(pred, y, c):
    """(B,H,W,C) -> 各指标 (B,C)。rel=相对L2, rmse=m/s, corr=空间相关, acc=相对气候态 c 的异常相关"""
    m = mask.float(); n = m.sum()
    d = (pred - y) * m
    rmse = (d.pow(2).sum(dim=(1, 2)) / n).sqrt()
    rel = d.pow(2).sum(dim=(1, 2)).sqrt() / (y * m).pow(2).sum(dim=(1, 2)).sqrt().clamp_min(1e-8)

    def corr(a, b):
        a = (a - (a * m).sum(dim=(1, 2), keepdim=True) / n) * m
        b = (b - (b * m).sum(dim=(1, 2), keepdim=True) / n) * m
        return (a * b).sum(dim=(1, 2)) / (a.pow(2).sum(dim=(1, 2)).sqrt() * b.pow(2).sum(dim=(1, 2)).sqrt()).clamp_min(1e-8)
    return {'rel': rel, 'rmse': rmse, 'corr': corr(pred, y), 'acc': corr(pred - c, y - c)}


KEYS = ['rel', 'rmse', 'corr', 'acc']
res = {(p, k): [] for p in ['model', 'persist', 'clim'] for k in KEYS}
with torch.no_grad():
    for s in range(0, len(ds), args.batch_size):
        idx = range(s, min(s + args.batch_size, len(ds)))
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        y = torch.stack([ds[i][1] for i in idx]).to(device)
        t0 = torch.tensor([ds.starts[i] + tin for i in idx], device=device)  # 第 1 个目标日的天索引
        last, x_cur = x[:, -1], x
        acc = {key: [] for key in res}
        for k in range(S):
            pred = predict_next(model, x_cur, norm_t, pos_lst, residual, amp=args.amp, mask=mask)
            c = clim[doy[t0 + k]]
            for p, f in (('model', pred), ('persist', last), ('clim', c)):
                for key, v in metrics(f, y[:, k], c).items():
                    acc[(p, key)].append(v)
            x_cur = torch.cat([x_cur[:, 1:], pred.unsqueeze(1)], dim=1)
        print(f'  batch {s // args.batch_size + 1}/{(len(ds) + args.batch_size - 1) // args.batch_size}', flush=True)
        for key in res:
            res[key].append(torch.stack(acc[key], 1).cpu())
out = {f'{p}_{k}': torch.cat(v).mean(0).numpy() for (p, k), v in res.items()}   # (S,C)
for k in KEYS:
    print(f'\n[{k}] lead | model u v | persist u v | clim u v' + ('   (clim 的 acc 恒为 0，仅作占位)' if k == 'acc' else ''))
    for d in range(S):
        print(f'{d+1:4d} | ' + ' | '.join(f'{out[f"{p}_{k}"][d,0]:.3f} {out[f"{p}_{k}"][d,1]:.3f}' for p in ['model', 'persist', 'clim']))
print('\nmean over leads (u,v):', {n: v.mean(0).round(3).tolist() for n, v in out.items() if n.endswith('_rel')})
np.savez(f'{args.run}/eval_{args.split}.npz', **out)
