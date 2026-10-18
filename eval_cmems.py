"""Cmems 真实 20 天自回归评估：逐 lead 的 rel-L2 / RMSE / 空间相关 / ACC，对比 persistence 与 train 气候态。
有效点 = 原始数据有限的点(排除陆地与缺测)。同时保存一个窗口的预测场用于画图。"""
import argparse, json, os, pickle
import numpy as np, torch, yaml

p = argparse.ArgumentParser()
p.add_argument('--run', required=True); p.add_argument('--ckpt', default='checkpoint_best.pth')
p.add_argument('--split', default='val', choices=['val', 'test']); p.add_argument('--steps', type=int, default=20)
p.add_argument('--stride', type=int, default=20); p.add_argument('--batch_size', type=int, default=4)
p.add_argument('--gpu', default='0'); p.add_argument('--amp', action='store_true')
p.add_argument('--sample', type=int, default=10, help='保存第几个评估窗口的场')
args = p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from cmems_data import CmemsWindows, split_array
from gba_data import predict_next
from libs.model import Model
from libs.utils import dict2namespace

device = torch.device('cuda')
config = dict2namespace(yaml.safe_load(open(f'{args.run}/cmems.yml')))
residual = json.load(open(f'{args.run}/args.json')).get('residual', False)
norm = pickle.load(open(f'{args.run}/norm.pkl', 'rb'))
norm_t = tuple(torch.tensor(norm[k], device=device).float() for k in ('mean', 'std', 'dstd'))
root = config.data_dir; tin = config.model.in_time_window; S = args.steps
land = np.load(f'{root}/static/mask.npy'); H, W = land.shape
ocean = torch.from_numpy(~land).to(device)[None, :, :, None]
ds = CmemsWindows(root, args.split, tin, S, norm['mean'], args.stride)
dates = np.load(f'{root}/{args.split}/dates.npy')
doy_of = lambda d: (d - d.astype('datetime64[Y]').astype('datetime64[D]')).astype(int) + 1   # 1..366

# train 气候态(逐年积日平均, 仅 train, 忽略 NaN)，缓存
cf = f'{root}/clim_train.npy'
if not os.path.exists(cf):
    a = split_array(root, 'train'); td = np.load(f'{root}/train/dates.npy'); dy = doy_of(td) - 1
    sm = np.zeros((366, 2, H, W)); ct = np.zeros((366, 2, H, W))
    for i in range(0, len(a), 300):
        blk = np.asarray(a[i:i + 300]); fin = np.isfinite(blk)
        for j in range(len(blk)):
            sm[dy[i + j]] += np.where(fin[j], blk[j], 0.0); ct[dy[i + j]] += fin[j]
    cl = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
    np.save(cf, np.transpose(cl, (0, 2, 3, 1)).astype(np.float32))
clim = np.load(cf); clim = np.where(np.isfinite(clim), clim, norm['mean'].astype(np.float32))
clim = torch.from_numpy(clim).to(device)

L = getattr(config, 'pos_length', [1.0, 1.0])
pos_lst = [torch.linspace(0, L[0], H, device=device).unsqueeze(-1), torch.linspace(0, L[1], W, device=device).unsqueeze(-1)]
model = Model(config.model).to(device)
model.load_state_dict(torch.load(f'{args.run}/{args.ckpt}', map_location=device)); model.eval()
print(args.split, 'windows', len(ds), f'(stride {args.stride})', flush=True)


def metrics(pred, y, vy, c):
    m = vy.float(); n = m.sum((1, 2)).clamp_min(1)                       # (B,C)
    sse = ((pred - y) * m).pow(2).sum((1, 2))
    def corr(a, b):
        a = (a - (a * m).sum((1, 2), keepdim=True) / n[:, None, None, :]) * m
        b = (b - (b * m).sum((1, 2), keepdim=True) / n[:, None, None, :]) * m
        return (a * b).sum((1, 2)) / (a.pow(2).sum((1, 2)).sqrt() * b.pow(2).sum((1, 2)).sqrt()).clamp_min(1e-8)
    return {'rel': sse.sqrt() / (y * m).pow(2).sum((1, 2)).sqrt().clamp_min(1e-8), 'rmse': (sse / n).sqrt(),
            'corr': corr(pred, y), 'acc': corr(pred - c, y - c)}


KEYS = ['rel', 'rmse', 'corr', 'acc']; PRED = ['model', 'persist', 'clim']
res = {(q, k): [] for q in PRED for k in KEYS}; sample = None
with torch.no_grad():
    for s in range(0, len(ds), args.batch_size):
        idx = list(range(s, min(s + args.batch_size, len(ds))))
        items = [ds[i] for i in idx]
        x = torch.stack([it[0] for it in items]).to(device); y = torch.stack([it[1] for it in items]).to(device)
        vy = torch.stack([it[2] for it in items]).to(device)
        last, x_cur = x[:, -1], x; acc = {key: [] for key in res}; preds = []
        for k in range(S):
            pred = predict_next(model, x_cur, norm_t, pos_lst, residual, args.amp, ocean)
            dy = np.array([doy_of(dates[ds.starts[i] + tin + k]) - 1 for i in idx])
            c = clim[torch.from_numpy(dy).to(device)]
            for q, f in (('model', pred), ('persist', last), ('clim', c)):
                for key, v in metrics(f, y[:, k], vy[:, k], c).items(): acc[(q, key)].append(v)
            preds.append(pred); x_cur = torch.cat([x_cur[:, 1:], pred.unsqueeze(1)], dim=1)
        for key in res: res[key].append(torch.stack(acc[key], 1).cpu())
        if args.sample in idx:
            j = idx.index(args.sample)
            sample = {'sample_pred': torch.stack(preds, 1)[j].cpu().numpy(), 'sample_true': y[j].cpu().numpy(),
                      'sample_valid': vy[j].cpu().numpy(), 'sample_last': last[j].cpu().numpy(),
                      'sample_dates': np.array([str(dates[ds.starts[args.sample] + tin + k]) for k in range(S)])}
        print(f'  batch {s // args.batch_size + 1}/{(len(ds) + args.batch_size - 1) // args.batch_size}', flush=True)
out = {f'{q}_{k}': torch.cat(v).mean(0).numpy() for (q, k), v in res.items()}      # (S,C)
perwin = {f'{q}_{k}_perwin': torch.cat(v).numpy() for (q, k), v in res.items()}   # (N窗口, S天, C) 每个窗口每天的指标
win_dates = np.array([str(dates[ds.starts[i] + tin]) for i in range(len(ds))])                      # 每个窗口第 1 个预报日
for k in ['rel', 'acc', 'rmse', 'corr']:
    print(f'\n[{k}] lead | model u v | persist u v | clim u v')
    for d in (0, 1, 2, 4, 6, 9, 14, 19):
        if d < S: print(f'{d+1:4d} | ' + ' | '.join(f'{out[f"{q}_{k}"][d,0]:.3f} {out[f"{q}_{k}"][d,1]:.3f}' for q in PRED))
print('\nmean over leads (u,v):', {q: out[f'{q}_rel'].mean(0).round(3).tolist() for q in PRED})
np.savez(f'{args.run}/eval_{args.split}.npz', **out, **perwin, win_first_target_date=win_dates, **(sample or {}))
