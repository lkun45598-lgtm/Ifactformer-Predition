"""线性回归基线(1 天预测)。特征=前 7 天 [u,v] (可加 k×k 邻域)，全域共享一组系数，仅 train 段拟合。
指标与 main_gba.py 的 masked_rel_l2_ch 相同: 逐样本、逐通道相对 L2(海上点)，再对 u,v 与样本取平均。"""
import sys, time
import numpy as np, torch
from gba_data import load_gba, train_stats, fill_land

d = sys.argv[1]; ks = [int(k) for k in sys.argv[2].split(',')]; stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
data, valid, split = load_gba(d)
mean, std = train_stats(data, split); fill_land(data, mean)
x = torch.from_numpy((data - mean) / std).float().permute(0, 3, 1, 2)   # (T,C,H,W) 归一化
T, C, H, W = x.shape; tin = 7
m = torch.from_numpy(valid)                                            # (H,W) True=海
print(d, 'T', T, 'grid', H, W, 'ocean frac', float(m.float().mean()), flush=True)


def feats(t, k):
    """窗口 t..t+6 -> (H*W, 14*k*k)"""
    w = x[t:t + tin].reshape(tin * C, H, W)
    if k > 1:
        p = k // 2; wp = torch.nn.functional.pad(w[None], (p, p, p, p), mode='replicate')[0]
        w = torch.cat([wp[:, i:i + H, j:j + W] for i in range(k) for j in range(k)], 0)
    return w.reshape(w.shape[0], -1).T


def rel(pred, y):  # 物理空间，逐通道
    s = torch.from_numpy(std).float(); mu = torch.from_numpy(mean).float()
    pr = pred * s[:, None] + mu[:, None]; yy = y * s[:, None] + mu[:, None]
    return ((pr - yy).pow(2).sum(1).sqrt() / yy.pow(2).sum(1).sqrt()).mean().item()


a, b = split['train_start'], split['train_end']
tr = list(range(a, b - tin, stride))
va = list(range(split['val_start'], split['val_end'] - tin))
for k in ks:
    t0 = time.time(); dim = tin * C * k * k + 1
    A = torch.zeros(dim, dim, dtype=torch.float64); R = torch.zeros(dim, C, dtype=torch.float64)
    for t in tr:
        F = feats(t, k)[m.reshape(-1)]; F = torch.cat([F, torch.ones(len(F), 1)], 1)
        Y = x[t + tin].reshape(C, -1).T[m.reshape(-1)]
        A += (F.T @ F).double(); R += (F.T @ Y).double()
    wgt = torch.linalg.solve(A + 1e-3 * torch.eye(dim, dtype=torch.float64), R).float()
    res, pers = [], []
    for t in va:
        F = feats(t, k)[m.reshape(-1)]; F = torch.cat([F, torch.ones(len(F), 1)], 1)
        pred = (F @ wgt).T; y = x[t + tin].reshape(C, -1)[:, m.reshape(-1)]
        res.append(rel(pred, y)); pers.append(rel(x[t + tin - 1].reshape(C, -1)[:, m.reshape(-1)], y))
    print(f'k={k} feats={dim-1:4d} | val rel-L2(day1) linear {np.mean(res):.4f} | persistence {np.mean(pers):.4f} | {time.time()-t0:.0f}s', flush=True)
