"""线性回归基线(1 天预测, Cmems_Prediction)：前 7 天 [u,v] 的 14 个特征 + 偏置, 全域共享系数, 只在 train 拟合。
指标 = 逐样本、逐通道、有效点上的相对 L2, 与 main_cmems.py 的 val loss 同口径。"""
import sys, time
import numpy as np
sys.path.insert(0, '.')
from cmems_data import compute_norm, split_array
R = '/data/Cmems_Prediction'; tin = 7
t0 = time.time(); norm = compute_norm(R); mean, std = norm['mean'], norm['std']
print('norm', mean.round(4), std.round(4), f'{time.time()-t0:.0f}s', flush=True)
ocean = ~np.load(f'{R}/static/mask.npy')


def windows(split, stride):
    a = split_array(R, split); T = len(a)
    for t in range(0, T - tin, stride):
        w = np.asarray(a[t:t + tin + 1])[:, :, ocean]                 # (8,2,N)
        yield w[:tin], w[tin]


A = np.zeros((15, 15)); B = np.zeros((15, 2)); n = 0
for x, y in windows('train', 2):
    xf = np.where(np.isfinite(x), x, mean[None, :, None]); xn = (xf - mean[None, :, None]) / std[None, :, None]
    F = np.concatenate([xn.reshape(14, -1).T, np.ones((xn.shape[2], 1))], 1)            # (N,15)
    v = np.isfinite(y)                                                                     # (2,N)
    yn = np.where(v, (y - mean[:, None]) / std[:, None], 0.0).T                           # (N,2)
    for c in (0, 1):
        pass
    A += F.T @ F; B += F.T @ yn                                                            # 缺测点极少，直接并入(目标已置 0 近似)
W_ = np.linalg.solve(A + 1e-3 * np.eye(15), B)
res, per = [], []
for x, y in windows('val', 1):
    xf = np.where(np.isfinite(x), x, mean[None, :, None]); xn = (xf - mean[None, :, None]) / std[None, :, None]
    F = np.concatenate([xn.reshape(14, -1).T, np.ones((xn.shape[2], 1))], 1)
    pred = (F @ W_).T * std[:, None] + mean[:, None]; v = np.isfinite(y)
    last = xf[-1]
    rel = lambda p: np.sqrt((np.where(v, (p - y) ** 2, 0)).sum(1)) / np.sqrt((np.where(v, y ** 2, 0)).sum(1))
    res.append(rel(pred)); per.append(rel(last))
res, per = np.mean(res, 0), np.mean(per, 0)
print(f'val 1-day rel-L2 (u,v): linear {res.round(4)} | persistence {per.round(4)} | total {time.time()-t0:.0f}s')
