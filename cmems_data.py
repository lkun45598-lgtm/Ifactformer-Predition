"""Cmems_Prediction 数据：窗口只在各划分内部取；NaN(陆地+零星缺测)填 train 均值并给出 valid 掩膜。"""
import json, os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def split_array(root, split):
    return np.load(f'{root}/{split}/uv.npy', mmap_mode='r')          # (T,2,H,W) float64, 原始数据


def compute_norm(root, chunk=300):
    """仅 train：有效点上的 mean/std，以及相邻两天都有效的日增量 std。返回 {'mean','std','dstd'}"""
    a = split_array(root, 'train'); ocean = ~np.load(f'{root}/static/mask.npy')
    s1 = np.zeros(2); s2 = np.zeros(2); n = np.zeros(2); d2 = np.zeros(2); dn = np.zeros(2); d1 = np.zeros(2); prev = None
    for i in range(0, len(a), chunk):
        blk = np.asarray(a[i:i + chunk])[:, :, ocean]                 # (t,2,Nocean)
        if prev is not None: blk_d = np.concatenate([prev[None], blk]) 
        else: blk_d = blk
        fin = np.isfinite(blk); b0 = np.where(fin, blk, 0.0)
        s1 += b0.sum((0, 2)); s2 += (b0 ** 2).sum((0, 2)); n += fin.sum((0, 2))
        dd = blk_d[1:] - blk_d[:-1]; df = np.isfinite(dd); dz = np.where(df, dd, 0.0)
        d1 += dz.sum((0, 2)); d2 += (dz ** 2).sum((0, 2)); dn += df.sum((0, 2)); prev = blk[-1]
    mean = s1 / n; std = np.sqrt(s2 / n - mean ** 2)
    dmean = d1 / dn; dstd = np.sqrt(d2 / dn - dmean ** 2)
    return {'mean': mean, 'std': np.maximum(std, 1e-6), 'dstd': np.maximum(dstd, 1e-6)}


class CmemsWindows(Dataset):
    """x=days[t:t+tin], y=days[t+tin:t+tin+tout]，t 只在本划分内取。返回 x, y(已填充), valid_y(bool, 原始是否有效)"""
    def __init__(self, root, split, tin, tout, mean, stride=1):
        self.path = f'{root}/{split}/uv.npy'; self.tin, self.tout = tin, tout
        self.mean = np.asarray(mean, dtype=np.float32); self.arr = None
        T = np.load(self.path, mmap_mode='r').shape[0]
        self.starts = list(range(0, T - tin - tout + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        if self.arr is None: self.arr = np.load(self.path, mmap_mode='r')   # 每个 worker 各自打开
        t = self.starts[i]
        w = np.transpose(np.asarray(self.arr[t:t + self.tin + self.tout]), (0, 2, 3, 1))   # (T,H,W,C)
        valid = np.isfinite(w)
        w = np.where(valid, w, self.mean).astype(np.float32)
        return torch.from_numpy(w[:self.tin]), torch.from_numpy(w[self.tin:]), torch.from_numpy(valid[self.tin:])
