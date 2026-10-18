import numpy as np
import torch
from torch.utils.data import Dataset


def load_gba(data_dir):
    """返回 data (T,H,W,C) float32、有效区域 mask (H,W) bool(True=海)、split 字典"""
    data = np.load(f'{data_dir}/uv_daily.npy').astype(np.float32)  # (T,C,H,W)
    data = np.transpose(data, (0, 2, 3, 1))                        # (T,H,W,C)
    land = np.load(f'{data_dir}/mask.npy')                         # True=陆地
    valid = ~land & ~np.isnan(data).any(axis=(0, 3))   # 陆地 NaN 保留，由 fill_land 按统计量填充
    z = np.load(f'{data_dir}/split.npz')
    split = {k: int(z[k]) for k in ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']}
    return data, valid, split


def train_stats(data, split):
    """仅用 train 段；(C,) 的 mean/std"""
    tr = data[split['train_start']:split['train_end']]
    return np.nanmean(tr, axis=(0, 1, 2)), np.maximum(np.nanstd(tr, axis=(0, 1, 2)), 1e-6)


class WindowDataset(Dataset):
    """在 [start,end) 内开窗: x=data[t:t+tin], y=data[t+tin:t+tin+tout]。
    hist_overlap=True 时输入允许越过 start 往前取，目标仍在段内。"""
    def __init__(self, data, start, end, tin, tout, hist_overlap=False):
        self.data, self.tin, self.tout = torch.from_numpy(data), tin, tout
        first = start - tin if hist_overlap else start
        self.starts = list(range(max(first, 0), end - tin - tout + 1))
        if hist_overlap:
            self.starts = [t for t in self.starts if t + tin >= start]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]
        return self.data[t:t + self.tin], self.data[t + self.tin:t + self.tin + self.tout]


def delta_std(data, split):
    tr = data[split['train_start']:split['train_end']]
    return np.maximum(np.nanstd(tr[1:] - tr[:-1], axis=(0, 1, 2)), 1e-6)


def fill_land(data, mean):
    """陆地 NaN 填 mean（归一化后为 0），原地操作"""
    nan = np.isnan(data)
    if nan.any():
        data[nan] = np.broadcast_to(mean.astype(np.float32), data.shape)[nan]
    return data


def predict_next(model, x_cur, norm_t, pos_lst, residual, amp=False, mask=None):
    """x_cur: (B,T,H,W,C) 物理量 -> 下一天 (B,H,W,C) 物理量"""
    mean, std, dstd = norm_t
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        out = model((x_cur - mean) / std, pos_lst)
    out = out.float()
    pred = x_cur[:, -1] + out * dstd if residual else out * std + mean
    return torch.where(mask, pred, mean) if mask is not None else pred   # 陆地不反馈


def masked_rel_l2_ch(pred, y, mask):
    """逐通道相对 L2 再对通道平均 -> (B,)，避免 u 的量级压过 v"""
    d = ((pred - y) * mask).pow(2).sum(dim=(1, 2))
    n = (y * mask).pow(2).sum(dim=(1, 2)).clamp_min(1e-16)
    return (d / n).sqrt().mean(dim=1)
