"""Build /data/Cmems_Prediction from the OceanSR key-area product (northern Gulf of Mexico only).
No numeric processing: raw float64 values, land stays NaN. Only merge -> sort by date -> split 8:1:1 (chronological)."""
import os, glob, json, shutil, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

REGION = 'northern_gulf_of_Mexico'
SRC = f'/data/OceanSR/copernicus_key_areas_data/{REGION}/'
DST = '/data/Cmems_Prediction/'
PROJ = os.getcwd()
INK, MUTED, LAND = '#222222', '#666666', '#d9d9d9'
COL = {'train': '#2a78d6', 'val': '#eb6834', 'test': '#6250d6'}   # validated: CVD + contrast pass
assert os.listdir(DST) == [], 'target not empty'


def key(f):
    y, m, d = os.path.basename(f)[:-4].split('-'); return dt.date(int(y), int(m), int(d))


# ---------- 1. gather + continuity check ----------
files = {}
for s in ('train', 'val', 'test'):
    for f in glob.glob(f'{SRC}{s}/hr/*.npy'):
        k = key(f); assert k not in files, f'duplicate day {k}'; files[k] = f
days = sorted(files); N = len(days)
assert all((b - a).days == 1 for a, b in zip(days, days[1:])), 'dates not continuous'
ntr = int(0.8 * N); nval = (N - ntr) // 2; nte = N - ntr - nval
cuts = {'train': (0, ntr), 'val': (ntr, ntr + nval), 'test': (ntr + nval, N)}
print(f'N={N} train={ntr} val={nval} test={nte}', flush=True)

lon = np.load(SRC + 'static/lon.npy'); lat = np.load(SRC + 'static/lat.npy'); mask = np.load(SRC + 'static/mask.npy')
H, W = mask.shape
os.makedirs(DST + 'static', exist_ok=True)
for n, a in (('lon', lon), ('lat', lat), ('mask', mask)): np.save(DST + f'static/{n}.npy', a)

# ---------- 2. write splits (raw float64) ----------
bad = {'shape': 0, 'dtype': 0, 'land_has_value': 0, 'inf': 0}
ocean_nan = []   # (split, date, [n_missing_u, n_missing_v]) : ocean cells that are NaN in the source


def load_check(f):
    a = np.load(f)
    nan = np.isnan(a)
    r = (a.shape != (2, H, W), a.dtype != np.float64, bool((~nan[:, mask]).any()), bool(np.isinf(a).any()))
    return a, r, [int(nan[c][~mask].sum()) for c in (0, 1)]


for s, (i0, i1) in cuts.items():
    os.makedirs(DST + s, exist_ok=True)
    sd = days[i0:i1]
    np.save(DST + f'{s}/dates.npy', np.array(sd, dtype='datetime64[D]'))
    out = np.lib.format.open_memmap(DST + f'{s}/uv.npy', mode='w+', dtype=np.float64, shape=(len(sd), 2, H, W))
    with ThreadPoolExecutor(16) as ex:
        for i, (a, r, miss) in enumerate(ex.map(load_check, [files[d] for d in sd])):
            out[i] = a
            for k, v in zip(bad, r): bad[k] += int(v)
            if any(miss): ocean_nan.append((s, str(sd[i]), miss))
            if i % 1000 == 0: print(f'  {s} {i}/{len(sd)}', flush=True)
    out.flush(); del out
print('checks (count of violating days):', bad, '| days with missing ocean cells:', len(ocean_nan), flush=True)
assert sum(bad.values()) == 0

# ---------- 3. stats / baseline (information only; data are NOT normalised) ----------
ocean = ~mask
tr = np.load(DST + 'train/uv.npy', mmap_mode='r')
s1 = np.zeros(2); s2 = np.zeros(2); mn = np.full(2, np.inf); mx = np.full(2, -np.inf); cnt = np.zeros(2)
for i in range(0, len(tr), 400):
    blk = np.asarray(tr[i:i + 400])[:, :, ocean]                 # (t,2,Nocean)
    fin = np.isfinite(blk); b0 = np.where(fin, blk, 0.0)
    s1 += b0.sum((0, 2)); s2 += (b0 ** 2).sum((0, 2)); cnt += fin.sum((0, 2))
    mn = np.minimum(mn, np.where(fin, blk, np.inf).min((0, 2))); mx = np.maximum(mx, np.where(fin, blk, -np.inf).max((0, 2)))
mean = s1 / cnt; std = np.sqrt(s2 / cnt - mean ** 2)
va = np.asarray(np.load(DST + 'val/uv.npy', mmap_mode='r'))[:, :, ocean]
vfin = np.isfinite(va); pers = {}
for lag in (1, 2, 3, 5, 7, 10):
    a_, b_ = va[lag:], va[:-lag]; ok = vfin[lag:] & vfin[:-lag]
    d = np.sqrt(np.where(ok, (a_ - b_) ** 2, 0.0).sum(2)); n = np.sqrt(np.where(ok, a_ ** 2, 0.0).sum(2))
    pers[lag] = (d / n).mean(0)
del va
dlon = float(np.diff(lon, axis=1).mean()); dlat = float(np.diff(lat, axis=0).mean())
ckm_ew = dlon * 111.32 * np.cos(np.deg2rad(lat.mean())); ckm_ns = dlat * 110.57

# ---------- 4. split.json ----------
meta = {'region': REGION, 'source': SRC, 'ratio': '8:1:1 (chronological, contiguous)', 'n_days': N,
        'array_layout': '(T, 2, 240, 240) = (day, [u, v], lat-index, lon-index)', 'dtype': 'float64', 'land': 'NaN (mask==True)',
        'ocean_nan_days': [{'split': a, 'date': b, 'n_nan_ocean_cells_u_v': c} for a, b, c in ocean_nan],
        'splits': {s: {'start': str(days[i0]), 'end': str(days[i1 - 1]), 'n_days': i1 - i0, 'index_start': i0, 'index_end_exclusive': i1}
                   for s, (i0, i1) in cuts.items()}}
json.dump(meta, open(DST + 'split.json', 'w'), indent=2, ensure_ascii=False)

# ---------- 5. figure ----------
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'normal'})
arr = {s: np.load(DST + f'{s}/uv.npy', mmap_mode='r') for s in cuts}
D0 = dt.date(2015, 6, 1); sp = [s for s, (i0, i1) in cuts.items() if days[i0] <= D0 <= days[i1 - 1]][0]
snap = np.asarray(arr[sp][(D0 - days[cuts[sp][0]]).days])
fig = plt.figure(figsize=(19, 11.5)); gs = GridSpec(2, 3, height_ratios=[1.45, 1], hspace=0.32, wspace=0.38)


def panel(ax, f, title, cmap, label, vmin=None, vmax=None, sym=False, vec=False):
    f = np.ma.masked_invalid(f)
    if sym: v = np.nanpercentile(np.abs(f.compressed()), 99); vmin, vmax = -v, v
    cm = plt.get_cmap(cmap).copy(); cm.set_bad(LAND)
    im = ax.pcolormesh(lon, lat, f, cmap=cm, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    if vec:
        s = slice(6, None, 12)
        q = ax.quiver(lon[s, s], lat[s, s], np.ma.masked_invalid(snap[0][s, s]), np.ma.masked_invalid(snap[1][s, s]), angles='xy',
                      scale_units='xy', scale=1.0, pivot='mid', width=0.003, headwidth=4, headlength=5, color=INK, alpha=0.8)
        ax.quiverkey(q, 0.15, 0.93, 0.5, '0.5 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 8})
    ax.set_title(title, loc='left'); ax.set_xlabel('Longitude (°E, negative = west)'); ax.set_ylabel('Latitude (°N)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5)); ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label(label, color=INK)
    for k in ('top', 'right'): ax.spines[k].set_visible(False)


panel(fig.add_subplot(gs[0, 0]), snap[0], f'Eastward velocity u, {D0}', 'RdBu_r', 'm/s', sym=True)
panel(fig.add_subplot(gs[0, 1]), snap[1], f'Northward velocity v, {D0}', 'RdBu_r', 'm/s', sym=True)
panel(fig.add_subplot(gs[0, 2]), np.hypot(snap[0], snap[1]), f'Speed |u,v| with vectors, {D0}', 'Blues', 'm/s', 0, 1.0, vec=True)

ax = fig.add_subplot(gs[1, :]); alld = np.array(days, dtype='datetime64[D]')
for s, (i0, i1) in cuts.items():
    sm = np.concatenate([np.nanmean(np.hypot(*np.asarray(arr[s][j:j + 500]).transpose(1, 0, 2, 3)), axis=(1, 2)) for j in range(0, i1 - i0, 500)])
    ax.plot(alld[i0:i1], sm, color=COL[s], lw=0.9, label=f'{s.capitalize()}  {days[i0]} ~ {days[i1-1]}  ({i1-i0} days, {100*(i1-i0)/N:.0f}%)')
    ax.axvspan(alld[i0], alld[i1 - 1], color=COL[s], alpha=0.06, lw=0)
    ax.text(alld[i0] + (alld[i1 - 1] - alld[i0]) / 2, 1.02, s.capitalize(), transform=ax.get_xaxis_transform(), ha='center', color=INK, fontsize=10)
ax.set_xlim(alld[0], alld[-1]); ax.set_ylabel('Domain-mean speed (m/s)'); ax.set_xlabel('Date'); ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.set_title('Chronological 8:1:1 split of the daily record', loc='left', pad=22)
ax.grid(alpha=0.25, lw=0.6); ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
for k in ('top', 'right'): ax.spines[k].set_visible(False)
fig.suptitle(f'Cmems_Prediction: CMEMS 1/12° surface currents, Northern Gulf of Mexico, {H}×{W} grid, daily {days[0]} ~ {days[-1]}',
             x=0.01, y=0.975, ha='left', fontsize=14, color=INK)
fig.savefig(DST + 'overview.png', dpi=110, bbox_inches='tight'); plt.close(fig)
shutil.copy(DST + 'overview.png', f'{PROJ}/cmems_prediction_overview.png')

# ---------- 6. README ----------
sz = {s: os.path.getsize(DST + f'{s}/uv.npy') / 1e9 for s in cuts}
sp_rows = '\n'.join(f"| {s} | {days[i0]} ~ {days[i1-1]} | {i1-i0} | {100*(i1-i0)/N:.1f}% | [{i0}, {i1}) | `{s}/uv.npy` ({sz[s]:.2f} GB) |" for s, (i0, i1) in cuts.items())
pr_rows = '\n'.join(f'| {lag} | {pers[lag][0]:.3f} | {pers[lag][1]:.3f} |' for lag in pers)
readme = f'''# Cmems_Prediction：墨西哥湾北部 1/12° 表层海流时序预测数据集

> 一句话：哥白尼（CMEMS）全球海洋再分析（1/12°）在墨西哥湾北部截取的 {H}×{W} 区域，**{days[0]} ~ {days[-1]} 共 {N} 天**逐日表层流速 `[u, v]`，按时间顺序 **8:1:1** 划分为 train / val / test，**未做任何数值处理**（不归一化、不填 NaN、不改 dtype）。

![overview](overview.png)

## 1. 路径与目录结构

数据位置：`/data/Cmems_Prediction/`

```text
Cmems_Prediction/
├── README.md              # 本文件
├── overview.png           # 可视化：u/v/速度快照 + 划分时间线
├── split.json             # 划分信息（日期范围、天数、索引区间、数组布局）
├── static/
│   ├── lon.npy            # (240, 240) float32  经度，负值为西经
│   ├── lat.npy            # (240, 240) float32  纬度
│   └── mask.npy           # (240, 240) bool     True = 陆地/无效点，False = 海洋
├── train/
│   ├── uv.npy             # ({ntr}, 2, 240, 240) float64  [u, v]，陆地 NaN
│   └── dates.npy          # ({ntr},) datetime64[D]       每帧对应日期
├── val/
│   ├── uv.npy             # ({nval}, 2, 240, 240) float64
│   └── dates.npy
├── test/
│   ├── uv.npy             # ({nte}, 2, 240, 240) float64
│   └── dates.npy
└── scripts/
    └── build_cmems_prediction.py   # 生成本数据集的脚本
```

## 2. 数组含义

| 文件 | 形状 | dtype | 说明 |
|---|---|---|---|
| `{{split}}/uv.npy` | (T, 2, 240, 240) | float64 | 轴 0 = 天（按日期升序）；轴 1 = 通道，**0 = u（向东为正）、1 = v（向北为正）**，单位 m/s；轴 2、3 = 纬向/经向格点索引。陆地为 NaN |
| `{{split}}/dates.npy` | (T,) | datetime64[D] | 与 `uv.npy` 第 0 轴一一对应 |
| `static/mask.npy` | (240, 240) | bool | **True = 陆地**；陆地点在 `uv.npy` 中恒为 NaN（已逐日校验）。注意个别海洋点也会缺测，见下节 |
| `static/lon.npy`、`lat.npy` | (240, 240) | float32 | 格点经纬度 |

## 3. 数据划分（8:1:1，按时间连续切分）

| 划分 | 日期范围 | 天数 | 占比 | 全序列索引 | 文件 |
|---|---|---|---|---|---|
{sp_rows}

- 按时间顺序连续切三段，**不打乱**：train 最旧，test 最新。
- 全部 {N} 天日期连续、无重复、无缺失。
- 滑动窗口样本请**在各划分内部**构造，不要跨划分边界取窗口，否则相邻划分会共享日期造成泄漏。

### 缺测海洋点（NaN）

`mask.npy` 只标陆地。源数据里有 **{len(ocean_nan)} 天**存在零星海洋点缺测（值为 NaN）：train {sum(1 for o in ocean_nan if o[0]=='train')} 天、val {sum(1 for o in ocean_nan if o[0]=='val')} 天、test {sum(1 for o in ocean_nan if o[0]=='test')} 天；每天缺 {min(max(o[2]) for o in ocean_nan)} ~ {max(max(o[2]) for o in ocean_nan)} 个点（每帧约 {int(ocean.sum())} 个海洋点）。这些点未做任何填补，明细见 `split.json` 的 `ocean_nan_days`。
**所以计算 loss / 指标时，请用 `np.isfinite(x)` 当有效点，不要只用 `~mask`**，否则这些天的结果会变成 NaN。

## 4. 来源与空间信息

- 产品：Copernicus Marine Service `GLOBAL_MULTIYEAR_PHY_001_030`（Global Ocean Physics Reanalysis），数据集 `cmems_mod_glo_phy_my_0.083deg_P1D-m`，版本 202311。
- 变量：`uo`、`vo`；**表层（约 0.494 m）日平均流速**，不是深度平均。这是模式再分析结果，不是实测。
- 区域：{REGION}（沿用 OceanSR 的命名），从全球网格截取起点 100°W、18.5°N 的 {H}×{W} 子网格。
- 原始区域数据：`{SRC}`（本数据集由其 `train/val/test` 三个文件夹合并、按日期排序后重新切分得到；原划分的日期范围互相重叠，所以未沿用）。

| 项目 | 数值 |
|---|---|
| 网格 | {H} × {W} |
| 经度范围 | {lon.min():.2f}° ~ {lon.max():.2f}°（负值为西经） |
| 纬度范围 | {lat.min():.2f}°N ~ {lat.max():.2f}°N |
| 格距 | {dlon:.4f}° × {dlat:.4f}°（1/12°） |
| 单格大小 | 约 {ckm_ew:.1f} km（东西）× {ckm_ns:.1f} km（南北） |
| 区域大小 | 约 {ckm_ew*W:.0f} km × {ckm_ns*H:.0f} km |
| 海洋点占比 | {ocean.mean():.1%}（{int(ocean.sum())} 个海洋格点 / 帧，不含上述零星缺测） |
| 时间分辨率 | 1 天 |

## 5. 统计信息（仅 train 段、仅有效海洋点，**仅供参考，数据本身未被归一化**）

| 通道 | mean | std | min | max |
|---|---|---|---|---|
| u | {mean[0]:.4f} | {std[0]:.4f} | {mn[0]:.3f} | {mx[0]:.3f} |
| v | {mean[1]:.4f} | {std[1]:.4f} | {mn[1]:.3f} | {mx[1]:.3f} |

做归一化时请只用 train 段统计，不要用全量重算。

### 参考基线：persistence（用第 t 天预测第 t+k 天），val 段

逐日、逐通道、仅两天都有效的海洋点的相对 L2 误差 `‖x(t+k) − x(t)‖₂ / ‖x(t+k)‖₂`，再对天取平均：

| 间隔 k（天） | u | v |
|---|---|---|
{pr_rows}

模型在同一口径下应明显优于该基线才算有预测能力。

## 6. 使用示例

```python
import numpy as np

root = "/data/Cmems_Prediction"
train = np.load(f"{{root}}/train/uv.npy", mmap_mode="r")   # (T, 2, 240, 240) float64
dates = np.load(f"{{root}}/train/dates.npy")
mask = np.load(f"{{root}}/static/mask.npy")                # True = 陆地

# 在 train 内部取窗口：前 7 天 -> 后 20 天
tin, tout = 7, 20
t = 100
x = np.asarray(train[t : t + tin])                          # (7, 2, 240, 240)
y = np.asarray(train[t + tin : t + tin + tout])             # (20, 2, 240, 240)
valid = np.isfinite(x)                                      # 有效点（排除陆地和个别缺测点），用于 loss / 指标
x = np.nan_to_num(x)                                        # 陆地/缺测 NaN -> 0（按需，数据文件本身保持 NaN）
```

## 7. 注意事项

1. **没有做任何处理**：数值、dtype（float64）、NaN 都与源数据一致。float64 转 float32 不是无损的（抽样校验过），如需省显存，请在训练时再转换。
2. **陆地是 NaN，另有零星海洋缺测点也是 NaN**：请用 `np.isfinite` 得到有效点来排除，不要把它们当成 0 计入 loss 或指标。
3. **只用 train 算统计量**，val / test 使用同一组统计量。
4. **表层流速 + 模式再分析**：与深度平均流速（如 GBA 近海数据）物理含义不同，不要直接对比数值。
5. 生成脚本：`scripts/build_cmems_prediction.py`（合并、排序、切分、逐日校验：形状、dtype、有无 Inf、陆地点是否全为 NaN，并记录缺测的海洋点）。
'''
open(DST + 'README.md', 'w').write(readme)
os.makedirs(DST + 'scripts', exist_ok=True)
shutil.copy(__file__, DST + 'scripts/build_cmems_prediction.py')
print('BUILD DONE', flush=True)
