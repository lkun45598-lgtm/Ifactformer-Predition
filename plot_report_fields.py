"""场对比图：真值 / 微调后预报 / persistence 的速度场，以及三者的误差图(含相对 L2 与 RMSE)。
用法: python plot_cmems_fields.py OUT.png   (读 results_cmems_remote/ 下 K2_ftA 与 K1_res 的 eval npz 中保存的同一窗口)"""
import sys
import numpy as np, datetime as dt, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else 'report_fields.png'
RR = 'results_cmems_remote/'; DATA = '/data/Cmems_Prediction/'
ft = np.load(RR + 'K4_ftB_full/eval_val_ftB_full.npz'); base = np.load(RR + 'K1_res_cont/eval_val_k1_cont.npz')
lon = np.load(DATA + 'static/lon.npy'); lat = np.load(DATA + 'static/lat.npy')
T, V, Lst, D = ft['sample_true'], ft['sample_valid'], ft['sample_last'], ft['sample_dates']
assert np.array_equal(T, base['sample_true']) and np.array_equal(D, base['sample_dates'])     # 两个模型评估的是同一窗口
PF, PB = ft['sample_pred'], base['sample_pred']
d0 = dt.date.fromisoformat(str(D[0])); in0, in1 = d0 - dt.timedelta(days=7), d0 - dt.timedelta(days=1)
INK, MUTED, LAND = '#222222', '#666666', '#d9d9d9'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
spd = lambda a: np.hypot(a[..., 0], a[..., 1]); OK = V.all(-1)
vec = lambda F, i: np.hypot(F[..., 0] - T[i][..., 0], F[..., 1] - T[i][..., 1])      # 向量误差大小 √(Δu²+Δv²)
leads = [1, 3, 7, 14, 20]


def stats(F, i):
    """u、v 分量各自在有效海洋点上的相对 L2 与 RMSE (与逐日曲线图同一定义)"""
    m = V[i]; d = (F - T[i]) * m; n = m.sum((0, 1)); sse = (d ** 2).sum((0, 1))
    return np.sqrt(sse) / np.sqrt(((T[i] * m) ** 2).sum((0, 1))), np.sqrt(sse / n)


def panel(a, f, ok, cmap, vmin, vmax):
    f = np.ma.masked_where(~ok, f); cm = plt.get_cmap(cmap).copy(); cm.set_bad(LAND)
    im = a.pcolormesh(lon, lat, f, cmap=cm, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    a.set_aspect(1 / np.cos(np.deg2rad(lat.mean()))); a.xaxis.set_major_locator(plt.MaxNLocator(4))
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    return im


def note(a, title, F, i):
    rel, rmse = stats(F, i)
    a.text(0.03, 0.975, f'{title}\nRel. L2:  u {rel[0]:.3f}  v {rel[1]:.3f}  mean {rel.mean():.3f}\nRMSE:  u {rmse[0]:.3f}  v {rmse[1]:.3f} m/s',
           transform=a.transAxes, va='top', fontsize=8.6, color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2.5))


ROWS = ['Reanalysis (truth)\nsurface current speed\n|V| = √(u²+v²)  [m/s]',
        'Forecast: multi-step\nfine-tuned model (K=4)\nspeed |V|  [m/s]',
        'Forecast: persistence\n(last observed day\nheld fixed)\nspeed |V|  [m/s]',
        'Error of fine-tuned model\nvector error\n|Δ(u,v)| = √(Δu²+Δv²)  [m/s]',
        'Error of single-step model\n(K=1, 12 epochs)\nvector error  [m/s]',
        'Error of persistence\nvector error\n|Δ(u,v)|  [m/s]']
fig, ax = plt.subplots(6, len(leads), figsize=(4.5 * len(leads) + 2.2, 26), gridspec_kw={'hspace': 0.17, 'wspace': 0.1, 'left': 0.145, 'right': 0.935, 'top': 0.878, 'bottom': 0.05})
for j, l in enumerate(leads):
    i = l - 1; ok = OK[i]
    im1 = panel(ax[0, j], spd(T[i]), ok, 'Blues', 0, 1.0); panel(ax[1, j], spd(PF[i]), ok, 'Blues', 0, 1.0); panel(ax[2, j], spd(Lst), ok, 'Blues', 0, 1.0)
    im2 = panel(ax[3, j], vec(PF[i], i), ok, 'YlOrRd', 0, 0.4); panel(ax[4, j], vec(PB[i], i), ok, 'YlOrRd', 0, 0.4)
    panel(ax[5, j], vec(Lst, i), ok, 'YlOrRd', 0, 0.4)
    note(ax[3, j], 'Fine-tuned (K=4)', PF[i], i); note(ax[4, j], 'Single-step (K=1)', PB[i], i); note(ax[5, j], 'Persistence', Lst, i)
    ax[0, j].set_title(f'Forecast lead {l} d\nvalid on {D[i]}', fontsize=12, loc='left', color=INK)
    ax[5, j].set_xlabel('Longitude (°E)', fontsize=9.5)
for r in range(6): ax[r, 0].set_ylabel(ROWS[r] + '\n\nLatitude (°N)', fontsize=10.5, labelpad=8)
fig.colorbar(im1, ax=ax[:3, :], shrink=0.35, pad=0.012, label='Speed |V| (m/s)')
fig.colorbar(im2, ax=ax[3:, :], shrink=0.4, pad=0.012, label='Vector error magnitude |forecast − truth| = √(Δu²+Δv²)  (m/s)')
fig.suptitle('Forecast vs. reanalysis: surface current speed, Northern Gulf of Mexico (100°W–80°W, 18.5°N–38.4°N)', x=0.012, y=0.972, ha='left', fontsize=16, color=INK)
fig.text(0.012, 0.950, f'Initial state: 7 observed daily means, {in0} to {in1}.  Autoregressive 20-day forecast, shown at leads 1, 3, 7, 14, 20 d.  One validation window (index 10 of the 51 evaluated).\n'
         'Data: CMEMS GLOBAL_MULTIYEAR_PHY_001_030 reanalysis, 1/12° daily-mean surface currents (0.494 m).', fontsize=10.5, color=MUTED, va='top')
fig.text(0.012, 0.012, 'Relative L2 = ‖forecast − truth‖₂ / ‖truth‖₂, computed separately for u and v over the valid ocean cells of this window ("mean" = average of the u and v values); RMSE in m/s.\n'
         'Error maps show the vector error magnitude √(Δu²+Δv²) on a fixed 0–0.4 m/s scale (the same scale in every panel); the relative L2 is the norm of exactly this error field divided by the norm of the truth.', fontsize=9.5, color=MUTED)
fig.savefig(out, dpi=80); plt.close(fig)
print('ok', in0, in1, D[0])
