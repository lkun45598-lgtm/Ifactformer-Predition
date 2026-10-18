"""画 Cmems 评估。用法: python plot_cmems_eval.py eval_xxx.npz OUTPREFIX "checkpoint 说明"
 (1) {pre}_leads.png : 四个指标随预报天数变化，关键天数直接标数值
 (2) {pre}_fields.png: 一个窗口 真值/模型/persistence/模型误差/persistence误差，每个误差面板标 RMSE 与相对L2"""
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npz, pre = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else ''
R = '/data/Cmems_Prediction/'
e = np.load(npz); lon = np.load(R + 'static/lon.npy'); lat = np.load(R + 'static/lat.npy')
INK, MUTED, LAND = '#222222', '#666666', '#d9d9d9'
COL = {'model': '#2a78d6', 'persist': '#eb6834', 'clim': '#6250d6'}
NAME = {'model': 'Model (autoregressive rollout)', 'persist': 'Persistence (last observed day held fixed)', 'clim': 'Climatology (train, by day of year)'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'normal'})

# ---------- Fig 1 ----------
S = e['model_rel'].shape[0]; lead = np.arange(1, S + 1); SEL = [d for d in (1, 5, 10, 15, 20) if d <= S]
spec = [('rel', 'Relative L2 error  ‖pred − truth‖ / ‖truth‖   (lower is better)', 'lower', '{:.2f}', (0, 1.25)),
        ('rmse', 'RMSE (m/s)   (lower is better)', 'lower', '{:.3f}', (0.03, 0.228)),
        ('acc', 'ACC: correlation of anomalies w.r.t. climatology   (higher is better)', 'higher', '{:.2f}', (-0.1, 1.12)),
        ('corr', 'Spatial correlation of the full fields   (higher is better)', 'higher', '{:.2f}', (-0.1, 1.12))]
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
for a, (k, t, better, fmt, yl) in zip(ax.ravel(), spec):
    for q in ('model', 'persist', 'clim'):
        if k == 'acc' and q == 'clim': continue
        y = e[f'{q}_{k}'].mean(1); a.plot(lead, y, color=COL[q], lw=2, label=NAME[q])
        if q == 'clim': continue
        a.plot(SEL, y[np.array(SEL) - 1], 'o', color=COL[q], ms=5)
        up = (q == 'model') == (better == 'higher')            # 好的那条线的标签放在它"更好"的一侧
        for d in SEL:
            a.annotate(fmt.format(y[d - 1]), (d, y[d - 1]), textcoords='offset points', xytext=(0, 8 if up else -14), ha='center', fontsize=8.5, color=INK)
    a.set_title(t, loc='left', fontsize=10.5); a.set_xlabel('Forecast lead (days)'); a.set_xlim(0.5, S + 0.8); a.set_xticks([1, 5, 10, 15, 20])
    a.grid(alpha=0.25, lw=0.6)
    if yl: a.set_ylim(*yl)
    if k == 'acc': a.axhline(0, color=MUTED, lw=0.8)
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
ax[0, 0].legend(frameon=False, loc='lower right', fontsize=9)
fig.suptitle(f'Cmems, Northern Gulf of Mexico: 20-day rollout vs baselines ({tag})', x=0.01, ha='left', fontsize=13, color=INK)
fig.text(0.01, 0.005, 'Mean of u and v over validation windows (every 20th window). The model is trained for 1-step prediction only and rolled out autoregressively. '
         'Climatology has no ACC (it is the reference).', fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(f'{pre}_leads.png', dpi=110); plt.close(fig)

# ---------- Fig 2 ----------
if 'sample_pred' in e.files:
    P, T, V, Lst, D = e['sample_pred'], e['sample_true'], e['sample_valid'], e['sample_last'], e['sample_dates']
    sp = lambda a: np.hypot(a[..., 0], a[..., 1]); OK = V.all(-1)
    leads = [l for l in (1, 3, 7, 14, 20) if l <= len(P)]

    def stats(F, i):
        m = V[i]; d = (F - T[i]) * m; n = m.sum((0, 1))
        return np.sqrt((d ** 2).sum((0, 1)) / n), np.sqrt((d ** 2).sum((0, 1))) / np.sqrt(((T[i] * m) ** 2).sum((0, 1)))

    rows = ['Truth\n(speed |u,v|)', 'Model\n(speed |u,v|)', 'Persistence\n(last observed day,\nheld fixed)', 'Model − Truth\n(speed difference)', 'Persistence − Truth\n(speed difference)']
    fig, ax = plt.subplots(5, len(leads), figsize=(4.5 * len(leads), 21.5), gridspec_kw={'hspace': 0.16, 'wspace': 0.1})

    def panel(a, f, ok, cmap, vmin, vmax):
        f = np.ma.masked_where(~ok, f); cm = plt.get_cmap(cmap).copy(); cm.set_bad(LAND)
        im = a.pcolormesh(lon, lat, f, cmap=cm, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
        a.set_aspect(1 / np.cos(np.deg2rad(lat.mean()))); a.xaxis.set_major_locator(plt.MaxNLocator(4))
        for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
        return im

    def note(a, F, i, name):
        rm, rel = stats(F, i)
        a.text(0.03, 0.97, f'{name}\nu: RMSE {rm[0]:.3f} m/s, rel-L2 {rel[0]:.2f}\nv: RMSE {rm[1]:.3f} m/s, rel-L2 {rel[1]:.2f}', transform=a.transAxes,
               va='top', fontsize=8.5, color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=2.5))

    for j, l in enumerate(leads):
        i = l - 1; ok = OK[i]
        im1 = panel(ax[0, j], sp(T[i]), ok, 'Blues', 0, 1.0); panel(ax[1, j], sp(P[i]), ok, 'Blues', 0, 1.0); panel(ax[2, j], sp(Lst), ok, 'Blues', 0, 1.0)
        im2 = panel(ax[3, j], sp(P[i]) - sp(T[i]), ok, 'RdBu_r', -0.5, 0.5); panel(ax[4, j], sp(Lst) - sp(T[i]), ok, 'RdBu_r', -0.5, 0.5)
        note(ax[3, j], P[i], i, 'Model error'); note(ax[4, j], Lst, i, 'Persistence error')
        ax[0, j].set_title(f'Lead {l} d   ({D[i]})', fontsize=12, loc='left')
        ax[4, j].set_xlabel('Longitude (°E)', fontsize=9)
    for r_ in range(5): ax[r_, 0].set_ylabel(rows[r_] + '\n\nLatitude (°N)', fontsize=10.5)
    fig.colorbar(im1, ax=ax[:3, :], shrink=0.35, pad=0.012, label='speed (m/s)')
    fig.colorbar(im2, ax=ax[3:, :], shrink=0.55, pad=0.012, label='speed difference (m/s), red = overestimate')
    fig.suptitle(f'Truth vs {len(P)}-day rollout from {D[0]} (one validation window), Northern Gulf of Mexico   [{tag}]', x=0.01, y=0.915, ha='left', fontsize=14, color=INK)
    fig.text(0.01, 0.085, 'RMSE and rel-L2 are computed over valid ocean cells of this single window, separately for u and v. '
             'Maps show speed only; the numbers include the direction error.', fontsize=9, color=MUTED)
    fig.savefig(f'{pre}_fields.png', dpi=85, bbox_inches='tight'); plt.close(fig)
print('plotted')
