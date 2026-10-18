"""CMEMS 1/12-degree surface currents: grid facts + overview figures (English labels, lon/lat axes)."""
import glob, os, datetime as dt
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/data/OceanSR/copernicus_key_areas_data/'
OUT = '.'
REG = {'northern_gulf_of_Mexico': 'Northern Gulf of Mexico', 'chesapeake_bay': 'Chesapeake Bay / Gulf Stream',
       'yangtze_river_estuary': 'Yangtze Estuary / Kuroshio', 'pearl_river_estuary': 'Pearl River Estuary / South China Sea'}
INK, MUTED, LAND = '#222222', '#666666', '#d9d9d9'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'normal'})


def day_file(r, d):
    for s in ('train', 'val', 'test'):
        f = f'{B}{r}/{s}/hr/{d.year}-{d.month}-{d.day}.npy'
        if os.path.exists(f): return f
    raise FileNotFoundError(d)


def load_day(r, d):
    return np.load(day_file(r, d)).astype(np.float32)          # (2,H,W), land = NaN


def static(r):
    return [np.load(f'{B}{r}/static/{k}.npy') for k in ('lon', 'lat', 'mask')]


# ---------- grid facts ----------
print(f'{"region":26s} {"grid":9s} {"lon range":18s} {"lat range":16s} {"dlon":7s} {"dlat":7s} {"cell km (E-W x N-S)":20s} {"domain km":14s} ocean')
for r in REG:
    lon, lat, mask = static(r)
    dlon = np.diff(lon, axis=1).mean(); dlat = np.diff(lat, axis=0).mean()
    ckm_ew = dlon * 111.32 * np.cos(np.deg2rad(lat.mean())); ckm_ns = dlat * 110.57
    print(f'{r:26s} {lon.shape[0]}x{lon.shape[1]:<5d} {lon.min():7.2f}~{lon.max():7.2f}E  {lat.min():5.2f}~{lat.max():5.2f}N  '
          f'{dlon:.4f}  {dlat:.4f}  {ckm_ew:5.1f} x {ckm_ns:5.1f}      '
          f'{ckm_ew*lon.shape[1]:5.0f} x {ckm_ns*lat.shape[0]:5.0f}  {(~mask).mean():.0%}')


def panel(ax, lon, lat, mask, f, title, cmap, label, vmin=None, vmax=None, sym=False, quiver=None):
    f = np.ma.masked_invalid(np.where(mask, np.nan, f))
    if sym:
        v = np.nanpercentile(np.abs(f.compressed()), 99); vmin, vmax = -v, v
    cm = plt.get_cmap(cmap).copy(); cm.set_bad(LAND)
    im = ax.pcolormesh(lon, lat, f, cmap=cm, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)
    if quiver is not None:
        u, v = quiver; s = slice(6, None, 12)
        um = np.ma.masked_invalid(np.where(mask, np.nan, u)[s, s]); vm = np.ma.masked_invalid(np.where(mask, np.nan, v)[s, s])
        q = ax.quiver(lon[s, s], lat[s, s], um, vm, angles='xy', scale_units='xy', scale=1.0, pivot='mid',
                      width=0.003, headwidth=4, headlength=5, color=INK, alpha=0.8)
        ax.quiverkey(q, 0.15, 0.93, 0.5, '0.5 m/s', labelpos='E', coordinates='axes', fontproperties={'size': 8})
    ax.set_title(title, loc='left'); ax.set_xlabel('Longitude (°E, negative = west)' if lon.min() < 0 else 'Longitude (°E)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5)); ax.set_ylabel('Latitude (°N)')
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02); cb.set_label(label, color=INK)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    return im


D0 = dt.date(2015, 6, 1)

# ---------- Fig 1: four regions, same day ----------
fig, axs = plt.subplots(2, 2, figsize=(13, 12))
for ax, r in zip(axs.ravel(), REG):
    lon, lat, mask = static(r); x = load_day(r, D0)
    panel(ax, lon, lat, mask, np.hypot(x[0], x[1]), f'{REG[r]}  ({D0})', 'Blues', 'speed (m/s)', 0, 1.0, quiver=(x[0], x[1]))
fig.suptitle('CMEMS 1/12° surface currents, four 240×240 regions (~20°×20°, 1993–2021.6 daily)', x=0.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f'{OUT}/cmems_regions_overview.png', dpi=110); plt.close(fig)

# ---------- Fig 2: Gulf of Mexico detail ----------
r = 'northern_gulf_of_Mexico'; lon, lat, mask = static(r)
days = sorted(set(dt.date(1993, 1, 1) + dt.timedelta(days=i) for i in range(0, 10408, 15)))
sub = np.stack([load_day(r, d) for d in days]); mu = np.nanmean(sub, 0); sd = np.nanstd(sub, 0)
x = load_day(r, D0)
fig, ax = plt.subplots(2, 3, figsize=(19, 11.5))
panel(ax[0, 0], lon, lat, mask, x[0], f'Eastward velocity u, {D0}', 'RdBu_r', 'm/s', sym=True)
panel(ax[0, 1], lon, lat, mask, x[1], f'Northward velocity v, {D0}', 'RdBu_r', 'm/s', sym=True)
panel(ax[0, 2], lon, lat, mask, np.hypot(x[0], x[1]), f'Speed |u,v| with vectors, {D0}', 'Blues', 'm/s', 0, 1.0, quiver=(x[0], x[1]))
panel(ax[1, 0], lon, lat, mask, mu[0], 'u: temporal mean (every 15th day)', 'RdBu_r', 'm/s', sym=True)
panel(ax[1, 1], lon, lat, mask, sd[0], 'u: temporal std', 'Blues', 'm/s', 0, None)
panel(ax[1, 2], lon, lat, mask, sd[1], 'v: temporal std', 'Blues', 'm/s', 0, None)
fig.suptitle(f'Northern Gulf of Mexico, 240×240 grid, 1/12°  (n={len(days)} sampled days for mean/std)', x=0.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f'{OUT}/cmems_gom_detail.png', dpi=110); plt.close(fig)

# ---------- Fig 3: how fast does the field change? t, t+1, t+3, t+7 ----------
lags = [0, 1, 3, 7]; xs = [load_day(r, D0 + dt.timedelta(days=k)) for k in lags]
fig, ax = plt.subplots(2, 4, figsize=(21, 10.5))
for j, (k, xx) in enumerate(zip(lags, xs)):
    panel(ax[0, j], lon, lat, mask, np.hypot(xx[0], xx[1]), f'Speed, day t+{k}' if k else f'Speed, day t ({D0})', 'Blues', 'm/s', 0, 1.0, quiver=(xx[0], xx[1]))
    if k == 0:
        ax[1, 0].axis('off')
    else:
        d = xx[0] - xs[0][0]
        rel = np.sqrt(np.nansum(d ** 2) / np.nansum(xs[0][0] ** 2))
        panel(ax[1, j], lon, lat, mask, d, f'u(t+{k}) − u(t),  rel-L2 = {rel:.2f}', 'RdBu_r', 'm/s', -0.5, 0.5)
fig.suptitle('Northern Gulf of Mexico: day-to-day evolution (compare: GBA coastal shelf loses correlation in ~2 days)', x=0.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f'{OUT}/cmems_gom_evolution.png', dpi=110); plt.close(fig)
print('done')
