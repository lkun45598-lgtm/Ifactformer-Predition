"""GBA dataset overview figures (English labels, lon/lat axes)."""
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/data1/user/gjx/ocean_uv_forecast/data_gba/'
S = '/data1/user/gjx/ocean_uv_forecast/data_gba_crop149x148/'
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
DAY = 100
INK, MUTED = '#222222', '#666666'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK,
                     'axes.titlesize': 11, 'axes.titleweight': 'normal'})

uv = np.load(B + 'uv_daily.npy', mmap_mode='r')
land = np.load(B + 'mask.npy'); h = np.load(B + 'h.npy')
lon = np.load(B + 'lon.npy'); lat = np.load(B + 'lat.npy')
slon, slat = np.load(S + 'lon.npy'), np.load(S + 'lat.npy')
i0, j0 = np.unravel_index((np.abs(lon - slon[0, 0]) + np.abs(lat - slat[0, 0])).argmin(), lon.shape)
box = (slice(i0, i0 + 149), slice(j0, j0 + 148))
sub = np.asarray(uv[::5]); mu = np.nanmean(sub, 0); sd = np.nanstd(sub, 0)

def panel(ax, f, title, cmap, label, sym=False):
    f = np.ma.masked_where(land, f)
    v = np.nanpercentile(np.abs(f.compressed()), 99) if sym else None
    kw = dict(vmin=-v, vmax=v) if sym else {}
    cm = plt.get_cmap(cmap).copy(); cm.set_bad('#d9d9d9')
    im = ax.pcolormesh(lon, lat, f, cmap=cm, shading='auto', rasterized=True, **kw)
    # outline of the 149x148 crop
    ring = [(lon[i0, j0:j0+148], lat[i0, j0:j0+148]), (lon[i0:i0+149, j0+147], lat[i0:i0+149, j0+147]),
            (lon[i0+148, j0:j0+148][::-1], lat[i0+148, j0:j0+148][::-1]), (lon[i0:i0+149, j0][::-1], lat[i0:i0+149, j0][::-1])]
    ax.plot(np.concatenate([r[0] for r in ring]), np.concatenate([r[1] for r in ring]), color=INK, lw=1.2)
    ax.set_title(title, loc='left'); ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02); cb.set_label(label, color=INK)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

fig, ax = plt.subplots(3, 4, figsize=(21, 15))
for c, n in enumerate('uv'):
    name = {'u': 'Eastward velocity u', 'v': 'Northward velocity v'}[n]
    panel(ax[0, c*2], uv[DAY, c], f'{name}, day {DAY}', 'RdBu_r', 'm/s', True)
    panel(ax[0, c*2+1], uv[DAY+1, c], f'{name}, day {DAY+1}', 'RdBu_r', 'm/s', True)
    panel(ax[1, c*2], mu[c], f'{name}: temporal mean', 'RdBu_r', 'm/s', True)
    panel(ax[1, c*2+1], sd[c], f'{name}: temporal std', 'Blues', 'm/s')
panel(ax[2, 0], np.hypot(uv[DAY, 0], uv[DAY, 1]), f'Speed |u,v|, day {DAY}', 'Blues', 'm/s')
panel(ax[2, 1], np.hypot(sd[0], sd[1]), 'Combined velocity std', 'Blues', 'm/s')
panel(ax[2, 2], h, 'Bathymetry h', 'Blues', 'depth (m)')
ax[2, 3].pcolormesh(lon, lat, land.astype(float), cmap='Greys', shading='auto', rasterized=True)
ax[2, 3].set_title(f'Land mask (dark = land, {land.mean():.1%})', loc='left')
ax[2, 3].set_xlabel('Longitude (°E)'); ax[2, 3].set_ylabel('Latitude (°N)')
ax[2, 3].set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
ax[2, 3].set_xticks([113.5, 114.0, 114.5, 115.0, 115.5])
for s in ('top', 'right'): ax[2, 3].spines[s].set_visible(False)
fig.suptitle('GBA depth-averaged currents, 298×296 domain (black outline = 149×148 crop)', x=0.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f'{OUT}/gba_dataset_overview.png', dpi=110); plt.close(fig)

# time series: u at three ocean points, 400 days
fig, ax = plt.subplots(figsize=(13, 3.8))
pts = [(i0 + 75, j0 + 74, 'Crop center'), (40, 40, 'Domain SW'), (250, 250, 'Domain NE')]
cols = ['#1f4e9c', '#d9822b', '#4d9a5b']
for (i, j, l), c in zip(pts, cols):
    if land[i, j]: continue
    ax.plot(uv[:400, 0, i, j], color=c, lw=1.6, label=f'{l} ({lon[i, j]:.2f}°E, {lat[i, j]:.2f}°N)')
ax.set_xlabel('Day index'); ax.set_ylabel('u (m/s)'); ax.set_title('Eastward velocity u, first 400 days', loc='left')
ax.grid(alpha=0.25, lw=0.6); ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.22))
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.savefig(f'{OUT}/gba_timeseries.png', dpi=110, bbox_inches='tight'); plt.close(fig)
print('crop offset', i0, j0)
