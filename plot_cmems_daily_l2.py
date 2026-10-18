"""逐日相对 L2 折线图 + 逐日数值表 (Cmems, 20-day rollout)。用法: python plot_cmems_daily_l2.py"""
import csv
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results_cmems_remote/K1_res/'
RUNS = [('epoch 1 ckpt (val 0.1922)', 'eval_val_epoch1ckpt.npz', '#a9c8ef', ':'),
        ('epoch 3 ckpt (val 0.1820)', 'eval_val_epoch3ckpt.npz', '#6aa3e3', '--'),
        ('epoch 4 ckpt (val 0.1762)', 'eval_val_epoch4ckpt.npz', '#2a78d6', '-')]
INK, MUTED = '#222222', '#666666'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
E = {n: np.load(R + f) for n, f, _, _ in RUNS}
cur = E[RUNS[-1][0]]; days = np.arange(1, 21)
mod = cur['model_rel'].mean(1); per = cur['persist_rel'].mean(1); cli = cur['clim_rel'].mean(1)

fig = plt.figure(figsize=(17, 9.6)); ax = fig.add_axes([0.14, 0.40, 0.84, 0.52])
for n, f, c, ls in RUNS[:-1]: ax.plot(days, E[n]['model_rel'].mean(1), color=c, ls=ls, lw=1.8, label=f'Model, {n}')
ax.plot(days, per, color='#eb6834', lw=2.2, label='Persistence (last observed day held fixed)')
ax.plot(days, cli, color='#6250d6', lw=2.2, label='Climatology (train, by day of year)')
ax.plot(days, mod, color='#2a78d6', lw=2.6, marker='o', ms=5, label=f'Model, {RUNS[-1][0]}')
for d, v, p in zip(days, mod, per):
    ax.annotate(f'{v:.3f}', (d, v), textcoords='offset points', xytext=(0, -15 if v < p else 9), ha='center', fontsize=8.5, color='#1d5bab')
ax.set_xlim(0.5, 20.5); ax.set_ylim(0, 1.0); ax.set_xticks(days); ax.set_xlabel('Forecast day'); ax.set_ylabel('Relative L2 error  ‖pred − truth‖ / ‖truth‖')
ax.grid(alpha=0.25, lw=0.6); ax.legend(frameon=False, loc='lower right', fontsize=9.5)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_title('Cmems (Northern Gulf of Mexico): relative L2 error of each forecast day, 20-day autoregressive rollout (validation, mean of u and v)', loc='left', fontsize=12, color=INK)

rows = [('Model (epoch 4)', mod), ('Model u', cur['model_rel'][:, 0]), ('Model v', cur['model_rel'][:, 1]), ('Persistence', per), ('Climatology', cli)]
cells = [[f'{x:.3f}' for x in r[1]] for r in rows]
tax = fig.add_axes([0.14, 0.03, 0.84, 0.28]); tax.axis('off')
tb = tax.table(cellText=cells, rowLabels=[r[0] for r in rows], colLabels=[f'D{d}' for d in days], loc='center', cellLoc='center')
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.55)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor('#dddddd')
    if r == 0 or c == -1: cell.set_text_props(color=INK, fontweight='bold'); cell.set_facecolor('#f3f3f1')
    if r == 1 and c >= 0: cell.set_facecolor('#e8f0fb')
fig.text(0.06, 0.005, 'Relative L2 is computed over valid ocean cells for each window, then averaged over 51 validation windows (every 20th). '
         'Model trained for 1-step prediction only, then rolled out autoregressively. Lower is better.', fontsize=8.5, color=MUTED)
fig.savefig('cmems_daily_l2.png', dpi=110); plt.close(fig)

with open('cmems_daily_l2.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['day', 'model_epoch4_mean', 'model_epoch4_u', 'model_epoch4_v', 'persistence_mean', 'climatology_mean',
                                   'model_epoch3_mean', 'model_epoch1_mean'])
    for i, d in enumerate(days):
        w.writerow([d] + [f'{x:.4f}' for x in (mod[i], cur['model_rel'][i, 0], cur['model_rel'][i, 1], per[i], cli[i],
                                               E[RUNS[1][0]]['model_rel'][i].mean(), E[RUNS[0][0]]['model_rel'][i].mean())])
print('ok')
