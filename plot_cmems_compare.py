"""逐日相对 L2：微调前(单步, epoch 5) vs 微调后(阶段 A, K=2)。用法: python plot_cmems_compare.py OUTPREFIX
 读 results_cmems_remote/ 下的 eval npz；三层: 折线图 / 每天误差降低量柱状图 / 与横轴逐列对齐的数值表"""
import csv, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pre = sys.argv[1] if len(sys.argv) > 1 else 'cmems_compare'
R = 'results_cmems_remote/'
base = np.load(R + 'K1_res/eval_val_epoch5ckpt.npz'); ft = np.load(R + 'K2_ftA/eval_val_ftA.npz')
INK, MUTED = '#222222', '#666666'
C_FT, C_BASE, C_PER, C_CLI = '#2a78d6', '#8fb8ea', '#eb6834', '#6250d6'
plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
days = np.arange(1, 21)
m = lambda e: e['model_rel'].mean(1)
b, f, p, c = m(base), m(ft), ft['persist_rel'].mean(1), ft['clim_rel'].mean(1)
red = b - f; redp = 100 * red / b
SEL = [1, 3, 5, 7, 10, 13, 16, 20]

fig = plt.figure(figsize=(16, 11.5))
ax = fig.add_axes([0.215, 0.50, 0.765, 0.40])
ax.fill_between(days, f, b, color=C_FT, alpha=0.16, lw=0, label='Error removed by multi-step fine-tuning')
ax.plot(days, c, color=C_CLI, lw=2.2, label='Climatology (train-period mean by day of year)')
ax.plot(days, p, color=C_PER, lw=2.2, label='Persistence (last observed day held fixed)')
ax.plot(days, b, color=C_BASE, lw=2.4, ls='--', label='Single-step model, epoch 5 (fine-tuning starting point)')
ax.plot(days, f, color=C_FT, lw=2.8, marker='o', ms=5, label='Multi-step fine-tuned model (K=2 rollout loss)')
for d in SEL:
    ax.annotate(f'{f[d-1]:.3f}', (d, f[d-1]), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9.5, color='#1d5bab', fontweight='bold')
    ax.annotate(f'{b[d-1]:.3f}', (d, b[d-1]), textcoords='offset points', xytext=((-14, -15) if d == 20 else (0, 8)), ha='center', fontsize=9, color='#5b86bd')
ax.set_xlim(0.5, 20.5); ax.set_ylim(0, 1.0); ax.set_xticks(days); ax.tick_params(labelbottom=False)
ax.set_ylabel('Relative L2 error of the forecast\n‖forecast − truth‖₂ / ‖truth‖₂   (lower is better)'); ax.grid(alpha=0.22, lw=0.6)
ax.legend(frameon=False, loc='lower right', fontsize=10, ncol=1)
for s in ('top', 'right'): ax.spines[s].set_visible(False)

ax2 = fig.add_axes([0.215, 0.315, 0.765, 0.14], sharex=ax)
bars = ax2.bar(days, red, width=0.62, color=C_FT, alpha=0.85)
for d, v, pc in zip(days, red, redp): ax2.text(d, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, color=INK)
ax2.set_ylim(0, red.max() * 1.28); ax2.set_ylabel('Absolute reduction in\nrelative L2 (single-step − fine-tuned)'); ax2.grid(axis='y', alpha=0.22, lw=0.6)
ax2.set_xticks(days); ax2.set_xlabel('Forecast day', labelpad=4)
for s in ('top', 'right'): ax2.spines[s].set_visible(False)

rows = [('Fine-tuned model (K=2): rel. L2', f, '{:.3f}'), ('Single-step model (epoch 5): rel. L2', b, '{:.3f}'), ('Relative reduction (%)', redp, '{:.1f}'), ('Persistence: rel. L2', p, '{:.3f}'), ('Climatology: rel. L2', c, '{:.3f}')]
tax = fig.add_axes([0.215, 0.035, 0.765, 0.20]); tax.axis('off')
tb = tax.table(cellText=[[fmt.format(x) for x in v] for _, v, fmt in rows], rowLabels=[n for n, _, _ in rows], colLabels=[str(d) for d in days],
               bbox=[0, 0, 1, 1], cellLoc='center')
tb.auto_set_font_size(False); tb.set_fontsize(9.5)
for (r, col), cell in tb.get_celld().items():
    cell.set_edgecolor('#dcdcda')
    if r == 0 or col == -1: cell.set_text_props(color=INK, fontweight='bold'); cell.set_facecolor('#f3f3f1')
    if r == 1 and col >= 0: cell.set_facecolor('#e8f0fb')
    if r == 3 and col >= 0: cell.set_facecolor('#eaf5ec')
fig.suptitle('Relative L2 error of each forecast day, 20-day autoregressive rollout: single-step vs multi-step fine-tuned model (Northern Gulf of Mexico)', x=0.012, y=0.975, ha='left', fontsize=14, color=INK)
fig.text(0.012, 0.935, f'20-day mean relative L2:  fine-tuned {f.mean():.3f}   |   single-step {b.mean():.3f}   |   persistence {p.mean():.3f}   |   climatology {c.mean():.3f}', fontsize=11, color=MUTED)
fig.text(0.012, 0.006, 'Relative L2 = ‖forecast − truth‖₂ / ‖truth‖₂ over valid ocean cells, per window and per component (u, v), averaged over u, v and 51 validation windows (every 20th).\n'
         'Forecast: autoregressive 20-day rollout from 7 observed daily means.  Relative reduction = (single-step − fine-tuned) / single-step.  Fine-tuned weights selected on validation loss.', fontsize=8.8, color=MUTED)
fig.savefig(f'{pre}.png', dpi=110); plt.close(fig)

with open(f'{pre}.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['day', 'after_mean', 'after_u', 'after_v', 'before_mean', 'before_u', 'before_v', 'reduction', 'reduction_pct', 'persistence', 'climatology'])
    for i, d in enumerate(days):
        w.writerow([d] + [f'{x:.4f}' for x in (f[i], ft['model_rel'][i, 0], ft['model_rel'][i, 1], b[i], base['model_rel'][i, 0], base['model_rel'][i, 1], red[i], redp[i], p[i], c[i])])
print('ok')
