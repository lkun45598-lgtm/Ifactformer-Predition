"""单个窗口的逐日相对 L2 曲线 + 相邻两天的误差增量(检查是否有突变)。
用法: python plot_cmems_window.py OUT.png  (读 results_cmems_remote/ 下 K2_ftA 与 K1_res eval npz 中保存的同一窗口的预报场)"""
import sys, datetime as dt
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else 'cmems_window_curve.png'
RR = 'results_cmems_remote/'
ft = np.load(RR + 'K2_ftA/eval_val_ftA.npz'); base = np.load(RR + 'K1_res/eval_val_epoch5ckpt.npz')
T, V, Lst, D = ft['sample_true'], ft['sample_valid'], ft['sample_last'], ft['sample_dates']
PF, PB = ft['sample_pred'], base['sample_pred']; S = len(T); days = np.arange(1, S + 1)
d0 = dt.date.fromisoformat(str(D[0])); in0, in1 = d0 - dt.timedelta(days=7), d0 - dt.timedelta(days=1)


def rel(F, i):
    m = V[i]; d = (F - T[i]) * m
    return np.sqrt((d ** 2).sum((0, 1))) / np.sqrt(((T[i] * m) ** 2).sum((0, 1)))      # (u, v)


R = {n: np.stack([rel(F if F.ndim == 3 else F[i], i) for i in range(S)]) for n, F in (('ft', PF), ('base', PB), ('per', Lst))}    # (S,2)
M = {n: r.mean(1) for n, r in R.items()}; INC = {n: np.diff(m, prepend=np.nan) for n, m in M.items()}
INK, MUTED = '#222222', '#666666'; C_FT, C_BASE, C_PER = '#2a78d6', '#8fb8ea', '#eb6834'
plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})

fig = plt.figure(figsize=(16, 11.5))
ax = fig.add_axes([0.215, 0.50, 0.765, 0.38])
ax.plot(days, M['per'], color=C_PER, lw=2.2, label='Persistence (last observed day held fixed)')
ax.plot(days, M['base'], color=C_BASE, lw=2.3, ls='--', label='Single-step model, epoch 5 (fine-tuning starting point)')
ax.plot(days, R['ft'][:, 0], color=C_FT, lw=1.1, ls=':', alpha=0.85, label='Fine-tuned model, u component')
ax.plot(days, R['ft'][:, 1], color=C_FT, lw=1.1, ls='-.', alpha=0.85, label='Fine-tuned model, v component')
ax.plot(days, M['ft'], color=C_FT, lw=2.8, marker='o', ms=5, label='Multi-step fine-tuned model (K=2), mean of u and v')
for d, v in zip(days, M['ft']): ax.annotate(f'{v:.3f}', (d, v), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=8.8, color='#1d5bab', fontweight='bold')
ax.set_xlim(0.5, S + 0.5); ax.set_ylim(0, 1.0); ax.set_xticks(days); ax.tick_params(labelbottom=False); ax.grid(alpha=0.22, lw=0.6)
ax.set_ylabel('Relative L2 error of the forecast\n‖forecast − truth‖₂ / ‖truth‖₂   (lower is better)')
ax.legend(frameon=False, loc='lower right', fontsize=9.5)
for s in ('top', 'right'): ax.spines[s].set_visible(False)

ax2 = fig.add_axes([0.215, 0.315, 0.765, 0.14], sharex=ax)
w = 0.27
ax2.bar(days - w, np.nan_to_num(INC['per']), w, color=C_PER, alpha=0.85, label='Persistence'); ax2.bar(days, np.nan_to_num(INC['base']), w, color=C_BASE, label='Single-step (epoch 5)')
ax2.bar(days + w, np.nan_to_num(INC['ft']), w, color=C_FT, alpha=0.9, label='Fine-tuned (K=2)')
ax2.axhline(0, color=MUTED, lw=0.8); ax2.set_xticks(days); ax2.set_xlabel('Forecast day', labelpad=4); ax2.grid(axis='y', alpha=0.22, lw=0.6)
ax2.set_ylabel('Day-to-day change in\nrelative L2 (day k − day k−1)'); ax2.legend(frameon=False, loc='upper right', fontsize=9, ncol=3)
for s in ('top', 'right'): ax2.spines[s].set_visible(False)
k = int(np.nanargmax(INC['ft'][1:])) + 2

rows = [('Fine-tuned (K=2): rel. L2, mean of u and v', M['ft'], '{:.3f}'), ('Single-step (epoch 5): rel. L2', M['base'], '{:.3f}'), ('Persistence: rel. L2', M['per'], '{:.3f}'),
        ('Fine-tuned: change vs. previous day', INC['ft'], '{:+.3f}')]
tax = fig.add_axes([0.215, 0.045, 0.765, 0.16]); tax.axis('off')
tb = tax.table(cellText=[[('–' if np.isnan(x) else fmt.format(x)) for x in v] for _, v, fmt in rows], rowLabels=[n for n, _, _ in rows], colLabels=[str(d) for d in days], bbox=[0, 0, 1, 1], cellLoc='center')
tb.auto_set_font_size(False); tb.set_fontsize(9.3)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor('#dcdcda')
    if r == 0 or c == -1: cell.set_text_props(color=INK, fontweight='bold'); cell.set_facecolor('#f3f3f1')
    if r == 1 and c >= 0: cell.set_facecolor('#e8f0fb')
fig.suptitle(f'Single validation window: relative L2 error of each forecast day (first target day {d0})', x=0.012, y=0.975, ha='left', fontsize=15, color=INK)
fig.text(0.012, 0.935, f'Initial state: 7 observed daily means {in0} to {in1}.  20-day autoregressive forecast.  Largest one-day increase of the fine-tuned model: '
         f'+{np.nanmax(INC["ft"][1:]):.3f} (day {k - 1} → day {k}).', fontsize=11, color=MUTED)
fig.text(0.012, 0.006, 'Relative L2 = ‖forecast − truth‖₂ / ‖truth‖₂ over valid ocean cells of this single window, per component (u, v); bold line and table use the mean of u and v.\n'
         'Lower panel: change of the mean relative L2 from the previous forecast day; a sudden jump would appear as one bar much taller than its neighbours.', fontsize=8.8, color=MUTED)
fig.savefig(out, dpi=110); plt.close(fig)
inc = INC['ft'][1:]
print('fine-tuned increments: min %.4f  max %.4f (day %d->%d)  median %.4f' % (inc.min(), inc.max(), k - 1, k, np.median(inc)))
print('increments by model (days 2..20) max:', {n: round(float(np.nanmax(INC[n][1:])), 4) for n in INC})
print('day-1 rel L2 u,v ft:', R['ft'][0].round(3), '| day-20:', R['ft'][-1].round(3))
