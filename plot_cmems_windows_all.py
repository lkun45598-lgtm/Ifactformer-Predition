"""全部评估窗口的逐日相对 L2：每个窗口一条曲线 + 中位数/分位带；相邻两天误差增量的分布(检查突变)。
用法: python plot_cmems_windows_all.py OUT.png   (读 results_cmems_remote/K2_ftA/eval_val_ftA_pw.npz 与 K1_res/eval_val_base5_pw.npz)"""
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else 'cmems_windows_all.png'
RR = 'results_cmems_remote/'
ft = np.load(RR + 'K2_ftA/eval_val_ftA_pw.npz'); base = np.load(RR + 'K1_res/eval_val_base5_pw.npz')
A = ft['model_rel_perwin'].mean(2); B = base['model_rel_perwin'].mean(2); P = ft['persist_rel_perwin'].mean(2)        # (N,S)
N, S = A.shape; days = np.arange(1, S + 1); wd = ft['win_first_target_date']
INC = np.diff(A, axis=1)                                                                                         # (N,S-1) 微调模型相邻两天增量
INK, MUTED = '#222222', '#666666'; C_FT, C_BASE, C_PER = '#2a78d6', '#8fb8ea', '#eb6834'
plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})

fig = plt.figure(figsize=(16, 12.5))
ax = fig.add_axes([0.215, 0.575, 0.765, 0.32])
for i in range(N): ax.plot(days, A[i], color=C_FT, lw=0.9, alpha=0.28)
ax.fill_between(days, np.percentile(A, 10, axis=0), np.percentile(A, 90, axis=0), color=C_FT, alpha=0.16, lw=0, label='Fine-tuned model, 10th–90th percentile over windows')
ax.plot(days, np.median(P, axis=0), color=C_PER, lw=2.2, label='Persistence, median over windows')
ax.plot(days, np.median(B, axis=0), color=C_BASE, lw=2.2, ls='--', label='Single-step model (epoch 5), median over windows')
ax.plot(days, np.median(A, axis=0), color=C_FT, lw=2.8, marker='o', ms=4.5, label=f'Fine-tuned model (K=2), median over windows;  thin lines: each of the {N} windows')
ax.set_xlim(0.5, S + 0.5); ax.set_ylim(0, 1.2); ax.set_xticks(days); ax.tick_params(labelbottom=False); ax.grid(alpha=0.22, lw=0.6)
ax.set_ylabel('Relative L2 error of the forecast\n‖forecast − truth‖₂ / ‖truth‖₂   (mean of u, v)'); ax.legend(frameon=False, loc='upper left', fontsize=9.5)
for s in ('top', 'right'): ax.spines[s].set_visible(False)

ax2 = fig.add_axes([0.215, 0.335, 0.765, 0.19], sharex=ax)
bp = ax2.boxplot([INC[:, k] for k in range(S - 1)], positions=days[1:], widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                 medianprops=dict(color=INK, lw=1.4), boxprops=dict(facecolor='#cfe0f6', edgecolor=C_FT), whiskerprops=dict(color=C_FT), capprops=dict(color=C_FT))
rng = np.random.default_rng(0)
for k in range(S - 1): ax2.plot(days[k + 1] + rng.uniform(-0.2, 0.2, N), INC[:, k], '.', color=C_FT, ms=3, alpha=0.5)
ax2.axhline(0, color=MUTED, lw=0.8); ax2.set_xticks(days); ax2.set_xticklabels([str(d) for d in days]);   # boxplot 自带 19 个标签, 需重设以免错位
ax2.set_xlabel('Forecast day', labelpad=4); ax2.grid(axis='y', alpha=0.22, lw=0.6)
ax2.set_ylabel('Day-to-day change in\nrelative L2 (day k − day k−1)')
ax2.text(0.995, 0.97, 'Each dot = one window; box position k = change from day k−1 to day k', transform=ax2.transAxes, ha='right', va='top', fontsize=9, color=MUTED)
for s in ('top', 'right'): ax2.spines[s].set_visible(False)

rows = [('Median over windows', np.median(A, axis=0)), ('10th percentile', np.percentile(A, 10, axis=0)), ('90th percentile', np.percentile(A, 90, axis=0)),
        ('Minimum over windows', A.min(0)), ('Maximum over windows', A.max(0))]
tax = fig.add_axes([0.215, 0.045, 0.765, 0.17]); tax.axis('off')
fig.text(0.215, 0.222, f'Fine-tuned model (K=2): relative L2 of each forecast day across the {N} windows', fontsize=10.5, color=INK)
tb = tax.table(cellText=[[f'{x:.3f}' for x in v] for _, v in rows], rowLabels=[n for n, _ in rows], colLabels=[str(d) for d in days], bbox=[0, 0, 1, 1], cellLoc='center')
tb.auto_set_font_size(False); tb.set_fontsize(9.3)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor('#dcdcda')
    if r == 0 or c == -1: cell.set_text_props(color=INK, fontweight='bold'); cell.set_facecolor('#f3f3f1')
    if r == 1 and c >= 0: cell.set_facecolor('#e8f0fb')
iw, ik = np.unravel_index(np.argmax(INC), INC.shape)
fig.suptitle(f'Relative L2 error of each forecast day for all {N} validation windows: is there an abrupt jump?', x=0.012, y=0.975, ha='left', fontsize=15, color=INK)
fig.text(0.012, 0.935, f'Fine-tuned model (K=2).  Largest one-day increase in any window: +{INC.max():.3f} (window starting {wd[iw]}, day {ik + 1} → {ik + 2});  median one-day increase over all windows and days: {np.median(INC):+.3f}.\n'
         f'Windows whose error decreases from one day to the next at least once: {int((INC < 0).any(1).sum())} of {N}.', fontsize=10.5, color=MUTED, va='top')
fig.text(0.012, 0.006, 'Relative L2 = ‖forecast − truth‖₂ / ‖truth‖₂ over valid ocean cells, per window, mean of u and v. Validation windows: every 20th window of the validation period.\n'
         'Box plots: interquartile range (box), median (line), 5th–95th percentile (whiskers); dots: individual windows. A sudden jump would appear as dots far above the box at some day.', fontsize=8.8, color=MUTED)
fig.savefig(out, dpi=110); plt.close(fig)

d1 = INC[:, 0]; late = INC[:, 1:]
print('N windows', N, '| 单日最大增量', round(float(INC.max()), 4), '窗口', wd[iw], f'day {ik+1}->{ik+2}', '| 中位数', round(float(np.median(INC)), 4))
print('第1->2天增量: 中位 %.3f 最大 %.3f | 第2天起所有增量: 中位 %.3f 95分位 %.3f 最大 %.3f' % (np.median(d1), d1.max(), np.median(late), np.percentile(late, 95), late.max()))
print('出现误差下降(增量<0)的窗口数', int((INC < 0).any(1).sum()), '| 增量<0 的(窗口,天)对数', int((INC < 0).sum()), '/', INC.size)
top = np.dstack(np.unravel_index(np.argsort(INC, axis=None)[::-1][:6], INC.shape))[0]
print('最大的 6 个单日增量:', [(str(wd[i]), f'd{k+1}->d{k+2}', round(float(INC[i, k]), 3)) for i, k in top])
print('第20天误差: 中位 %.3f 最小 %.3f 最大 %.3f | 第20天 persistence 中位 %.3f' % (np.median(A[:, -1]), A[:, -1].min(), A[:, -1].max(), np.median(P[:, -1])))
print('微调模型在第20天 优于 persistence 的窗口数:', int((A[:, -1] < P[:, -1]).sum()), '/', N, '| 第10天:', int((A[:, 9] < P[:, 9]).sum()), '/', N)

big = np.argwhere(INC[:, 1:] > 0.15)
print('第2->3天起(不含第1->2天) 单日增量>0.15 的次数:', len(big), '| >0.10 的次数:', int((INC[:, 1:] > 0.10).sum()), '/', INC[:, 1:].size)
print('其中:', [(str(wd[i]), f'd{k+2}->d{k+3}', round(float(INC[i, k+1]), 3)) for i, k in big])
