"""汇报用图：逐日误差曲线+数值表、val/test 与 ACC、51 窗口逐日分布。输出 report_*.png / report_leadcurve.csv"""
import csv, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = 'results_cmems_remote/'; INK, MUTED = '#222222', '#666666'
BLUE, LBLUE, ORANGE, PURPLE, GRAY, GREEN = '#2a78d6', '#8fb8ea', '#eb6834', '#6250d6', '#9a9a9a', '#2f9e6b'
plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
L = lambda p: np.load(R + p); days = np.arange(1, 21)
v = {'K1 ep5': L('K1_res/eval_val_base5_pw.npz'), 'K1 ep11': L('K1_res_cont/eval_val_k1_cont.npz'), 'A': L('K2_ftA_full/eval_val.npz'), 'B': L('K4_ftB_full/eval_val_ftB_full.npz')}
rel = lambda e, k='model_rel': e[k].mean(1)
for s in ('top', 'right'): plt.rcParams['axes.spines.' + s] = False

# ---- 图 1：逐日相对 L2（val）+ 数值表 ----
f = {k: rel(e) for k, e in v.items()}; p = rel(v['B'], 'persist_rel'); c = rel(v['B'], 'clim_rel')
fig = plt.figure(figsize=(15, 10.5)); ax = fig.add_axes([0.2, 0.42, 0.78, 0.46])
ax.plot(days, c, color=PURPLE, lw=2, label='Climatology (train mean by day of year)')
ax.plot(days, p, color=ORANGE, lw=2.2, label='Persistence (last observed day held fixed)')
ax.plot(days, f['K1 ep5'], color=GRAY, lw=1.8, ls=':', label='Single-step K=1, epoch 5 (early stop)')
ax.plot(days, f['K1 ep11'], color=LBLUE, lw=2.4, ls='--', label='Single-step K=1, 12 epochs')
ax.plot(days, f['A'], color=GREEN, lw=2.2, label='Fine-tuned stage A (K=2, 8 epochs)')
ax.plot(days, f['B'], color=BLUE, lw=2.8, marker='o', ms=5, label='Fine-tuned stage B (K=4, 6 epochs) — final')
for d in (1, 3, 5, 7, 10, 15, 20): ax.annotate(f"{f['B'][d-1]:.3f}", (d, f['B'][d-1]), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9.5, color='#1d5bab', fontweight='bold')
ax.set(xlim=(0.5, 20.5), ylim=(0, 1.0), xticks=days, ylabel='Relative L2 error  ‖forecast − truth‖₂ / ‖truth‖₂\n(lower is better)'); ax.tick_params(labelbottom=False)
ax.grid(alpha=.22); ax.legend(frameon=False, loc='lower right', fontsize=10)
rows = [('Final model (stage B, K=4)', f['B'], '{:.3f}'), ('Stage A (K=2)', f['A'], '{:.3f}'), ('Single-step (K=1, 12 ep.)', f['K1 ep11'], '{:.3f}'), ('Persistence', p, '{:.3f}'), ('Climatology', c, '{:.3f}'),
        ('Reduction vs persistence (%)', 100 * (p - f['B']) / p, '{:.1f}'), ('Reduction vs single-step (%)', 100 * (f['K1 ep11'] - f['B']) / f['K1 ep11'], '{:.1f}')]
tax = fig.add_axes([0.2, 0.04, 0.78, 0.3]); tax.axis('off')
tb = tax.table(cellText=[[fm.format(x) for x in val] for _, val, fm in rows], rowLabels=[n for n, _, _ in rows], colLabels=[str(d) for d in days], bbox=[0, 0, 1, 1], cellLoc='center')
tb.auto_set_font_size(False); tb.set_fontsize(9)
for (r, col), cell in tb.get_celld().items():
    cell.set_edgecolor('#dcdcda')
    if r == 0 or col == -1: cell.set_text_props(color=INK, fontweight='bold'); cell.set_facecolor('#f3f3f1')
    if r == 1 and col >= 0: cell.set_facecolor('#e8f0fb')
tax.text(0.5, 1.07, 'Forecast day', ha='center', fontsize=10.5, color=INK, transform=tax.transAxes)
fig.suptitle('20-day autoregressive forecast error by lead time — Northern Gulf of Mexico surface currents (validation set)', x=0.012, y=0.975, ha='left', fontsize=14, color=INK)
fig.text(0.012, 0.93, f"20-day mean relative L2:  final {f['B'].mean():.3f}  |  stage A {f['A'].mean():.3f}  |  single-step {f['K1 ep11'].mean():.3f}  |  persistence {p.mean():.3f}  |  climatology {c.mean():.3f}", fontsize=11, color=MUTED)
fig.text(0.012, 0.006, 'Relative L2 per window and per component (u, v) over valid ocean cells, averaged over u, v and 51 validation windows (every 20th). Input: 7 observed daily means; output: 20-day autoregressive rollout.', fontsize=8.8, color=MUTED)
fig.savefig('report_leadcurve.png', dpi=110); plt.close(fig)
with open('report_leadcurve.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['day', 'final_B', 'stageA', 'single_step_ep11', 'single_step_ep5', 'persistence', 'climatology'])
    for i, d in enumerate(days): w.writerow([d] + [f'{x[i]:.4f}' for x in (f['B'], f['A'], f['K1 ep11'], f['K1 ep5'], p, c)])

# ---- 图 2：ACC 与 val/test（final vs single-step vs persistence） ----
tpath = R + 'K4_ftB_full/eval_test.npz'; t1 = R + 'K1_res_cont/eval_test.npz'; have_t = os.path.exists(tpath) and os.path.exists(t1)
fig, ax = plt.subplots(1, 3 if have_t else 2, figsize=(18 if have_t else 12, 5.2))
acc = lambda e, k='model_acc': e[k].mean(1)
ax[0].plot(days, acc(v['B'], 'persist_acc'), color=ORANGE, lw=2, label='Persistence'); ax[0].plot(days, acc(v['K1 ep11']), color=LBLUE, lw=2.2, ls='--', label='Single-step K=1')
ax[0].plot(days, acc(v['A']), color=GREEN, lw=2, label='Stage A (K=2)'); ax[0].plot(days, acc(v['B']), color=BLUE, lw=2.6, marker='o', ms=4, label='Final (stage B, K=4)')
for d in (1, 5, 10, 20): ax[0].annotate(f"{acc(v['B'])[d-1]:.3f}", (d, acc(v['B'])[d-1]), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=9, color='#1d5bab')
ax[0].set(title='Anomaly correlation coefficient (ACC), validation set\nrelative to train day-of-year climatology', xlabel='Forecast day', ylabel='ACC (higher is better)', ylim=(0, 1)); ax[0].grid(alpha=.25); ax[0].legend(frameon=False)
rm = lambda e: e['model_rmse'].mean(1)
ax[1].plot(days, v['B']['persist_rmse'].mean(1), color=ORANGE, lw=2, label='Persistence'); ax[1].plot(days, rm(v['K1 ep11']), color=LBLUE, lw=2.2, ls='--', label='Single-step K=1')
ax[1].plot(days, rm(v['B']), color=BLUE, lw=2.6, marker='o', ms=4, label='Final (stage B, K=4)')
for d in (1, 10, 20): ax[1].annotate(f"{rm(v['B'])[d-1]:.3f}", (d, rm(v['B'])[d-1]), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=9, color='#1d5bab')
ax[1].set(title='RMSE of u and v components, validation set', xlabel='Forecast day', ylabel='RMSE (m/s)'); ax[1].grid(alpha=.25); ax[1].legend(frameon=False)
if have_t:
    tB, tK, = np.load(tpath), np.load(t1); tp = rel(tB, 'persist_rel'); tm = rel(tB)
    ax[2].plot(days, tp, color=ORANGE, lw=2, label=f'Persistence (mean {tp.mean():.3f})'); ax[2].plot(days, rel(tK), color=LBLUE, lw=2.2, ls='--', label=f'Single-step K=1 (mean {rel(tK).mean():.3f})')
    ax[2].plot(days, tm, color=BLUE, lw=2.6, marker='o', ms=4, label=f'Final (stage B, K=4) (mean {tm.mean():.3f})'); ax[2].plot(days, rel(tB, 'clim_rel'), color=PURPLE, lw=1.6, label='Climatology')
    for d in (1, 10, 20): ax[2].annotate(f'{tm[d-1]:.3f}', (d, tm[d-1]), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color='#1d5bab')
    ax[2].set(title='Relative L2 on the held-out TEST set\n(2018-08-25 to 2021-06-30, 51 windows)', xlabel='Forecast day', ylabel='Relative L2 (lower is better)', ylim=(0, 1)); ax[2].grid(alpha=.25); ax[2].legend(frameon=False, loc='lower right')
for a_ in ax: a_.set_xticks(days[::2]); a_.set_xlim(0.5, 20.5)
plt.tight_layout(); plt.savefig('report_acc_test.png', dpi=130); plt.close(fig)

# ---- 图 3：51 个窗口的逐日分布（final, val） ----
mw = v['B']['model_rel_perwin'].mean(2); pw = v['B']['persist_rel_perwin'].mean(2)       # (51,20)
fig, ax = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={'width_ratios': [1.6, 1]})
bp = ax[0].boxplot([mw[:, i] for i in range(20)], positions=days, widths=.6, patch_artist=True, showfliers=False)
for b in bp['boxes']: b.set(facecolor='#cfe0f6', edgecolor=BLUE)
for m_ in bp['medians']: m_.set(color=BLUE, lw=2)
ax[0].plot(days, np.median(pw, 0), color=ORANGE, lw=2, label='Persistence (median over windows)'); ax[0].plot(days, np.median(mw, 0), color=BLUE, lw=0, label='Final model: box = IQR, line = median')
ax[0].set(title='Spread of forecast error across 51 validation windows (final model)', xlabel='Forecast day', ylabel='Relative L2 (u, v mean)', xticks=days, ylim=(0, 1.3)); ax[0].grid(alpha=.25); ax[0].legend(frameon=False, loc='upper left')
frac = (mw < pw).mean(0) * 100
ax[1].bar(days, frac, color=BLUE, alpha=.85); ax[1].axhline(50, color=MUTED, ls=':', lw=1)
for d in (1, 5, 10, 15, 20): ax[1].text(d, frac[d-1] + 1.5, f'{frac[d-1]:.0f}%', ha='center', fontsize=9, color=INK)
ax[1].set(title='Share of windows where the model beats persistence', xlabel='Forecast day', ylabel='Windows (%)', ylim=(0, 112), xticks=days[::2]); ax[1].grid(axis='y', alpha=.25)
plt.tight_layout(); plt.savefig('report_windows.png', dpi=130); plt.close(fig)
print('ok', 'test' if have_t else 'val only', f"B mean {f['B'].mean():.3f} A {f['A'].mean():.3f} K1 {f['K1 ep11'].mean():.3f}")
