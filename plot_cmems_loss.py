import re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = 'results_cmems_remote'; BLUE, ORANGE, PURPLE = '#2a78d6', '#eb6834', '#6250d6'
def ep(*runs):
    return np.array([re.findall(r'train ([\d.]+) \| val ([\d.]+)', l)[0] for r in runs for l in open(f'{R}/{r}/training_epoch_log.txt') if l.startswith('Epoch')], float)
k1, a, b = ep('K1_res', 'K1_res_cont'), ep('K2_ftA_full'), ep('K4_ftB_full')
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
def panel(ax, d, title, refs=()):
    x = np.arange(len(d))
    ax.plot(x, d[:, 0], 'o-', c=BLUE, label='Train loss'); ax.plot(x, d[:, 1], 's-', c=ORANGE, label='Validation loss')
    for i in x:
        ax.annotate(f'{d[i,0]:.4f}', (i, d[i, 0]), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=6.5, color=BLUE)
        ax.annotate(f'{d[i,1]:.4f}', (i, d[i, 1]), textcoords='offset points', xytext=(0, -12), ha='center', fontsize=6.5, color=ORANGE)
    for v, t, c, ls in refs: ax.axhline(v, c=c, ls=ls, lw=1); ax.text(x[-1], v, t, ha='right', va='bottom', fontsize=8, color=c)
    ax.set(title=title, xlabel='Epoch', ylabel='Loss (relative L2)'); ax.set_xticks(x); ax.legend(loc='upper right' if len(d) < 10 else 'center right'); ax.grid(alpha=.3)
    ax.margins(y=.12)
panel(axs[0], k1, f'Stage 0: single-step (K=1), 12 epochs\nbest val {k1[:,1].min():.4f}', [(0.251, 'Persistence 0.251', 'gray', '--')])
panel(axs[1], a, f'Stage A: fine-tune K=2, 8 epochs\nloss = mean of 2 rollout steps; best val {a[:,1].min():.4f}')
panel(axs[2], b, f'Stage B: fine-tune K=4, 6 epochs\nloss = mean of 4 rollout steps; best val {b[:,1].min():.4f}')
fig.suptitle('Training loss per stage (relative L2, u/v mean; losses across stages use different K and are not comparable)', fontsize=10)
plt.tight_layout(); plt.savefig('cmems_loss_k1.png', dpi=140); print(len(k1), len(a), len(b))
