import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import os
import glob # 导入glob库用于查找文件

# --- 科研绘图风格设置 (保持不变) ---
plt.style.use('seaborn-v0_8-paper')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=False)
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

# --- calculate_relative_l2_error 函数 (保持不变) ---
def calculate_relative_l2_error(true_data, pred_data):
    """计算相对L2误差 - 分别计算每个通道"""
    assert true_data.shape == pred_data.shape, "数据形状不一致"
    time_steps = true_data.shape[0]
    num_channels = true_data.shape[-1]
    relative_errors = np.zeros((time_steps, num_channels))
    for t in range(time_steps):
        for c in range(num_channels):
            error_norm = np.linalg.norm(true_data[t, :, :, c] - pred_data[t, :, :, c])
            true_norm = np.linalg.norm(true_data[t, :, :, c])
            relative_errors[t, c] = error_norm / true_norm if true_norm > 0 else 0
    relative_errors_total = np.zeros(time_steps)
    for t in range(time_steps):
        error_norm = np.linalg.norm(true_data[t] - pred_data[t])
        true_norm = np.linalg.norm(true_data[t])
        relative_errors_total[t] = error_norm / true_norm if true_norm > 0 else 0
    return relative_errors, relative_errors_total

# --- plot_relative_l2_error 函数 (保持不变) ---
# 注意：函数定义中的默认路径不再重要，因为我们会从main函数传入正确的路径
def plot_relative_l2_error(result_dir):
    """绘制相对L2误差曲线 - 分别显示每个通道"""
    npy_dir = os.path.join(result_dir, 'npy')
    true_data = np.load(os.path.join(npy_dir, 'label.npy'))[0]
    pred_data = np.load(os.path.join(npy_dir, 'predict.npy'))[0]
    print(f"真解数据形状: {true_data.shape}")
    print(f"模型解数据形状: {pred_data.shape}")
    relative_errors_channels, relative_errors_total = calculate_relative_l2_error(true_data, pred_data)
    fig, ax = plt.subplots(figsize=(12, 7))
    channel_names = ['temperature', 'v-velocity', 'u-velocity']
    colors = ['#E63946', '#06A77D', '#1D3557', '#F77F00']
    time_steps = np.arange(len(relative_errors_total))
    for c in range(3):
        ax.plot(time_steps, relative_errors_channels[:, c],
                color=colors[c], linewidth=2.0, alpha=0.8,
                label=f'{channel_names[c]} (channel {c})',
                marker='o', markersize=2, markevery=10)
    ax.plot(time_steps, relative_errors_total,
            color=colors[3], linewidth=2.5, alpha=0.9,
            label='Overall (all channels)',
            marker='s', markersize=3, markevery=10, linestyle='--')
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative L2 Error', fontsize=14, fontweight='bold')
    ax.set_title('Relative L2 Error vs Time Steps (Per Channel)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(relative_errors_total)-1)
    ax.set_ylim(0, max(np.max(relative_errors_channels), np.max(relative_errors_total)) * 1.1)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_path = os.path.join(result_dir, 'relative_l2_error.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    print(f"\n各通道相对L2误差统计:")
    for c in range(3):
        print(f"  {channel_names[c]}:")
        print(f"    平均: {np.mean(relative_errors_channels[:, c]):.6f}")
        print(f"    最大: {np.max(relative_errors_channels[:, c]):.6f}")
        print(f"    最小: {np.min(relative_errors_channels[:, c]):.6f}")
    print(f"\n总体相对L2误差:")
    print(f"  平均: {np.mean(relative_errors_total):.6f}")
    print(f"  最大: {np.max(relative_errors_total):.6f}")
    print(f"  最小: {np.min(relative_errors_total):.6f}")
    print(f"\n误差曲线已保存到: {save_path}")


# --- plot_2d_comparison 和 plot_single_component 函数 (保持不变) ---
def plot_2d_comparison(result_dir):
    """绘制2D对比图 - 为每个通道生成单独的图"""
    npy_dir = os.path.join(result_dir, 'npy')
    true_data = np.load(os.path.join(npy_dir, 'label.npy'))[0]
    pred_data = np.load(os.path.join(npy_dir, 'predict.npy'))[0]
    time_indices = [1,19,39,59,79,99]
    component_names = ['temperature', 'v-velocity', 'u-velocity']
    component_cmaps = ['viridis', 'RdBu_r', 'RdBu_r']
    for component_index in range(3):
        plot_single_component(true_data, pred_data, time_indices,
                            component_index, component_names[component_index],
                            component_cmaps[component_index], result_dir)

def plot_single_component(true_data, pred_data, time_indices, component_index,
                         component_name, cmap, result_dir):
    """绘制单个通道的2D对比图"""
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    vmin_data = min(true_data[:, :, :, component_index].min(), pred_data[:, :, :, component_index].min())
    vmax_data = max(true_data[:, :, :, component_index].max(), pred_data[:, :, :, component_index].max())
    for i, t_idx in enumerate(time_indices):
        ax = axes[0, i]
        true_slice = true_data[t_idx, :, :, component_index]
        im = ax.imshow(true_slice.T, cmap=cmap, origin='lower', aspect='auto', interpolation='bilinear', vmin=vmin_data, vmax=vmax_data)
        ax.set_title(f'Ground Truth\nt = {t_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        if i == 0: ax.set_ylabel('Y', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    for i, t_idx in enumerate(time_indices):
        ax = axes[1, i]
        pred_slice = pred_data[t_idx, :, :, component_index]
        im = ax.imshow(pred_slice.T, cmap=cmap, origin='lower', aspect='auto', interpolation='bilinear', vmin=vmin_data, vmax=vmax_data)
        ax.set_title(f'IFactFormer\nt = {t_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        if i == 0: ax.set_ylabel('Y', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    for i, t_idx in enumerate(time_indices):
        ax = axes[2, i]
        true_slice = true_data[t_idx, :, :, component_index]
        pred_slice = pred_data[t_idx, :, :, component_index]
        error_slice = np.abs(true_slice - pred_slice)
        im = ax.imshow(error_slice.T, cmap='hot', origin='lower', aspect='auto', interpolation='bilinear')
        rel_error = np.linalg.norm(error_slice) / np.linalg.norm(true_slice)
        ax.set_title(f'Absolute Error\nRL2 = {rel_error:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        if i == 0: ax.set_ylabel('Y', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('2D Flow Field Comparison: Ground Truth vs IFactFormer Prediction\n' +
                 f'Component: {component_name} (channel {component_index})',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_path = os.path.join(result_dir, f'2d_comparison_channel{component_index}_{component_name.replace("-", "_")}_best_epochs_long_step.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    print(f"  {component_name} 对比图已保存到: {save_path}")


# --- main 函数 (核心修改部分) ---
def main():
    """主函数"""
    print("="*60)
    print("开始数据可视化分析...")
    print("="*60)

    # 直接指定您想要可视化的结果目录
    result_dir = 'results/dim96_depth1_iter10_T1_5'
    
    print(f"将使用固定目录进行可视化: {result_dir}")

    # 检查数据文件是否存在
    npy_dir = os.path.join(result_dir, 'npy')
    if not os.path.exists(os.path.join(npy_dir, 'label.npy')):
        print(f"错误: 找不到 {os.path.join(npy_dir, 'label.npy')} 文件")
        return

    if not os.path.exists(os.path.join(npy_dir, 'predict.npy')):
        print(f"错误: 找不到 {os.path.join(npy_dir, 'predict.npy')} 文件")
        return

    # 任务1: 绘制相对L2误差曲线
    print("\n=== 任务1: 绘制相对L2误差曲线 ===")
    plot_relative_l2_error(result_dir)

    # 任务2: 绘制2D对比图
    print("\n=== 任务2: 绘制2D对比图（所有3个通道） ===")
    plot_2d_comparison(result_dir)
    print("\n" + "="*60)
    print("可视化完成！")
    print(f"生成的图像文件保存在: {result_dir}/")
    print("  - relative_l2_error.png")
    print("  - 2d_comparison_... .png")
    print("="*60)

if __name__ == "__main__":
    main()