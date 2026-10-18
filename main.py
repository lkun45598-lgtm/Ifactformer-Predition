import os
from pathlib import Path

# 使用空闲的GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pickle
import yaml
import shutil
import torch
import numpy as np

from timeit import default_timer
from einops import rearrange
from torch.utils.data import TensorDataset, DataLoader
from libs.model import Model
from libs.utils import *

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(1234)

# -------------------------- parameters --------------------------
# read the parameters file
with open(os.path.join('configs', 'IFactFormer.yml'), 'r') as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data path - 使用真实的RB数据
data_path = Path(__file__).resolve().parent / "data" / "rb_Ra1e+05_bott_1_1_upt_0_0.npy"
if not data_path.exists():
    raise FileNotFoundError(f"{data_path} not found. Please ensure the data file is in the data folder.")

# create save_dirs
i = 0
save_dirs = Path(config.log_dir) / f'dim{config.model.dim}_depth{config.model.depth}_iter{config.model.n_layer}_T{config.model.in_time_window}_{i}'
while save_dirs.exists():
    i += 1
    save_dirs = Path(config.log_dir) / f'dim{config.model.dim}_depth{config.model.depth}_iter{config.model.n_layer}_T{config.model.in_time_window}_{i}'
save_dirs.mkdir(parents=True, exist_ok=False)
print(save_dirs)

# copy the parameters file
shutil.copyfile('configs/IFactFormer.yml', save_dirs / 'IFactFormer.yml')


# -------------------------- data_generation --------------------------
# load data with shape (B, T, X, Y, C)
data = np.load(data_path)
data = data[:-1, :, :, :, :3]  # 只取前两个通道
print(f"Original data shape: {data.shape}")

if data.ndim != 5:
    raise ValueError(f"Expected 5D data with shape (B, T, X, Y, C), got shape {data.shape}")

# 转换为torch tensor
data = torch.from_numpy(data).float()
print(f"Loaded dataset {data_path.name} with shape {tuple(data.shape)}")

input_list = []
output_list = []

b, t, nx, ny, c = data.shape
in_time_window, out_time_window = config.model.in_time_window, config.model.out_time_window
sample_num = t - (in_time_window + out_time_window)

print(f"Dataset info: B={b}, T={t}, X={nx}, Y={ny}, C={c}")
print(f"Time window: in={in_time_window}, out={out_time_window}")
print(f"Samples per trajectory: {sample_num}")

# 为每个batch和时间窗口创建样本
for i in range(b):
    for j in range(sample_num):
        input_list.append(data[i, j: j + in_time_window, ...])  # shape: (in_time_window, nx, ny, c)
        output_list.append(data[i, j + in_time_window: j + in_time_window + out_time_window, ...])  # shape: (out_time_window, nx, ny, c)

input_data = torch.stack(input_list)  # shape: (n_samples, in_time_window, nx, ny, c)
output_data = torch.stack(output_list)  # shape: (n_samples, out_time_window, nx, ny, c)

print(f"Input data shape: {tuple(input_data.shape)}")
print(f"Output data shape: {tuple(output_data.shape)}")

# 划分训练集和测试集
ntrain = int(0.8 * input_data.shape[0])
ntest = input_data.shape[0] - ntrain
train_x = input_data[:ntrain]
train_y = output_data[:ntrain]
test_x = input_data[ntrain:]
test_y = output_data[ntrain:]

print(f"Train samples: {ntrain}, Test samples: {ntest}")

# 计算归一化参数
norm = {}
norm['x_mean'] = torch.mean(train_x, dim=(0,1,2,3)).to(device)  # shape: (c,)
norm['x_std'] = torch.std(train_x, dim=(0,1,2,3)).to(device)    # shape: (c,)
norm['y_mean'] = torch.mean(train_y, dim=(0,1,2,3)).to(device)  # shape: (c,)
norm['y_std'] = torch.std(train_y, dim=(0,1,2,3)).to(device)    # shape: (c,)

# 计算能量变化范围
energy = torch.sum(data**2, dim=(2,3))  # shape: (b, t, c)
delta_energy = energy[:,1:,:] - energy[:,:-1,:]  # shape: (b, t-1, c)
delta_energy_max = torch.max(delta_energy.reshape(-1, c), dim=0).values.to(device)
delta_energy_min = torch.min(delta_energy.reshape(-1, c), dim=0).values.to(device)
norm['delta_energy_max'] = delta_energy_max
norm['delta_energy_min'] = delta_energy_min

# 保存归一化参数
with open(save_dirs / 'norm.pkl', 'wb') as f:
    pickle.dump(norm, f)

print(f"Normalization stats:")
print(f"  x_mean: {norm['x_mean'].cpu().numpy()}")
print(f"  x_std: {norm['x_std'].cpu().numpy()}")

# 位置编码 - 使用实际的空间尺寸
size_lst = [(nx, ny)]
# length是物理域的尺寸（归一化后），不是网格点数
# 对于Rayleigh-Bénard对流，通常是长宽比例，这里假设为3:1的比例
length = [3.0, 1.0]  # 根据实际物理域的长宽比调整
pos_lst = get_pos_lst(size_lst, length)[0]

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config.training.batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=config.training.batch_size, shuffle=False)


# -------------------------- define --------------------------
model = Model(config.model).to(device)

info = f'count_params: {count_params(model)}'
print(info)
with open(os.path.join(save_dirs, 'training_epoch_log.txt'), 'a') as file:
    file.write(info + '\n')

optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.scheduler_step, gamma=config.training.scheduler_gamma)

loss_fn = LpLoss(reduction=False)

train_loss_lst = []
test_loss_lst = []
min_test_loss = 1e10

# -------------------------- training --------------------------
print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

t0 = default_timer()
for ep in range(config.training.epochs):
    model.train()
    t1 = default_timer()
    train_losses = []

    for n_iter, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # 自回归多步预测并汇总loss
        steps = out_time_window
        x_cur = x  # 保持 5D 形状 (b, 1, nx, ny, c)
        total_loss = 0.0
        for k in range(steps):
            x_norm = (x_cur - norm['x_mean']) / norm['x_std']
            y_pred_k = model(x_norm, pos_lst) * norm['y_std'] + norm['y_mean']
            total_loss = total_loss + torch.mean(loss_fn(y_pred_k, y[:, k, ...]))
            # 下一个时刻的输入（保持时间维度）
            x_cur = y_pred_k.unsqueeze(1)

        train_loss = total_loss / steps
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        if n_iter % 10 == 0:
            print(f'Epoch {ep}, Iter {n_iter}/{len(train_loader)}, train_loss: {train_loss.item():.6f}')

    avg_train_loss = np.mean(train_losses)
    train_loss_lst.append(avg_train_loss)
    scheduler.step()

    # 测试（自回归多步）
    model.eval()
    with torch.no_grad():
        test_losses = []
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            steps = out_time_window
            x_cur = x
            total_loss = 0.0
            for k in range(steps):
                x_norm = (x_cur - norm['x_mean']) / norm['x_std']
                y_pred_k = model(x_norm, pos_lst) * norm['y_std'] + norm['y_mean']
                total_loss = total_loss + torch.mean(loss_fn(y_pred_k, y[:, k, ...]))
                x_cur = y_pred_k.unsqueeze(1)

            test_losses.append((total_loss / steps).item())

        L2 = torch.tensor(test_losses).mean()

    t2 = default_timer()

    info = (f'Epoch {ep:3d} | Time: {(t2-t1):.2f}s | Train L2: {avg_train_loss:.6f} | '
            f'Test L2: {L2.item():.6f} | LR: {optimizer.param_groups[0]["lr"]:.6f} | '
            f'Mem: {torch.cuda.memory_reserved() / 1024**3:.2f}GB')
    print(info)
    with open(os.path.join(save_dirs, 'training_epoch_log.txt'), 'a') as file:
        file.write(info + '\n')

    # 追加测试损失并保存日志
    test_loss_lst.append(L2.item())
    loss_log = {
        'train': np.array(train_loss_lst, dtype=np.float32),
        'test': np.array(test_loss_lst, dtype=np.float32)
    }
    np.save(os.path.join(save_dirs, 'loss_log.npy'), loss_log)

    # 绘图：固定横轴为总 epoch 数；纵轴对数尺度，最小 1e-3
    total_epochs = config.training.epochs
    current_max = max(float(loss_log['train'].max()) if loss_log['train'].size else 1e-3,
                      float(loss_log['test'].max()) if loss_log['test'].size else 1e-3)
    y_max = max(1e-3 * 10, current_max * 1.2)
    plot_loss_curve((1, total_epochs), (1e-3, y_max), loss_log['train'], loss_log['test'], os.path.join(save_dirs, 'loss_curve.png'))

    # 保存最佳模型
    if min_test_loss > L2:
        min_test_loss = L2
        torch.save(model.state_dict(), f'{save_dirs}/checkpoint_best.pth')
        print(f"  >>> Saved best model with test L2: {L2.item():.6f}")

    # 每5个epoch保存一次
    if ep % 5 == 0:
        torch.save(model.state_dict(), f'{save_dirs}/checkpoint_{ep}.pth')

# 最终保存
torch.save(model.state_dict(), f'{save_dirs}/checkpoint_final.pth')
print("\n" + "="*50)
print(f"Training completed in {default_timer()-t0:.2f}s")
print(f"Best test L2: {min_test_loss.item():.6f}")
print(f"Results saved to: {save_dirs}")
print("="*50)
