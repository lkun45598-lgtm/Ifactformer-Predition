import os
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

print("="*60)
print("Starting Evaluation")
print("="*60)

# 修改为最新训练的模型目录
load_dir = 'results/dim96_depth1_iter10_T1_5'

# -------------------------- parameters --------------------------
# read the parameters file
with open(f'{load_dir}/IFactFormer.yml', 'r') as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

# -------------------------- data_generation --------------------------
# 加载2D流体动力学数据
data_path = "data/rb_Ra1e+05_bott_1_1_upt_0_0.npy"
data = np.load(data_path)
# 使用第一个样本进行评估
data = data[[-1], ...]  # shape: (1, 100, 72, 24, 3)
data = torch.from_numpy(data).float()

print(f"Evaluation data shape: {data.shape}")

T = data.shape[1]
predict_step = T - config.model.in_time_window

with open(f'{load_dir}/norm.pkl', 'rb') as f:
    norm = pickle.load(f)
    # 将归一化参数移到正确的设备上
    for key in norm:
        if isinstance(norm[key], torch.Tensor):
            norm[key] = norm[key].to(device)

b, t, nx, ny, c = data.shape
in_time_window, out_time_window = config.model.in_time_window, 1
sample_num = t - (in_time_window + out_time_window)

input_data = data[:, :in_time_window, ...]
label = data[:, in_time_window:in_time_window+predict_step, ...]

print(f"Input data shape: {input_data.shape}")
print(f"Label shape: {label.shape}")
print(f"Predict steps: {predict_step}")

# 2D数据的位置编码配置（与训练时一致）
size_lst = [(nx, ny)]  # (72, 24)
length = [3.0, 1.0]  # Rayleigh-Bénard对流的长宽比
pos_lst = get_pos_lst(size_lst, length)
pos_lst = pos_lst[0]


# -------------------------- define --------------------------
model = Model(config.model).to(device)
model.load_state_dict(torch.load(f'{load_dir}'+'/checkpoint_10.pth'))
info = f'count_params: {count_params(model)} | Using checkpoint_10.pth'
print(info)


# -------------------------- eval --------------------------
print("\nStarting autoregressive prediction...")
model.eval()
predict = []
with torch.no_grad():
    x = input_data.to(device)
    for i in range(predict_step):
        y = model((x - norm['x_mean']) / norm['x_std'], pos_lst) * norm['y_std'] + norm['y_mean']
        y = y[None, ...]
        predict.append(y.cpu())

        x = torch.cat([x[:,1:], y], dim=1)

        if (i+1) % 10 == 0:
            print(f"  Predicted step {i+1}/{predict_step}")

predict = torch.cat(predict, dim=1)
predict = torch.cat([input_data, predict], dim=1)
label = torch.cat([input_data, label], dim=1)

print(f"\nPredict shape: {predict.shape}")
print(f"Label shape: {label.shape}")

# 2D数据的MSE计算（注意维度变化：去掉了nz维度）
MSE_time = torch.mean((predict - label) ** 2, dim=(0, 2, 3, 4)) / torch.mean(label ** 2, dim=(0, 2, 3, 4))
print(f"\nMSE over time (first 10 steps):")
for i in range(min(10, len(MSE_time))):
    print(f"  Step {i}: {MSE_time[i]:.6f}")
print(f"\nMean MSE: {torch.mean(MSE_time):.6f}")
print(f"Max MSE: {torch.max(MSE_time):.6f}")
print(f"Min MSE: {torch.min(MSE_time):.6f}")

# save npy_data
save_dir = load_dir + '/npy'
if os.path.exists(save_dir) == False:
    os.makedirs(save_dir)
predict = predict.numpy()
label = label.numpy()
np.save(os.path.join(save_dir, 'predict.npy'), predict)
np.save(os.path.join(save_dir, 'label.npy'), label)

print(f"\n{'='*60}")
print(f"Results saved to: {save_dir}")
print(f"  - predict.npy: {predict.shape}")
print(f"  - label.npy: {label.shape}")
print(f"{'='*60}")
print("Evaluation completed successfully!")
