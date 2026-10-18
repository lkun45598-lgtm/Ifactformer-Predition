# IFactFormer2D-multiStep 代码详解

## 1. 项目概览

这套代码实现了一个用于二维 Rayleigh-Bénard 对流流场的时序预测模型 IFactFormer。

模型每次读取过去的流体场，预测下一个时间步的温度和速度场；训练和评估时再把预测帧反馈给模型，从而进行多步自回归预测。

整体流程：

    流体 .npy 数据
        |
        v
    滑动时间窗口构造样本
        |
        v
    按通道标准化
        |
        v
    IFactFormer
      - 输入投影 Conv2d
      - 坐标 Fourier 位置编码
      - x/y 方向因子化低秩注意力
      - 输出投影 Conv1d
        |
        v
    预测下一帧
        |
        v
    自回归滚动预测
        |
        v
    相对 L2 误差和二维可视化

当前配置下：

- 输入 1 个时间帧；
- 空间尺寸为 72 x 24；
- 每帧包含 3 个变量通道；
- 训练时每个样本连续预测 20 步；
- 评估时从一帧初始状态预测后续 99 步。

## 2. 目录结构

    configs/IFactFormer.yml       模型和训练配置
    data/*.npy                    流体数据
    main.py                       训练入口
    eval.py                       长时间自回归评估入口
    visualization.py              长时间预测可视化
    visualization_detail.py       早期时间步详细可视化

    libs/model.py                 总模型结构
    libs/factorization_module.py  因子化注意力模块
    libs/attention.py             LowRankKernel 实现
    libs/positional_encoding_module.py
                                  Fourier 特征和旋转位置编码
    libs/basics.py                基础层、MLP、归一化
    libs/utils.py                 配置、坐标、损失和绘图工具

    results/                      checkpoint、日志、预测和图片

主要入口：

- main.py：构造数据集并训练模型。
- eval.py：加载 checkpoint，对独立轨迹做长时间自回归预测。
- visualization.py：生成误差曲线和长时间二维场对比图。
- visualization_detail.py：生成早期时间步的详细对比图。

## 3. 配置文件

配置位于 configs/IFactFormer.yml：

    log_dir: results

    model:
        in_dim: 3
        out_dim: 3
        in_time_window: 1
        out_time_window: 20
        dim: 96
        heads: 5
        depth: 1
        dim_head: 96
        n_layer: 10
        model: IFactFormer_m

    training:
        epochs: 30
        lr: 0.0005
        batch_size: 8
        scheduler_step: 5
        scheduler_gamma: 0.7

参数含义：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| in_dim | 3 | 输入物理变量数量 |
| out_dim | 3 | 输出物理变量数量 |
| in_time_window | 1 | 历史输入帧数量 |
| out_time_window | 20 | 训练 rollout 的未来步数 |
| dim | 96 | 隐藏特征维度 |
| heads | 5 | 注意力头数 |
| dim_head | 96 | 每个注意力头的基础维度 |
| depth | 1 | Transformer block 数量 |
| n_layer | 10 | 每个 block 的重复残差更新次数 |
| epochs | 30 | 训练轮数 |
| lr | 0.0005 | AdamW 初始学习率 |
| batch_size | 8 | batch 大小 |
| scheduler_step | 5 | 每 5 个 epoch 调整一次学习率 |
| scheduler_gamma | 0.7 | 学习率乘数 |
| model | IFactFormer_m | 使用二维因子化模块 |

depth 和 n_layer 是两个不同概念。当前只有一个 block，但该 block 中的注意力模块会被调用 10 次；这 10 次调用共享同一个模块参数。

## 4. 数据格式和样本构造

数据文件：

    data/rb_Ra1e+05_bott_1_1_upt_0_0.npy

实际数组形状：

    (51, 100, 72, 24, 3)

代码约定布局：

    (B, T, X, Y, C)

含义：

- B=51：轨迹数量；
- T=100：每条轨迹的时间步数量；
- X=72、Y=24：二维网格尺寸；
- C=3：物理变量通道数量。

### 4.1 读取数据

main.py 中的读取逻辑是：

    data = np.load(data_path)
    data = data[:-1, :, :, :, :3]

这会去掉最后一条轨迹，训练阶段保留前 50 条；同时保留前三个通道。当前数据本身就是 3 通道，因此最终形状是：

    (50, 100, 72, 24, 3)

代码注释中“只取前两个通道”的说法与实际实现不一致。

### 4.2 滑动窗口

当前配置：

    in_time_window  = 1
    out_time_window = 20

每个样本的语义是：

    输入：第 j 帧
    标签：第 j+1 到第 j+20 帧

样本形状：

    input : (1, 72, 24, 3)
    output: (20, 72, 24, 3)

代码使用：

    sample_num = t - (in_time_window + out_time_window)

代入 T=100：

    sample_num = 100 - (1 + 20) = 79

因此训练样本总数约为：

    50 x 79 = 3950

再按 80%/20% 切分：

    训练样本：3160
    测试样本：790

样本按生成顺序切分，所以大致对应前 40 条轨迹训练、后 10 条轨迹测试，而不是把同一轨迹的窗口随机拆散。

注意：严格计算滑动窗口时，通常应使用：

    t - in_time_window - out_time_window + 1

当前代码少使用了每条轨迹最后一个有效窗口。

## 5. 数据标准化

标准化统计量只用训练集计算：

    norm['x_mean'] = mean(train_x, dim=(0, 1, 2, 3))
    norm['x_std']  = std(train_x,  dim=(0, 1, 2, 3))
    norm['y_mean'] = mean(train_y, dim=(0, 1, 2, 3))
    norm['y_std']  = std(train_y,  dim=(0, 1, 2, 3))

统计量形状都是：

    (3,)

模型输入前：

    x_norm = (x_cur - norm['x_mean']) / norm['x_std']

模型输出后：

    y_pred = model(x_norm, pos_lst)
    y_pred = y_pred * norm['y_std'] + norm['y_mean']

归一化参数保存到实验目录的 norm.pkl。

main.py 还计算了：

    energy = sum(data ** 2, dim=(2, 3))
    delta_energy = energy[:, 1:, :] - energy[:, :-1, :]

得到的 delta_energy_max 和 delta_energy_min 当前没有用于损失函数或前向计算，属于未使用的预留统计量。

## 6. 空间位置编码

训练和评估都使用：

    size_lst = [(nx, ny)]
    length = [3.0, 1.0]

因此坐标为：

    x：0 到 3，共 72 个点
    y：0 到 1，共 24 个点

FactorizedTransformer 会构造二维网格坐标，并展平为：

    pos: (72 x 24, 2) = (1728, 2)

之后使用：

    GaussianFourierFeatureTransform(2, dim // 2, 8)
    Linear(dim, dim)

把二维物理坐标映射成 96 维 Fourier 特征，并加到流场隐藏表示上。这样模型可以区分不同空间位置，也能表示较高频的空间变化。

## 7. 模型整体结构

模型定义在 libs/model.py。当前前向形状为：

    输入 [B, 1, 72, 24, 3]
        |
        v
    rearrange -> [B, 3, 1, 1728]
        |
        v
    to_in -> [B, 96, 1, 1728]
        |
        v
    rearrange -> [B, 72, 24, 96]
        |
        v
    FactorizedTransformer
        |
        v
    [B, 72, 24, 96]
        |
        v
    simple_to_out
        |
        v
    输出 [B, 72, 24, 3]

模型本身只预测一个下一时刻场；时间维上的多步预测由 main.py 和 eval.py 的自回归循环完成。

## 8. 输入投影 to_in

输入先由：

    u = rearrange(u, 'b t nx ny c -> b c t (nx ny)')

转换为卷积布局：

    [B, 1, 72, 24, 3] -> [B, 3, 1, 1728]

第一层卷积参数相当于：

    in_channels=3
    out_channels=48
    kernel_size=1
    groups=3

这是一个分组逐点卷积。每个物理通道独立产生 16 个中间特征，总共 48 个通道。

第二层在当前 time_window=1 时使用 kernel_size=(1,1)，本质上把 48 维投影到 96 维：

    [B, 48, 1, 1728] -> [B, 96, 1, 1728]

最后恢复空间布局：

    [B, 96, 1, 1728] -> [B, 72, 24, 96]

## 9. Factorized Transformer

FactorizedTransformer 为每个 depth block 创建：

1. 坐标 Fourier 位置编码；
2. 因子化注意力模块。

当前 depth=1，因此只有一个 block。该 block 内的更新形式为：

    u = u + attn_layer(u, pos_lst) / n_layer

这表示 10 次共享参数的残差迭代，而不是 10 个完全独立的 Transformer 层。

## 10. FABlock3D_m 的二维因子化注意力

虽然类名包含 3D，当前 IFactFormer_m 路径实际只处理二维 x/y 轴。

输入：

    u_in: [B, 72, 24, 96]

### 10.1 LayerNorm 和 Value 投影

    u = LayerNorm(u_in)
    v = Linear(96, heads * dim_head)(u)

由于 heads=5、dim_head=96：

    96 -> 5 x 96 = 480

拆成多头：

    [B, 72, 24, 480] -> [B, 5, 72, 24, 96]

### 10.2 x 方向聚合

    u_x = self.to_x(u)

PoolingReducer 沿 y 方向平均：

    [B, 72, 24, 96] -> [B, 72, 96]

此时每个 x 坐标对应一个聚合特征。

### 10.3 y 方向聚合

    u_y = self.to_y(u)

先交换 x/y，再沿 x 平均：

    [B, 72, 24, 96]
        -> [B, 24, 72, 96]
        -> [B, 24, 96]

此时每个 y 坐标对应一个聚合特征。

### 10.4 LowRankKernel

x 和 y 方向分别使用独立的 LowRankKernel：

    k_x = low_rank_kernel_x(u_x, pos_x)
    k_y = low_rank_kernel_y(u_y, pos_y)

kernel_multiplier=3，因此内部 Q/K 的每头维度为：

    96 x 3 = 288

最终核矩阵形状：

    k_x: [B, 5, 72, 72]
    k_y: [B, 5, 24, 24]

模型没有直接构造全部 1728 个空间点之间的 [1728,1728] 注意力，而是分别建立 x 轴和 y 轴关系，降低计算量。

### 10.5 Rotary Position Embedding

当前使用 rotary 位置编码：

    positional_embedding = 'rotary'

x 核使用 x 坐标，y 核使用 y 坐标。RoPE 根据坐标生成正余弦频率并旋转 Q/K，使注意力同时依赖流场特征和物理位置。

### 10.6 沿 x/y 方向传播

代码分别使用 einsum 传播 Value：

    u_phi_x = einsum('bhij,bhjmc->bhimc', k_x, u_phi)
    u_phi_y = einsum('bhlm,bhimc->bhilc', k_y, u_phi)

两路结果拼接后：

    [B, 5, 72, 24, 96]
        -> [B, 5, 72, 24, 192]
        -> [B, 72, 24, 960]

其中：

    960 = 5 x (96 + 96)

之后经过 InstanceNorm3d、Linear、GELU 和 Linear，再经过一个 96 维 MLP，输出新的：

    [B, 72, 24, 96]

## 11. 输出投影 simple_to_out

输出模块为：

    Rearrange('b nx ny c -> b c (nx ny)')
    Conv1d(96, 48, kernel_size=1)
    GELU
    Conv1d(48, 3, kernel_size=1)

形状变化：

    [B, 72, 24, 96]
        -> [B, 96, 1728]
        -> [B, 48, 1728]
        -> [B, 3, 1728]
        -> [B, 72, 24, 3]

## 12. 训练流程

main.py 的训练阶段对每个样本进行 20 步自回归：

    真实 t   -> 预测 t+1
    预测 t+1 -> 预测 t+2
    预测 t+2 -> 预测 t+3
    ...
    预测 t+19 -> 预测 t+20

伪代码：

    x_cur = x
    total_loss = 0

    for k in range(20):
        x_norm = normalize(x_cur)
        y_pred_k = denormalize(model(x_norm, pos_lst))
        total_loss += relative_l2(y_pred_k, y[:, k])
        x_cur = y_pred_k.unsqueeze(1)

    train_loss = total_loss / 20

从第二步开始，模型使用自己的预测，而不是标签中的真实帧，因此训练没有 teacher forcing。这种方式更贴近长期推理，但优化难度也更大。

测试阶段使用完全相同的 20 步 rollout，只是放在 eval 模式和 no_grad 环境中。

## 13. 损失函数

LpLoss 位于 libs/utils.py。当前调用：

    loss_fn = LpLoss(reduction=False)

其 __call__ 实际返回相对 L2：

    loss = ||prediction - target||_2 / ||target||_2

外层再对 batch 求平均。因此当前优化目标是每个预测时间步的平均相对 L2 误差。

代码没有加入：

- PDE 残差；
- 边界条件损失；
- 能量守恒损失；
- 动能变化约束；
- 像素级 MSE 作为额外项。

## 14. 优化器、日志和 checkpoint

优化器：

    AdamW(model.parameters(), lr=0.0005)

学习率调度：

    StepLR(step_size=5, gamma=0.7)

已有结果中的模型参数量约为：

    821,811

保存文件：

    checkpoint_best.pth   测试损失最小时保存
    checkpoint_{ep}.pth   周期性保存
    checkpoint_final.pth  训练结束时保存
    norm.pkl              标准化统计量
    loss_log.npy          train/test loss
    training_epoch_log.txt
    loss_curve.png

已有结果目录：

    results/dim96_depth1_iter10_T1_5

日志显示已有训练的最优测试损失约为 0.050943，约出现在 epoch 索引 26。

## 15. 评估流程

eval.py 默认读取：

    results/dim96_depth1_iter10_T1_5

然后选择原始数据的最后一条轨迹：

    data = data[[-1], ...]

训练时 main.py 使用 data[:-1]，所以该轨迹未参与训练。

评估从第一帧开始：

    input_data = data[:, :in_time_window, ...]
    label = data[:, in_time_window:, ...]

当前输入形状：

    [1, 1, 72, 24, 3]

预测步数：

    predict_step = 100 - 1 = 99

每次预测后将模型输出作为下一次输入。最后拼回初始帧，因此：

    predict: [1, 100, 72, 24, 3]
    label:   [1, 100, 72, 24, 3]

结果保存为：

    results/dim96_depth1_iter10_T1_5/npy/predict.npy
    results/dim96_depth1_iter10_T1_5/npy/label.npy

评估脚本中的 MSE_time 名称并不完全准确。其实际计算是每个时间步的归一化平方误差：

    mean((predict - label) ** 2) / mean(label ** 2)

## 16. 可视化

### visualization.py

该脚本读取 predict.npy 和 label.npy，并计算：

    temperature
    v-velocity
    u-velocity
    overall

四条相对 L2 误差曲线，保存为：

    relative_l2_error.png

二维图默认选择：

    time_indices = [1, 19, 39, 59, 79, 99]

每个通道生成一张 3 行 6 列图：

    第一行：Ground Truth
    第二行：IFactFormer Prediction
    第三行：Absolute Error

温度使用 viridis，速度分量使用 RdBu_r。

### visualization_detail.py

这个版本关注早期时间步：

    time_indices = [1, 4, 9, 14, 19, 24]

适合观察误差刚开始累积时的空间细节。

## 17. 当前实现中的重要注意事项

### 17.1 CPU 路径与 .cuda() 冲突

libs/utils.py 的 get_pos_lst 直接使用 .cuda()。因此即使主程序判断设备为 CPU，坐标生成仍可能强制调用 CUDA。更稳妥的写法是传入 device：

    torch.linspace(..., device=device)

### 17.2 CPU 加载 CUDA checkpoint

eval.py 当前直接使用 torch.load 加载 checkpoint。如果 checkpoint 在 GPU 上保存、当前环境没有 CUDA，应使用：

    torch.load(path, map_location=device)

norm.pkl 中的 Tensor 也可能携带 CUDA 设备信息，需要在加载后统一移动到当前 device。

### 17.3 评估默认不是最佳模型

训练会保存 checkpoint_best.pth，但 eval.py 默认加载 checkpoint_10.pth。因此评估结果不一定对应最小测试损失。若要报告最佳模型，应改为加载 checkpoint_best.pth。

### 17.4 IFactFormer_o 分支不完整

model.py 中存在 IFactFormer_o 分支，但导入语句没有正确导入 FABlock3D_o。并且 FABlock3D_o 使用了未完整初始化的 z 方向 kernel。当前配置应继续使用 IFactFormer_m。

### 17.5 InstanceNorm3d 的输入布局需要检查

FABlock3D_m 将特征整理为 [B, X, Y, C]，随后使用 InstanceNorm3d(C)。而 InstanceNorm3d 通常期望通道维位于第二维。当前实现会出现维度警告，归一化语义也需要确认。若目标是对最后特征维归一化，LayerNorm(C) 通常更直接。

### 17.6 滑动窗口数量少一个

当前 sample_num 为：

    t - (in_time_window + out_time_window)

通常应为：

    t - in_time_window - out_time_window + 1

当前版本每条轨迹少使用一个合法窗口。

### 17.7 训练 rollout 和评估 rollout 不同

训练只优化未来 20 步，评估却从第一帧连续预测 99 步。评估后半段属于比训练更长的外推，误差逐渐积累是预期现象。

## 18. 运行方式

安装依赖后：

    python main.py

评估：

    python eval.py

长时间预测可视化：

    python visualization.py

早期时间步详细可视化：

    python visualization_detail.py

训练会创建类似下列目录：

    results/dim96_depth1_iter10_T1_20_0

目录名由模型配置和序号组成：

    dim{dim}_depth{depth}_iter{n_layer}_T{in_time_window}_{序号}

## 19. 总结

这套代码的核心思想是：

1. 用卷积把输入流体帧映射到高维隐藏空间；
2. 用二维坐标 Fourier 特征注入物理空间位置信息；
3. 将二维全局关系分解为 x 轴和 y 轴上的低秩注意力核；
4. 通过共享参数的残差迭代提取空间长程依赖；
5. 输出下一帧流场；
6. 在训练和评估中反复反馈预测结果，实现多步自回归预测。

因此，IFactFormer 的关键并不是标准的全局 Transformer 注意力，而是“坐标感知的二维轴向因子化低秩核”。
