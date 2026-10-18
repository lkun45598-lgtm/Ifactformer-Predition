# IFactFormer2D-multiStep 讲解文档

> 用**因子化注意力神经算子（Factorized Attention Neural Operator）**做二维 **Rayleigh-Bénard 热对流**流场的多步时序预测。
> 本文档从「任务动机 → 数据流程 → 模型架构 → 核心算法原理 → 训练/推理机制」完整讲清整套代码。

---

## 0. 一句话总览

> 模型本身只学「看当前一帧流场 → 预测下一帧」这**一步演化算子 `G`**；
> 整段未来流场靠把预测结果**反复喂回输入（自回归）**一步步滚出来。
> `G` 内部用「因子化的可学习积分核注意力」实现，`O(N²) → O(nx²+ny²)`。

```
要预测流场演化
 → 学单步演化算子 G（一帧 → 下一帧）
    → G 内部 = 积分核算子：新值 = Σ κ(i,j)·value(j)
       → κ 由注意力现算（q·k，无 softmax，当积分核用）
       → 全场 κ 太贵 → 因子化成 X、Y 两个轴向核
       → κ 从「池化摘要」算，作用于「全分辨率 value」（省算力不丢信息）
       → RoPE 给相对位置（平移等变）；高斯傅里叶给绝对位置（高频抗糊）
    → G 用「同权重迭代 10 次」加深（神经 ODE 思想），仅 82 万参数
 → 训练：把 G 自回归展开 20 步 + 完整 BPTT，逼它长程稳定
 → 推理：从 1 帧真实数据自回归滚出整段未来流场
```

---

## 1. 任务与数据

**物理问题**：二维 Rayleigh-Bénard 热对流（底部加热的流体对流）。给 1 帧流场，预测未来若干帧。

**数据**（`data/rb_Ra1e+05_bott_1_1_upt_0_0.npy`，~100MB）：

| 维度 | 值 | 含义 |
|---|---|---|
| B | 51 | 轨迹条数（每条是一段完整时间演化） |
| T | 100 | 时间步 |
| X | 72 | 空间网格（横向） |
| Y | 24 | 空间网格（纵向） |
| C | 3 | 物理通道：温度 / v 速度 / u 速度 |

- 长宽比 3:1 → `length=[3.0, 1.0]`（`main.py:125`）。
- `data[:-1, :, :, :, :3]`：丢最后 1 条轨迹 → 50 条；取前 3 通道。
- **滑动窗口**造样本（`main.py:77`）：输入 `in=1` 帧、目标 `out=20` 帧，每条轨迹切 `100-(1+20)=79` 个样本，共 `50×79=3950`；8:2 顺序切分 train/test。

> ⚠️ **输入 1 帧、目标 20 帧** —— 不只要求下一帧对，还要求连续预测 20 帧都对。这就是项目名 **multiStep** 的由来。

**归一化**（`main.py:99`）：每通道标量 `mean/std`（用训练集统计，存 `norm.pkl`）。进模型前 `(x-mean)/std`，出模型后 `y*std+mean`。

---

## 2. 模型架构总览

配置（`configs/IFactFormer.yml`）：`dim=96, heads=5, dim_head=96, depth=1, n_layer=10, model=IFactFormer_m`。参数量约 **82 万**。

单帧预测的数据流（真实形状 `B=8, nx=72, ny=24, C=3, dim=96`）：

```
输入 u : (B, 1, 72, 24, 3)          一帧流场
  │  ① to_in    升维 + 吃掉时间维
  ▼
      (B, 72, 24, 96)               每个格点 = 96 维特征
  │  ② encoder  因子化注意力，迭代 10 次
  ▼
      (B, 72, 24, 96)               交互精炼后的特征
  │  ③ simple_to_out  降维回物理通道
  ▼
输出 u : (B, 72, 24, 3)             预测的下一帧
```

三部件分工：

| 部件 | 作用 | 类比 |
|---|---|---|
| ① `to_in` | 3 通道物理量 + 时间窗 → 96 维特征 | 编码器/入口 |
| ② `encoder` | 全格点通过因子化注意力反复交互，模拟空间演化 | **大脑** |
| ③ `simple_to_out` | 96 维特征 → 3 个物理通道 | 解码器/出口 |

### ⚠️ 关键澄清：两类不同的「维度」

| 轴 | 值 | 类型 | encoder 里会变吗 |
|---|---|---|---|
| 72 (x) | 72 | **空间轴**（网格横向有几格） | 不变 |
| 24 (y) | 24 | **空间轴**（网格纵向有几格） | 不变 |
| 96 | 96 | **特征轴**（每格点用几个数描述） | 内容变，长度不变 |

> **x、y 和 96 是并列的两类轴，不会互相转化。**
> x、y 决定「有多少格点」；96 决定「每格点用多少数描述」。
> 「3 → 96」变的是**通道轴**（物理量个数 → 特征个数），发生在 `to_in`；x、y 全程保留。

---

## 3. 各模块详解

### ① `to_in` 入口编码（`model.py:61`）

```python
nn.Conv2d(3, 48, kernel_size=1, groups=3),          # 逐通道 1×1 升维（通道不混）
nn.GELU(),
nn.Conv2d(48, 96, kernel_size=(in_time_window,1)),  # 沿时间窗卷积，把 t 维压成 1
```

- `groups=3`：每个物理通道独立升维，符合物理直觉。
- kernel `(in_time_window,1)`：**模型消化「输入几帧」的地方**（`in=1` 时压掉时间维；若 >1 则融合多帧）。
- 输出布局 channel-last `(B, 72, 24, 96)`。

### ② `encoder = FactorizedTransformer`（`model.py:11`）

```python
for pos_enc, attn_layer in self.layers:      # depth=1：几套独立权重的层
    u += pos_enc(pos)                         # 绝对位置，只加一次
    for i in range(self.n_layer):             # n_layer=10：同一块迭代 10 次
        u = u + attn_layer(u, pos_lst) / self.n_layer   # 残差小步
```

**`depth` vs `n_layer`**：
- `depth`（=1）：堆几套**独立权重**的因子化层。
- `n_layer`（=10）：每套里那个块**共享权重迭代**几次。

**迭代 10 次 = 神经 ODE 思想**：`u = u + attn(u)/10` 是残差小步更新，等价于用欧拉法解演化方程：
```
u_{k+1} = u_k + (1/10)·attn(u_k)   ⟺   x_{k+1} = x_k + Δt·f(x_k)
```
`attn(u)` 扮演「变化率 f」，`1/n_layer` 扮演「步长 Δt」。
好处：**省参数**（10 层深度、1 层参数）、**深度可调**（改配置即可）、**贴合物理**（同一套规律反复作用）、**稳定**（小步不易发散）。

**`pos_enc` 只加一次的原因**：
- 残差流会一路保留它，无需反复加；
- 每次都加会「越加越强」压垮特征；
- 符合「位置是固定背景、演化在其上进行」的直觉。

### ③ `simple_to_out` 解码（`model.py:70`）

逐格点两层 1×1 卷积（等价逐点 MLP）：`Conv1d(96→48) → GELU → Conv1d(48→3)`，把 96 维特征压回 3 个物理通道。

---

## 4. 核心算法：因子化核注意力

### 4.1 原理：注意力 = 可学习的积分核算子

物理直觉：**下一刻某点的值 = 现在一堆点的加权求和**。
```
新值(A) = Σ κ(A, x') × 现值(x')      对所有点 x' 求和
```
- `κ(x, x')`：**核**，表示「点 x′ 对点 x 的影响有多大」。
- 「对所有点加权求和」= **算子**；连续空间写成积分 → **积分算子**（网格上就是大求和）。

对照：

| | 权重从哪来 | 看多远 |
|---|---|---|
| 卷积 | 训练前写死 | 只看邻居 |
| **注意力（本模型）** | **每次由 q·k 现算** | **看全场** |

注意力算核（`attention.py:170`）：
```python
K = einsum('bhid,bhjd->bhij', q, k) * scaling   # κ(i,j) = <q_i, k_j>
```
> **`κ(i,j)` 就是注意力分数** ✅ —— 但要记住它是**不过 softmax 的**（`softmax=False`），被直接当积分核的权重。
> 不 softmax，因为积分核权重可正可负、无需归一化为概率 —— 这是「neural operator 视角」区别于「NLP 注意力视角」的分水岭。

### 4.2 因子化：把 2D 核拆成两个 1D 轴向核

全场 `κ((i,l),(j,m))` 是 `1728×1728`，太贵。假设**核沿轴可分离**：
```
κ((i,l),(j,m)) ≈ κ_x(i,j) · κ_y(l,m)
```
于是「先沿 Y 混合、再沿 X 混合」两次 1D 操作，复杂度 `O(N²) → O(nx²+ny²)`，核矩阵 `1728²` → `72²+24²`。**这就是全部效率来源。**

### 4.3 `FABlock3D_m` 逐句（`factorization_module.py:86`）

```python
u = self.in_norm(u_in)          # LayerNorm（pre-norm 风格）
v = self.to_v(u)                # ① value：被核作用的场   (B,72,24,480)
h = self.to_in(u)              # ② 算核的原料           (B,72,24,96)

u_x = self.to_x(h)              # 沿 ny 池化 → (B,72,96)  X 轴摘要
u_y = self.to_y(h)              # 沿 nx 池化 → (B,24,96)  Y 轴摘要

k_x = self.low_rank_kernel_x(u_x, pos_x)   # (B,5,72,72)  X 轴核
k_y = self.low_rank_kernel_y(u_y, pos_y)   # (B,5,24,24)  Y 轴核

u_phi   = rearrange(v, 'b i l (h c) -> b h i l c')       # (B,5,72,24,96)
u_phi_x = einsum('bhij,bhjmc->bhimc', k_x, u_phi)         # 沿 X 加权求和
u_phi_y = einsum('bhlm,bhimc->bhilc', k_y, u_phi)         # 沿 Y 加权求和

u_phi = cat([u_phi_x, u_phi_y], dim=4)   # 拼两方向 (B,5,72,24,192) → (B,72,24,960)
u_phi = self.to_out(u_phi)               # 融合 MLP → (B,72,24,96)
u     = self.to_out_1(u_phi)             # 再精炼 MLP → (B,72,24,96)
```

**关键分工 `to_v` vs `to_in`**：
- `v`（value）= 核要作用的场，**全分辨率**保留；
- `h`（to_in 输出）= 算核的原料，池化成轴摘要后进 kernel 生成 q/k。
> **核 κ 从「池化后的粗特征」算出来，但作用在「全分辨率 value」上** —— 池化只影响权重计算，不让数据缩水。

**两个 einsum = 两次 1D 积分**：
```
u_x(i,l) = Σ_j κ_x(i,j)·v(j,l)   固定 Y，沿 X 把所有位置按核加权求和
```

### 4.4 `_m` 的关键细节：并联轴向 + MLP 融合

- **`FABlock3D_o`（原版）**：两个 einsum **串联嵌套**，是严格可分离核 `κ_x∘κ_y`，2D 耦合内建。
- **本项目 `_m`**：两个 einsum **并联**（都从同一 `v` 出发，各混一轴），再 `cat`：
  ```
  u_out(i,l) = W·[ Σ_j κ_x(i,j)v(j,l)  ‖  Σ_m κ_y(l,m)v(i,m) ]
                     └── 只混 X ──┘        └── 只混 Y ──┘
  ```

> 在 `_m` 里，`to_out` 的融合 MLP **不是可有可无的收尾** —— 它承担「把两个独立的 1D 轴向注意力缝合成近似 2D 关系」的核心职责。

**`to_out` 融合 MLP 的两个精确点**：
1. 融合的不只是 X/Y 两支，还有 5 个 head（`960 = 2方向 × 5头 × 96`），一次全搅匀。
2. 它是**逐格点的通道融合**，**不再做空间交互**（空间交互已由两个 einsum 完成）。

### 4.5 `LowRankKernel` 里的旋钮（`attention.py:17`）

```python
K = einsum('bhid,bhjd->bhij', q, k) * scaling   # κ(i,j) = <q_i,k_j>
```
- **低秩**：q/k 投影到 `dim_head × kernel_multiplier = 96×3 = 288` 维 → 核矩阵秩 ≤ 288，是低秩双线性核。
- **RoPE 相对位置**（`attention.py:132`）：旋转 q/k，使 `<q_i,k_j>` 只依赖 `i-j` → 核**平移等变**，适配物理场。
- **多头**：`heads=5` = 5 个独立因子化算子并行。
- 本配置：`softmax=False`（不归一化）、`residual=False`（`gamma=0`，不加对角单位阵）、`normalize=False`、`scaling=1.0`。

---

## 5. 两种位置编码，分工互补

| | 类型 | 加在哪 | 次数 | 管什么 |
|---|---|---|---|---|
| `GaussianFourierFeature`（`pos_encode:69`） | **绝对**位置 | encoder 迭代**外**，加到 `u` | 1 次 | 「我在网格哪个格点」 |
| `RoPE`（`attention.py:132`） | **相对**位置 | 每个 `attn` **内部**的 q/k | 每次迭代 | 「两点相距多远」 |

**高斯傅里叶特征 + `scale=8`**（`pos_encode:87`）：
```python
self._B = randn((2, 48)) * scale     # 固定随机频率矩阵（不训练）
x = x @ self._B; x = 2π·x
return cat([sin(x), cos(x)])          # 坐标 → 一组不同频率的 sin/cos
```
- **为什么要**：神经网络有**谱偏差**（天生学不好高频/尖锐结构）。直接喂高频 sin/cos 当积木，绕开这个偏差。
- **`scale=8`**：控制频率高低。太小 → 位置糊；太大 → 像噪声；8 是调出来的折中。
- **为什么流场偏爱高频**：对流的羽流、涡、陡梯度本身是高频结构，低频编码只会把流场预测得一团糊。

---

## 6. 训练与推理机制

### 6.1 训练：自回归 rollout + 完整 BPTT（`main.py:160`）

```python
x_cur = x                          # 起点：真实第 1 帧
for k in range(20):                # 连续走 20 步
    x_norm = (x_cur - x_mean)/x_std
    y_pred = model(x_norm) * y_std + y_mean
    total_loss += LpLoss(y_pred, y[:,k])
    x_cur = y_pred.unsqueeze(1)     # ★ 自回归：把自己的预测当下一步输入
loss = total_loss / 20
loss.backward()                     # 梯度贯穿全部 20 步（不 detach）
```

- **为什么这样训**：只训单步 → 推理滚上百步误差滚雪球式发散。训练时就经历「用带误差的预测继续走」，模型才学会**自我纠错、长程稳定**。
- **完整 BPTT**：20 步计算图 → 显存吃到 ~45GB（日志所示）。这正是 `IFactFormer2D_earthquake` 分支引入梯度检查点/bf16 的动机。
- **损失** `LpLoss.rel`（`utils.py:68`）：相对 L2 `‖pred-true‖₂/‖true‖₂`，对量纲不敏感。
- 优化器 AdamW + StepLR（每 5 轮 ×0.7）。

### 6.2 推理：从 1 帧滚出整段（`eval.py:86`）

```python
x = 真实第 0 帧                     # 唯一喂给模型的真实数据
for i in range(99):                # 预测 99 步补齐整条轨迹
    y = model((x-mean)/std)*std+mean
    predict.append(y)
    x = cat([x[:,1:], y])           # 滑窗；in_tw=1 时即 x=y（纯自回归）
```

```
真实帧0 ─G→ 预测帧1 ─G→ 预测帧2 ─G→ … ─G→ 预测帧99
            (之后输入全是模型自己的输出)
```

### 6.3 评估

- **逐时刻归一化 MSE** `mean((pred-true)²)/mean(true²)` → `relative_l2_error.png`（看误差随时间累积）。
- `visualization*.py` 画预测 vs 真实的 2D 对比图 `2d_comparison_channelX_*.png`。
- 训练结果：30 轮，best Test L2 ≈ **0.051**（epoch 26）；曲线偶有尖峰（epoch 18/27），是自回归多步训练常见的不稳定。

---

## 7. ⚠️ 两个易混点 & 死代码

### 两个「循环」意义完全不同

| 循环 | 位置 | 作用 | 次数 |
|---|---|---|---|
| 迭代 `n_layer` | `model.py:48` | 精炼**单帧**特征（空间演化） | 10 |
| 自回归 `out_tw` | `main.py:168` | 一帧接一帧预测**时间序列** | 20（推理 99） |

> 每预测一帧（外层走一步），模型内部都做 10 次迭代精炼（内层十步）。

### 可忽略的死代码（导入了但本模型没用）

- `factorization_module.py`：**`FABlock3D_o`**（且 `low_rank_kernel_z` 未定义即调用，是坏的）。
- `basics.py`：`Sin / PositionalEmbedding / fftconv_ref / ExponentialModulation / get_time_embedding / masked_instance_norm`（Hyena/扩散遗留）。真正用到的只有 `PreNorm / MLP / GeAct`。
- `positional_encoding_module.py`：`SirenNet / Siren / Sine / get_emb / apply_2d_rotary / apply_3d_rotary`。
- `attention.py`：`'rff' / 'learnable' / 'none'` 分支（用的是 `'rotary'`）。

**真正的活代码**：`model.py` 全部 + `PoolingReducer`/`FABlock3D_m` + `LowRankKernel` + `GaussianFourierFeatureTransform`/`RotaryEmbedding` + `PreNorm`/`MLP`/`GeAct` + `LpLoss`/`get_pos_lst`。

---

## 8. 常动的旋钮

| 旋钮 | 配置项 | 影响 |
|---|---|---|
| 特征宽度 | `dim` | 表示能力 / 显存 |
| 迭代深度 | `n_layer` | 空间演化的「深度」 |
| 注意力头数 | `heads` | 并行算子个数 |
| 训练展开步数 | `out_time_window` | 长程稳定性 / 显存 |
| 核的秩 | `kernel_multiplier` | κ 的表达力 |

---

## 附：与 `IFactFormer2D_earthquake` 分支的区别

同一套 FactFormer 架构（`attention.py` 完全相同），但：

| | 本项目 multiStep | earthquake 分支 |
|---|---|---|
| 物理问题 | Rayleigh-Bénard 热对流 | 地震波场（WaveCastNet） |
| 数据 | ~100MB，(51,100,72,24,3) | ~149GB，mmap，(N,3,376,256,461) |
| 硬件 | 单 GPU | 8 卡 DDP |
| 显存优化 | 无（朴素 fp32，峰值 45GB） | bf16 + 双层梯度检查点 + 显式 del + 梯度裁剪 |
