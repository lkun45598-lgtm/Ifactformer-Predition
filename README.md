# IFactFormer 海流预测（Cmems / GBA）

用 IFactFormer-m（`libs/`，沿用原 Rayleigh–Bénard 版本的模型代码，未改动）做海表流速 `[u, v]` 的时序预测：
**输入连续 7 天，自回归预测未来 20 天**（模型每次只预测 1 天，预测值放回窗口继续预测）。

## 目录

| 路径 | 说明 |
|---|---|
| `libs/` | 原模型（`FABlock3D_m`） |
| `cmems_data.py`, `main_cmems.py`, `eval_cmems.py` | Cmems 数据加载、训练（单步 / 多步微调）、20 天评估 |
| `gba_data.py`, `main_gba.py`, `eval_gba.py` | GBA 数据与训练（`gba_data.py` 里的 `predict_next`、`masked_rel_l2_ch` 也被 Cmems 复用） |
| `build_cmems_prediction.py` | 生成 Cmems 数据集（合并、排序、8:1:1 切分、逐日校验） |
| `baseline_linear*.py` | 线性回归基线 |
| `plot_*.py` | 画图脚本（`plot_report*.py` 为汇报图） |
| `configs/` | 各数据集的模型与训练配置（`data_dir` 是作者机器上的路径，使用前需修改） |
| `v2_dim192_timeemb/` | 独立的一份实验代码：dim 192 + 目标日年周期时间嵌入（与上面的基线分开，便于对照） |
| `figures/` | 汇报用结果图 |
| `CODE_EXPLANATION.md`, `MODEL_EXPLAINED.md` | 代码与模型说明 |

数据、权重、训练日志**不在仓库中**。

## 数据（Cmems）

CMEMS `GLOBAL_MULTIYEAR_PHY_001_030` 1/12° 再分析，墨西哥湾北部 240×240，表层（约 0.494 m）日平均流速，
1993-01-01 ~ 2021-06-30 共 10408 天，按时间 8:1:1 切为 train 8326 / val 1041 / test 1041 天。
陆地与零星缺测为 NaN，指标一律只在 `np.isfinite` 的点上计算；归一化统计量只用 train。

## 方法要点

- **预测目标是日增量**：`pred = 最后一个观测日 + 网络输出 × dstd`（`--residual`），不是直接预测第二天的值。
- **损失 / 指标**：u、v 分别计算相对 L2（`‖pred − y‖₂ / ‖y‖₂`，仅有效海洋点）再平均；训练、验证、评估同一口径。
- **多步微调**：K=2 → K=4 自回归展开，K 步的相对 L2 取平均；梯度穿过整个展开，K>1 或开 `--ckpt_grad` 时手动 all-reduce 梯度。

## 结果（Cmems，val，51 个窗口，相对 L2，u 和 v 平均）

| 预报天 | 单步 K=1（12 epoch） | 阶段 A（K=2） | **最终（阶段 B，K=4）** | persistence | 气候态 |
|---|---|---|---|---|---|
| 1 | 0.158 | 0.159 | 0.159 | 0.252 | 0.847 |
| 10 | 0.578 | 0.549 | 0.524 | 0.721 | 0.850 |
| 20 | 0.783 | 0.725 | 0.674 | 0.870 | 0.847 |
| 20 天均值 | 0.557 | 0.526 | **0.500** | 0.685 | 0.850 |

test 集（未参与选模）：最终模型 20 天均值 0.523，单步 0.577，persistence 0.718。

**保留意见**

- 多步微调**不改善第 1 天**（0.158 → 0.159），改善集中在远期；它与"多训练了若干 epoch"混在一起，
  未做"同样日程、仍用 K=1"的对照，因此不能把改善全部归因于多步目标。
- 评估窗口只有 51 个；val 同时用于选权重。
- `v2_dim192_timeemb/` 的结果尚未完成，不在上表中。
- 代码中 `InstanceNorm3d` 被用于 `[b, nx, ny, c]` 的四维输入（`libs/factorization_module.py`），
  实际是对每个样本在 `(nx, ny, c)` 上整体标准化，而非逐通道；该层对精度的影响**未验证**。

## 使用

```bash
# 单步训练（8 卡示例）
torchrun --nproc_per_node=8 main_cmems.py --rollout 1 --residual --amp --lr 4e-4 --epochs 12 --batch_size 2 --warmup 1.0
# 多步微调（需 --init 与 --norm）
torchrun --nproc_per_node=8 main_cmems.py --rollout 2 --stride 3 --init <K=1 权重> --norm <norm.pkl> \
  --lr 1e-4 --epochs 8 --warmup 0.2 --batch_size 1 --residual --amp
# 20 天评估
python eval_cmems.py --run <结果目录> --ckpt checkpoint_best.pth --split val --stride 20 --batch_size 4 --amp --gpu 0
```
