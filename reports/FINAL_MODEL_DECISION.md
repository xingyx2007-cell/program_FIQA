# PHASE 4：Group C 打开前的最终模型决定

决定日期：2026-09-30。**本文件在首次 Group C 独立评估前写入；决定完全依据固定 A/B 的多种子结果，没有读取 Group C 的评价结果或官方 `val/`。**

## 候选和实际 Loss

- Candidate B：官方 ImageNet 预训练 MobileNetV3-Small 最终特征 + 单输出头；仅 SmoothL1。
- Candidate P：同一模型结构与预训练来源；`SmoothL1 + 0.5 × (1 − Pearson(pred,target))`，分母稳定项 `1e-8`。已从 PHASE 3 的配置快照、最佳检查点配置、`src/losses.py` 与训练调用链核实，属于 **A：SmoothL1 加 PearsonLoss**。
- 两候选的所有其他训练设置一致；没有组合 Ranking 或 Multi-scale，也没有新数据或第三方 FIQA 权重。共同随机种子为 20260930、20261001、20261002。

## A/B 多种子证据

| 候选 | SROCC mean ± std | PLCC mean ± std | Final Score mean ± std | MAE mean ± std | RMSE mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidate B | 0.919922 ± 0.000961 | 0.937746 ± 0.000833 | 0.928834 ± 0.000871 | 0.033153 ± 0.000184 | 0.046174 ± 0.000324 |
| Candidate P | **0.934273 ± 0.000158** | **0.944934 ± 0.000326** | **0.939603 ± 0.000088** | **0.031802 ± 0.000102** | **0.044898 ± 0.000126** |

`std` 使用三个种子的样本标准差。Candidate P 的 Final Score 平均值高 **0.010769**，三个相同种子上分别高 **0.011384、0.011135、0.009788**；其观察到的波动也更小。两者同结构，参数量与推理 FLOPs 一样。综合 SROCC、PLCC、MAE、RMSE 后，**锁定 Candidate P**。这只是三个种子上的稳定性证据，不声称保证任何未见数据的成绩。逐种子数据和验证方法见 [PHASE4_STABILITY.md](PHASE4_STABILITY.md)、[PHASE4_STABILITY.csv](PHASE4_STABILITY.csv)。

## 锁定内容

唯一锁定配置为 [final_locked.yaml](../configs/final_locked.yaml)，SHA-256：`7f45708da749b3e2ba8c1e56ac5be45d095707fd342742fc6eff6d91838172f1`。结构为原始单尺度 MobileNetV3-Small，官方 `IMAGENET1K_V1` 预训练，输入 224×224 保持比例并用灰色填充，RGB + ImageNet 归一化，无图像增强。Loss 为 SmoothL1 + 0.5×PearsonLoss；AdamW，主干学习率 `1e-4`、头部 `3e-4`、weight decay `1e-4`，1 轮线性 warmup 后余弦退火；batch 64、最多 30 轮、patience 6、AMP、梯度裁剪 1.0。

**Group C 的唯一检查点已事先固定**：原种子 `20260930` 的 [phase3_pearson/best.pt](../runs/phase3_pearson/best.pt)，第 24 轮；检查点 SHA-256：`f5a7b90db7ca7cd5df8893b17e13f88fa808cc82d089f0541ed850e8f027b28f`。不用新增种子里 B 组分数最幸运的检查点。该检查点只用 A 训练，B 用于选最佳轮次；C 不是训练或选轮次依据。

最终复杂度已经用该检查点重新 profile，见 [final_model_profile.txt](final_model_profile.txt)：**1,518,881 参数、58,626,560 fvcore FLOPs**；计入未支持逐元素算子的保守 FLOPs 上界 **81,428,960**，无剩余未知算子。参数 ≤5M：**PASS**；FLOPs ≤500M：**PASS**。FLOPs 指单张 `1×3×224×224` 输入的静态计数，不等于实际设备耗时。

## 打开 Group C 前的检查表

- [x] 模型结构已确定：单尺度 MobileNetV3-Small，无新增模块。
- [x] Loss 已确定：SmoothL1 + 0.5×PearsonLoss，`epsilon=1e-8`。
- [x] 超参数、预处理、种子与最佳检查点已固定在锁定配置，且配置、A/B 清单及检查点哈希已核对。
- [x] Group C 从未参与模型、Loss、权重或超参数的选择。
- [x] 即使 Group C 表现较低，也只分析原因，不据此修改模型或重测 C。
- [x] 最终参数量和 FLOPs 均在约束内。

因此允许**一次** Group C 独立评估。该评估完成后，本文件和锁定配置保持不变。已知限制：数据没有可靠的 identity / subject / source 分组字段，不能完全排除同一个人的不同照片跨 A/B/C；三种子复核也无法涵盖所有训练随机性。
