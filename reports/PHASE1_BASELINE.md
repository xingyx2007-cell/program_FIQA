# FIQA PHASE 1：干净基线流水线

> 历史阶段记录：PHASE 1 当时默认不使用预训练，代码也曾要求显式本地权重。PHASE 2 获得使用 torchvision 官方 ImageNet 权重的明确授权后，当前代码改为仅允许官方 `IMAGENET1K_V1` 枚举；正式结果见 [BASELINE_ANALYSIS.md](BASELINE_ANALYSIS.md)。下文描述的是 PHASE 1 执行时的状态。

完成日期：2026-09-30。**仅进行短时间 smoke test，没有正式长时间训练，没有让模型读取 Group C 或官方无标签 `val/`。**

## 数据清洗结果

依据[清洗报告](DATA_CLEANING.md)与[排除清单](../splits/excluded_samples.csv)：原始 27,686 条，排除 7 条，最终 27,679 条。损坏 0、图片或标签缺失 0、NaN/Inf/非法评分 0；完全相同且评分一致的多余副本 1 条；三组完全相同但评分冲突共 6 条。冲突组全部排除，没有挑选或平均标签。低评分但有效的图片保留。原始图片和 `train.csv` 没有修改。

## A/B/C 固定划分

以[clean_dataset.csv](../splits/clean_dataset.csv)为唯一输入，随机种子 `20260930`，对质量分数做十分位分桶后按桶分层划分。三组完全不重叠，合并恰好为全部干净样本，评分原文保持不变。详见[划分报告](SPLIT_PLAN.md)。

| 组 | 数量 | 最小评分 | 最大评分 | 平均评分 | 样本标准差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 9,226 | 0.041491 | 0.907592 | 0.365903 | 0.133470 |
| B | 9,226 | 0.029202 | 0.883606 | 0.365610 | 0.132681 |
| C | 9,227 | 0.026835 | 0.908365 | 0.365555 | 0.132597 |

最大十分位比例差为 **0.0079 个百分点**。源数据没有 identity/subject/source 字段，**无法保证同一个人的不同照片不跨组**；内容完全相同的图片已由清洗步骤解决。这一限制不能由分层抽样消除。

## Baseline 与图片处理

一张图先转 RGB，再保持原宽高比缩放，使用中性灰色填充成 `224×224`；不裁掉面部边缘，也不把人脸横向拉伸。训练和验证的基础处理一致，默认没有翻转或其他增强。图片转成取值约在 0–1 的 Tensor（类似可由 GPU 计算的 NumPy 数组）。

模型为 torchvision **MobileNetV3-Small**（轻量 CNN）单一最终 Feature 后接一个线性 Regression Head，输出 `1` 个连续分数。默认 `use_pretrained: false`；若以后设为 `true`，必须指定已存在的本地权重文件，不会自动下载 ImageNet 权重。训练只用 **SmoothL1Loss**、AdamW，学习率 `3e-4`、权重衰减 `1e-4`。不含多尺度、Pearson Loss、Ranking Loss、注意力或 Transformer。

Dataset 根据分组 CSV 找路径、解码图片，并返回 `(image, label, filename)`；DataLoader 按批取样。Batch Size `64` 指每步最多 64 张图片，并非一次装入全部 9,226 张。一个 Epoch 指把所选训练样本全部看过一遍。模型先预测分数，SmoothL1Loss 计算“错误分数”，反向传播求参数的调整方向（梯度），AdamW 根据梯度更新参数；学习率控制每步幅度。

## Smoke Test 与指标

命令 `python -B train.py --config configs/baseline_mbv3.yaml --smoke` 使用固定抽取的 **128 张 A** 训练 1 Epoch，**64 张 B** 验证，GPU 为 NVIDIA GeForce RTX 5070 Ti Laptop GPU（约 12 GiB 显存）。A 上有两批各 64 张，B 上一批 64 张。命令做了 forward、SmoothL1Loss、backward、AdamW 更新、验证、保存 `best.pt`/`last.pt`、重载 `best.pt` 并再次推理。所有步骤无报错。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| Dataset / DataLoader | PASS | A 128、B 64 样本均成功读取 |
| Forward | PASS | 单分数预测正常；复杂度脚本输入 `1×3×224×224`，输出 `1×1` |
| Backward / Optimizer | PASS | 两批完成反向传播与更新，训练损失 0.072583 |
| Validation / Metrics | PASS | B 组 64 张产生全部五项指标 |
| Checkpoint | PASS | `best.pt`、`last.pt` 均写入磁盘 |
| Reload / Inference | PASS | 重载后对 `007077.png` 再次推理得到有限数值 `-0.008110` |
| 独立 evaluate.py | PASS | 重新从 checkpoint 读取后五项指标与训练日志完全一致 |
| 固定随机性 | PASS（本机本环境） | 同一 smoke test 重跑两次，`metrics.csv` SHA-256 相同：`96FA4BB8BA6F1274E20C6FB29F3399C2E45A1CE0657585DE063FAD7AF0F9212A` |

验证指标：SROCC（质量排序相关） **0.0327408444**；PLCC（分数线性相关） **-0.0001862010**；Final Score `0.5×(SROCC+PLCC)` **0.0162773217**；MAE（平均绝对误差） **0.3582675127**；RMSE（均方根误差） **0.3828885939**。这些来自随机初始化、128/64 张图片的流水线测试，**不能作为正式实验结论**。单次重载预测为负数也说明线性回归头目前无输出范围约束；本阶段不凭一张图片修正模型或预测。

## 模型复杂度

命令 `python -B profile_model.py --output runs/baseline_smoke/profile.json` 在 CPU 上用 `1×3×224×224` 随机张量测量模型结构，不训练或读取任何数据图片。结果：**1,518,881 参数**，fvcore 默认计入 **58,626,560 FLOPs**，分别低于 5,000,000 和 500,000,000 限制。

fvcore 初次统计提示未计入 `hardswish_` 19 次、`hardsigmoid` 9 次、`mul` 9 次、`add_` 6 次、推理态 `dropout_` 1 次。脚本记录这些提示，再用每个相关输出元素 **20 次运算**的保守上界核算上述逐元素操作，得到 **81,428,960 FLOPs 上界**，仍低于 500M；此第二值是说明计数裕量的保守估计，不是赛方正式测量口径。模型没有为了消除提示而改结构。**当前单模型结构复杂度 PASS**；未来若改预处理、加入多次裁剪或集成，须重新核算完整推理路径。

## 重要命令和输出如何读

| 命令 | 做了什么 | 关键输出与异常含义 |
| --- | --- | --- |
| `python scripts/clean_data.py` | 只读原始训练图片与 CSV，生成干净/排除清单 | `clean 27679 excluded 7` 正常；若损坏或缺图数量增加，应先查数据变化。 |
| `python scripts/make_splits.py` | 从干净清单固定划分并检查完整性 | `PASS: disjoint, complete, cleaned, stratified` 表示三组不重叠且无遗漏；失败不能继续训练。 |
| `python -B train.py ... --smoke` | 少量 A 训练、B 验证、保存/重载 | `device=cuda` 表示 GPU 正在用；`checkpoint reload/inference PASS` 表示保存的参数可再次预测；报错须先修复流水线。 |
| `python -B evaluate.py ... --smoke` | 独立从磁盘加载权重，并在同一固定 B 子集算指标 | 与日志相同是正常；若差异大，应查模型/图片处理或样本选择不一致。 |
| `python -B profile_model.py ...` | 测参数和运算量 | 参数量及保守运算上界均低于限制；若任一超限，不能继续把该结构作为比赛模型。 |
| `python -B -m unittest discover -s tests -v` | 检查清洗/划分合同、指标公式和预训练本地文件约束 | 3 项通过；失败说明数据文件或实现需检查。 |

## 运行产物与限制

`runs/baseline_smoke/` 包含 `config.yaml`、`metrics.csv`、`train.log`、`best.pt`、`last.pt`，另有 `profile.json`。完整训练没有开始。随机种子固定 Python、NumPy、PyTorch 和 CUDA；CuDNN 使用确定性设置，DataLoader 为单进程。跨不同硬件/驱动/框架版本仍可能有细小差异。

尚未确认比赛是否允许外部预训练权重或额外训练数据，因此本次均未使用；后续需根据规则决定。官方网页所称约 30,000 张与本地实际 27,686 张的差额原因仍未知，但本地 CSV 与图片完整对应。没有身份/来源信息是评估独立性的主要剩余限制。
