# FIQA 项目

项目已完成最终全量训练与官方 val 盲测推理，预测 CSV 已通过本地校验。目标是从单张人脸图片预测连续质量分数，方案需满足赛题 **≤5M 参数、≤500 MFLOPs（224×224×3）**。赛题最终评分为 SROCC 与 PLCC 的平均值。详情见[赛题页面](https://www.njupt-mics.cn/join/medeng/computer-vision)。

## 数据与划分

- `train/`：实际 27,686 张 PNG；`train.csv`：无表头、27,686 行，依次为图片 ID 和人工评分。
- `val/val/`：1,000 张无公开评分图片，只用于最终推理。不要用它调参或修改单张预测。
- `splits/clean_dataset.csv`：清洗后 27,679 张；`splits/excluded_samples.csv`：排除 7 条记录（三组同图不同分共 6 张、一组同图同分的额外副本 1 张）。低质量但标签有效的图片保留，原始数据未移动或删除。
- `splits/group_A.csv`、`group_B.csv`、`group_C.csv`：基于清洗清单的固定划分，分别 9,226 / 9,226 / 9,227 张。
- 数据审计与清洗的逐张清单仅保留在本地工作目录，不随 GitHub 仓库或交付包分发；公开仓库提供[划分方案](reports/SPLIT_PLAN.md)及汇总统计。源数据无身份或来源字段，无法保证不同组没有同一人的不同照片。

## 阶段安排

1. A 训练、B 开发验证；调整方案。
2. 配置确定后，从相同初始化重新用 A+B 训练，C 做一次独立检查。
3. 方案固定后重新用 A+B+C 训练最终模型；只对官方无标签 `val/val/` 推理。

详细的初学者解释、术语和三折交叉验证比较见[实验方案](reports/EXPERIMENT_PLAN.md)。[论文索引](reports/LITERATURE_INDEX.md)列出赛题网页的真实链接，[文献综述](reports/LITERATURE_REVIEW.md)说明每篇论文及公开代码对本项目的价值。

## 环境与目录

`requirements.txt` 引用已验证的 `requirements-base.txt`。使用独立 Conda 环境 `fiqa`；实际环境版本记录随离线交付包的 `environment.txt` 提供。`configs/` 放实验配置，`src/` 放模型、数据与指标逻辑，`scripts/` 放审计和清洗/划分工具，`tests/` 放必要测试，`runs/` 放实验结果，`submission/` 放最终提交文件。正式基线权重位于本地 `runs/baseline_mbv3_pretrained/`，不纳入 GitHub 仓库。

## PHASE 1 baseline

`configs/baseline_mbv3.yaml` 使用 MobileNetV3-Small、单值回归头、SmoothL1Loss、AdamW；默认 `use_pretrained: false`，不会下载权重或额外数据。图片保持宽高比，缩放并填充到 224×224；默认不做增强。Batch Size 64 是**每次**处理 64 张，不是把整个 A 组装进显存。

在项目根目录使用 `fiqa` 环境的 Python：

```powershell
python -B train.py --config configs/baseline_mbv3.yaml --smoke
python -B evaluate.py --config configs/baseline_mbv3.yaml --checkpoint runs/baseline_smoke/best.pt --smoke
python -B profile_model.py --output runs/baseline_smoke/profile.json
python -B -m unittest discover -s tests -v
```

第一条只取固定的 **128 张 A** 和 **64 张 B**，训练 1 个 Epoch，并写入 `runs/baseline_smoke/`。第二条独立加载保存的模型再次验证。第三条测量模型计算量，第四条检查数据清单及指标公式。完整结果与局限见[PHASE 1 报告](reports/PHASE1_BASELINE.md)。本阶段没有让模型读取 C 或官方 `val/`。

若需重建清单，依次运行 `scripts/clean_data.py` 和 `scripts/make_splits.py`；正常实验应直接读取已固定的 CSV。`clean_data.py` 只读原始训练数据，输出写入 `splits/` 和 `reports/`。PHASE 1 的配置仍是随机初始化；PHASE 2 的 `use_pretrained: true` 仅选择 torchvision 官方 ImageNet 权重，首次使用会下载至 torchvision 缓存。

## PHASE 2 正式基线

正式配置为 `configs/baseline_mbv3_pretrained.yaml`。它只使用 torchvision 官方 `MobileNet_V3_Small_Weights.IMAGENET1K_V1` 初始化，随后只用赛事 Group A 进行 FIQA 监督训练；不使用第三方 FIQA 权重或额外训练集。该官方权重首次使用时由 torchvision 下载到其缓存。预训练模型的输入在保留宽高比、灰色填充后使用 ImageNet 均值/标准差归一化；Group B 每轮完整验证，Group C 和官方 `val/` 未用于模型实验。

```powershell
python -B scripts/verify_splits.py
python -B train.py --config configs/baseline_mbv3_pretrained.yaml
python -B scripts/analyze_baseline.py --run-dir runs/baseline_mbv3_pretrained
python -B profile_model.py --checkpoint runs/baseline_mbv3_pretrained/best.pt --output reports/baseline_model_profile.txt
```

本次完成 30 轮，最佳第 28 轮，Group B Final Score **0.928293**。公开仓库保留汇总曲线；逐张 B 组预测仅保留在本地项目目录，不进入公开仓库或交付包。**由于没有 identity/subject/source 信息，不能保证同一人的不同照片不跨组；B 组结果仅为当前数据条件下的本地验证。**

## PHASE 5 最终训练与提交文件

PHASE 4 锁定的 Candidate P 使用全部 27,679 张合法训练图片，按三个种子最佳轮次的中位数固定训练 25 轮。最终权重位于 `runs/final_full_train/final_model.pt`，预测文件位于 `submission/predictions.csv`。完整配置、训练健康检查、计算量与官方格式依据见[PHASE 5 报告](reports/FINAL_TRAINING.md)。官方 val 没有公开标签，不能从这批图片计算准确率或再调参。

需要复核提交文件时，在 `fiqa` 环境的项目根目录运行：

```powershell
python -B scripts/validate_submission.py
```

赛事公开页面还要求源码、权重、日志、CSV 与 PPT 组成交付包。正式赛事邮件由项目负责人发送。

## GitHub 归档范围

公开源码归档包含代码、配置、聚合实验报告和答辩文件；赛事原始图片、逐张标签/预测、模型权重、训练运行目录及最终 ZIP 不纳入 Git。获取赛事数据后，按赛题许可将图片放入 `train/`、`val/val/`，再运行数据清洗及固定划分脚本。完整的权重、日志和官方提交 CSV 由线下交付包提供。
