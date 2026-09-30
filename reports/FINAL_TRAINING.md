# PHASE 5：最终全量训练与官方盲测提交

## 训练集与配置依据

最终训练清单为 [`splits/final_train.csv`](../splits/final_train.csv)，27,679 行，SHA-256 为 `0cabfa34d653032243c8f6637ba4b9c59c4395aa0a2907aa0b65b390a8257ad0`。它与既有清洗清单逐行一致，也恰好等于 A（9,226）+ B（9,226）+ C（9,227）。PHASE 1 排除的 7 条记录未进入清单；所有标签均为有限数值，所有训练图片均已重新完整解码。原 `train.csv` 和图片未修改。C 在 PHASE 4 完成一次独立评估后，现可与 A/B 一起用于训练；本阶段不再利用 C 选择模型或超参数。

来源配置是 [`configs/final_locked.yaml`](../configs/final_locked.yaml)，SHA-256 为 `7f45708da749b3e2ba8c1e56ac5be45d095707fd342742fc6eff6d91838172f1`。最终运行配置保存在 [`runs/final_full_train/config.yaml`](../runs/final_full_train/config.yaml)，只改用完整训练清单及其哈希，并新增 `final_epochs: 25`；锁定文件本身未改。配置中保留的 `val_split`、`test_split`、既有 `checkpoint` 与 `early_stopping_patience` 是早期实验的来源记录，最终训练脚本**不读取这些字段，不创建验证集，不做早停或最佳轮次选择**。

Candidate P 为单尺度 MobileNetV3-Small，加单值线性回归头。初始化使用 torchvision 官方 `IMAGENET1K_V1` ImageNet 权重，其 SHA-256 为 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`；此后仅使用赛事官方训练图片和评分更新模型。Loss 为 `SmoothL1 + 0.5 × PearsonLoss`（稳定项 `1e-8`）。AdamW 的 backbone/head 初始学习率分别为 `1e-4`/`3e-4`，weight decay `1e-4`；1 轮线性 warmup 后余弦退火。batch size 64、seed 20260930、AMP、梯度裁剪 1.0。图片转 RGB，保持比例并用灰色填充至 224×224，再使用 ImageNet 均值/标准差归一化；无数据增强。推理预处理与训练读取器已对照，输出逐元素完全相同。

固定轮数取 PHASE 4 三个种子的 Candidate P 最佳轮次 24、26、25 的**中位数 25**。它在正式全量训练前确定；官方 val 没有真实标签，不能用于早停。训练 Loss 也不作为临时增加轮数的理由。

## 官方 val 与 CSV 格式依据

官方 [`赛题页面`](https://www.njupt-mics.cn/join/medeng/computer-vision) 的 Track 01 交付说明写明：提交 1,000 张验证图像的打分 CSV，“格式：图像文件名, 预测分数”，**必须包含表头**且与图片一一对应。项目内未找到 `sample_submission.csv`，网页也没有提供可下载的样例提交文件。因此，本项目直接采用官方文字中的两个中文名称 `图像文件名,预测分数` 作为表头；第一列填写含 `.png` 的实际文件名，不填目录路径或训练图像 ID。官方未规定排序，故按文件名字典序固定输出。此格式选择的依据是公开交付说明，而非臆造英文字段名；若课题组另有未公开的机器解析模板，仍需以其正式模板复核。

官方 `val/val/` 实测有 1,000 张 PNG，文件名无重复；全部能解码，图像模式都是 RGB。无公开标签，本报告不计算或报告官方 val 的 SROCC、PLCC、Final Score、MAE 或 RMSE。

## 复现命令

在不含已生成最终权重与预测文件的独立项目复制中，于项目根目录和 `fiqa` Conda 环境执行：

```powershell
python -B scripts/prepare_final_train.py
python -B train_final.py
python -B profile_model.py --checkpoint runs/final_full_train/final_model.pt --output reports/final_full_model_profile.txt
python -B scripts/infer_official_val.py
python -B scripts/validate_submission.py
```

`train_final.py` 检查 PHASE 4 最佳轮次、官方预训练文件哈希与锁定配置。它保存每轮训练损失和学习率，不读取 B/C 验证评分或官方 val 标签。推理脚本拒绝覆盖已有预测文件，因此正式 val 推理只能运行一次；如需重跑，必须先人工核对原因。以上路径均相对于项目根目录，代码和配置没有写入个人机器的绝对路径。

## 交付边界

上述网页还要求把源码、权重、日志、CSV 与 PPT 打成指定命名的 ZIP，并通过邮件交付。**单独一份预测 CSV 通过本地检查，并不等于整份赛事交付包已经完成。**本阶段按用户要求生成和校验 CSV，不自动打包、不制作 PPT、不发邮件或上传。

## 实际训练与健康检查

- 完整最终运行：25/25 轮，GPU，训练耗时 **2,129.9 秒（约 35 分 30 秒）**。[`metrics.csv`](../runs/final_full_train/metrics.csv) 记录每轮总 Train Loss、SmoothL1、Pearson Loss、两组学习率、裁剪前最大梯度范数、AMP 跳步次数和耗时；[`train.log`](../runs/final_full_train/train.log) 保留逐轮日志。没有 B/C 验证分数或官方 val 指标列。
- Train Loss：第 1 轮 `0.136323`，第 25 轮 `0.005780`。25 行数值指标全部有限；总共 4 次 AMP 梯度溢出步骤由缩放器自动跳过（第 1 轮 3 次、第 15 轮 1 次）。因此不能说“训练中从未检测到非有限梯度”；可以说未出现非有限训练 Loss，溢出未进入参数更新，模型正常完成。
- 第一次启动时，新加的健康检查把 AMP 可自动恢复的首批梯度溢出当成致命错误，训练在首轮中断，未生成最终模型。原因确认后仅修正监测逻辑以沿用原 `train.py` 的缩放器处理方式，保留 [`initial_amp_overflow.log`](../runs/final_full_train/initial_amp_overflow.log) 作为记录，随后从同一种子和官方预训练初始化**重新开始完整 25 轮**；没有从中断处续训，也没有改模型、Loss 或学习率。
- 最终权重：[`final_model.pt`](../runs/final_full_train/final_model.pt)，6,210,967 字节，SHA-256 `ba25187ccbfd319221bee918b31d448f685d2666902397e236665a849769a7b4`。检查点标记第 25 轮，可严格重载；重载后 16 张训练图像均得到有限且不同的连续输出。该检查只检查技术健康，不再调参。

## 最终轻量化复核

独立脚本在 `1×3×224×224` 输入、最终检查点下测得 **1,518,881 参数**，`fvcore` 已计数 **58,626,560 FLOPs**。未支持的逐元素算子按每输出元素 20 次运算保守补算，上界 **81,428,960 FLOPs**，无剩余未知算子。参数 ≤5M：**PASS**；保守 FLOPs ≤500M：**PASS**。与 PHASE 4 同结构结果完全一致。详细输出见 [`final_full_model_profile.txt`](final_full_model_profile.txt)。

## 官方盲测推理与提交校验

使用最终检查点 `model.eval()`、`torch.inference_mode()`、固定训练同款预处理，对全部 1,000 张官方 val 图片执行**一次**正式推理，成功 1,000、失败 0。没有随机增强、人工修改单张分数或以 val 调参。原始逐张结果保存在 [`val_predictions_raw.csv`](../runs/final_full_train/val_predictions_raw.csv)；正式 [`predictions.csv`](../submission/predictions.csv) 的 1,000 行文件名与预测数值逐行完全相同，仅按公开说明更换为中文表头。推理摘要见 [`val_inference_summary.json`](../runs/final_full_train/val_inference_summary.json)。

| 预测检查项 | 结果 |
| --- | ---: |
| Min | 0.0498400629 |
| Max | 0.8788049817 |
| Mean | 0.3638187693 |
| Std（总体） | 0.1332015968 |

预测有正常的数值变化，均为有限值；这里的统计**不是准确率**。由于无公开真值，不报告官方 val 的相关系数或误差。

[`validate_submission.py`](../scripts/validate_submission.py) 实际运行 **PASS**：UTF-8、表头 `图像文件名,预测分数`、恰好 1,000 行、与 val 文件名全集一致、按文件名字典序排列、无缺失/额外/重复、无空值或多余列、预测可转为数值且无 NaN/Inf。提交文件 SHA-256 为 `955a138489f40e1cb23a8187010ee29b4e59d255873e400a5cc40ed9379689ed`。`SUBMISSION READY = YES` 指这份 CSV 满足当前公开格式与本地校验；由于没有赛事官方样例，仍需在正式交付前核对课题组是否另发更具体模板。

项目既有 7 项单元测试再次运行并全部通过。未经授权，未上传、未发邮件，也未开始下一阶段。
