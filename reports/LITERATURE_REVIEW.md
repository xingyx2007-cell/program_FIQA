# PHASE 0 文献阅读与代码研究

本报告阅读[赛题索引的六篇论文](LITERATURE_INDEX.md)及其可确认的公开实现。这里的 **FIQA** 是给单张人脸图像预测质量分数；**rPPG** 是从连续视频中估计心跳相关的微弱颜色变化。rPPG 论文的任务与本赛题不同，不能把其心率结果当成人脸质量评分结果。**SROCC（也写 SRCC）**比较分数排序，**PLCC**比较线性对应；**Loss** 是训练中要减小的误差公式。**Backbone** 是提取图像 **Feature（特征）** 的网络主体。**Parameters** 是模型中的可学习数值数目，**FLOPs** 是前向计算量；视频模型的 MACs/帧与本赛题 224×224 单图 FLOPs 不可直接等同。其余初学者术语见[实验方案词汇表](EXPERIMENT_PLAN.md)。

## 1. EfficientPhys（WACV 2023）

[论文](https://openaccess.thecvf.com/content/WACV2023/html/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.html)；[团队工具箱](https://github.com/ubicomplab/rPPG-Toolbox)。

- **论文解决的问题：** 传统无接触脉搏测量常需要人脸定位、肤色区域等预处理；作者希望直接从视频有效、快速地读出心跳信号。
- **核心方法：** 在网络内部做帧间变化和归一化，再学习时间与空间线索。文中提出 CNN 版 EfficientPhys-C 和 Transformer 版 EfficientPhys-T。
- **输入和输出：** 连续人脸视频帧 → rPPG 波形，再从波形估计心率；不是单张图片 → 质量分数。
- **模型结构：** C 版是带内部差分/归一化的轻量卷积网络；T 版采用视频 Transformer。CNN（卷积神经网络）用局部滤波器逐层提取图像信息；Transformer 用注意力联系不同位置或时刻。
- **Loss：** C 版用负 Pearson 相关损失拟合波形；T 版采用 MSE。Pearson Loss 鼓励波形形状线性一致；不能直接证明它优化本赛题的 PLCC。
- **数据集和评价：** 论文用 AFRL、合成数据训练，比较 UBFC、PURE、MMSE 等；报告心率 MAE、RMSE、Pearson r 等，不报告本赛题 SROCC/PLCC。UBFC 上 C 版 MAE 约 1.14 bpm、r 约 0.99；与图像质量比赛分数不可比较。
- **参数量 / FLOPs：** 论文未报告这两项可直接用于本赛题核算的数字；报告了运行时间。
- **与本赛题的关系：** 对 rPPG 前端质量控制有背景价值，提示运动与光照干扰可能重要；不适合作为本赛题单图 FIQA 直接基线。

## 2. PhysFormer（CVPR 2022）

[论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.html)；[官方代码](https://github.com/ZitongYu/PhysFormer)。

- **论文解决的问题：** 视频中的生理信号弱，动作和光照容易掩盖心跳变化。
- **核心方法：** 时间差分引导的 Transformer 注意力，让网络聚焦跨帧的周期变化；同时约束时域波形和频域心率。
- **输入和输出：** 视频片段 → 一维 rPPG 波形/心率。
- **模型结构：** 视频帧特征提取、时间差分注意力 Transformer 和波形预测头；模型依赖时间维度。
- **Loss：** 负 Pearson 波形损失，加频率分类/分布约束；这些频域目标需要心率标注，本项目只有单图质量评分。
- **数据集和评价：** VIPL-HR、MAHNOB-HCI、MMSE-HR、OBF 等；指标为心率 MAE/RMSE/Pearson 等。VIPL-HR 的 MAE 约 4.97 bpm，不能与 FIQA 指标混用。
- **参数量 / FLOPs：** 论文报告约 **7.03M 参数、47.01 GFLOPs**（视频片段设定），已超过本赛题 5M 参数上限。
- **与本赛题的关系：** 有助于理解未来 rPPG 管线需要时间质量信息；本阶段单帧 FIQA 不采用该网络。

## 3. RhythmFormer（Pattern Recognition 2025）

[论文](https://doi.org/10.1016/j.patcog.2025.111511)；[官方代码](https://github.com/zizheng-guo/RhythmFormer)。

- **论文解决的问题：** 视频 rPPG 的周期规律可能被密集注意力和环境噪声稀释。
- **核心方法：** 原始帧与差分帧融合，以周期稀疏注意力选出有用的时空联系，再用时间金字塔整合多个时间尺度。多尺度的意思是同时利用粗、细两种以上的时间或空间信息。
- **输入和输出：** 连续视频帧 → 血容量脉搏波形/心率。
- **模型结构：** Fusion Stem、周期稀疏注意力模块、时域预测模块；不是单图回归网络。
- **Loss：** 时间波形负 Pearson 与频域分类损失组合。
- **数据集和评价：** PURE、UBFC、COHFACE、VIPL-HR、MMPD；心率 MAE/RMSE/MAPE/Pearson/SNR。MMPD 实验报告 MAE 约 3.07 bpm、Pearson r 约 0.86；未报告单图 FIQA 的 SROCC/PLCC。
- **参数量 / FLOPs：** 论文表中约 **3.25M 参数、240.55M MACs/帧（128×128）**。MACs 与 FLOPs 的计数口径和输入大小均不同，不能据此判定 224×224 时满足 500 MFLOPs。
- **与本赛题的关系：** 对 rPPG 质量控制的周期信息有价值；单图无法看到时间周期，故不移植核心模块。轻量化设计和干扰鲁棒性可作后续思路。

## 4. DSL-FIQA（CVPR 2024）

[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_DSL-FIQA_Assessing_Facial_Image_Quality_via_Dual-Set_Degradation_Learning_and_CVPR_2024_paper.html)；[项目页](https://dsl-fiqa.github.io/)；[官方代码](https://github.com/DSL-FIQA/DSL-FIQA)。

- **论文解决的问题：** 人脸质量受模糊、噪声、遮挡等多种退化影响，不同面部区域的重要性也不同。
- **核心方法：** 用合成与真实退化的双数据集学习退化表示，结合面部关键点引导局部区域评分。面部关键点是眼、鼻、口等大致位置。
- **输入和输出：** 单张人脸及检测出的关键点/局部裁剪 → 人工感知质量分数；它与本赛题输入输出最接近，但评分定义和数据集可能仍不同。
- **模型结构：** 预训练 ViT 特征、Swin 模块、退化编码器、关键点引导的解码/局部评分，并对多个裁剪结果汇总。ViT 是把图片拆成小块再用注意力联系各块的图像模型。
- **Loss：** 退化表征采用对比学习损失；质量分数采用 Charbonnier 鲁棒回归损失。对比学习让相关退化的表示更接近；Charbonnier 是平滑的绝对误差，对少量异常评分较稳。
- **数据集和评价：** GFIQA-20k、PIQ23、CGFIQA-40k；报告 PLCC/SRCC。例如 GFIQA-20k 的 PLCC/SRCC 为 0.9745/0.9740，PIQ23 为 0.7370/0.7333。跨数据集差异提示不能假定本赛题也会达到相同数值。
- **参数量 / FLOPs：** 论文未报告完整模型的参数量或 FLOPs。公开实现调用 ViT-Base，并含额外模块及多次局部推理；ViT 原论文的 Base 已有约 86M 参数，因此其完整实现显然不满足 5M 上限。
- **与本赛题的关系：** 最直接支持“关注质量退化与局部区域”的研究方向；可借鉴简单的鲁棒回归损失和区域感知想法，不能照搬整套模型。

## 5. ViT（ICLR 2021）

[论文](https://arxiv.org/abs/2010.11929)；[官方代码](https://github.com/google-research/vision_transformer)。

- **论文解决的问题：** 研究图像分类是否可以主要用 Transformer 完成，并从大规模预训练中受益。
- **核心方法：** 将图像切成固定大小 patch（小方块），编码后用注意力整合信息；先在大型数据集学习，再迁移到具体任务。
- **输入和输出：** 图片 → 类别概率，原论文并不直接输出 FIQA 分数。
- **模型结构：** patch embedding、若干 Transformer 层、分类头。
- **Loss：** 分类交叉熵，预测正确类别的概率越高越好；与质量分数回归不同。
- **数据集和评价：** ImageNet-21k、JFT-300M 预训练及 ImageNet 等分类测试；主要指标是分类准确率，而非 SROCC/PLCC。
- **参数量 / FLOPs：** ViT-Base **86M 参数**（Large 307M、Huge 632M）；论文未报告可直接用于本赛题 224×224 单图的 FLOPs。
- **与本赛题的关系：** 支持预训练和迁移学习的一般思路，但完整 ViT-Base 远超 5M 参数限制；优先选轻量 CNN。

## 6. DeepPhys（ECCV 2018）

[论文](https://openaccess.thecvf.com/content_ECCV_2018/html/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.html)；[后续工具箱实现](https://github.com/ubicomplab/rPPG-Toolbox)。

- **论文解决的问题：** 从视频估计脉搏与呼吸，减少动作、背景和光照干扰。
- **核心方法：** 外观分支找出有用区域，运动分支处理归一化帧差，注意力掩码让网络关注与生理信号相关的区域。
- **输入和输出：** 连续帧及帧差 → 脉搏/呼吸信号的时间变化，而非单图质量评分。
- **模型结构：** 双分支卷积注意力网络（CAN），具有外观与运动路径。
- **Loss：** 对目标波形变化使用 MSE 回归。
- **数据集和评价：** RGB Video I/II、MAHNOB-HCI、红外视频等；心率/呼吸率 MAE、RMSE、Pearson r 和 SNR。论文的运动场景结果显示注意力对抗运动干扰有益；不提供 FIQA 排序证据。
- **参数量 / FLOPs：** 论文未报告。
- **与本赛题的关系：** 人脸局部关注和运动干扰的背景思路有用；仅有单张图片，无法复用帧差分支。

## 官方代码实际结构与可借鉴内容

只阅读公开文件，未在本项目复制整套实现。下表中的文件名为仓库原路径；所有项目的原有数据读取格式和标注都与我们的无表头 `train.csv` 不同。

| 仓库 | 关键文件及职责 | 值得借鉴 | 不建议照搬 |
| --- | --- | --- | --- |
| [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) | [`main.py`](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/main.py) 分发训练/测试；[`dataset/data_loader/BaseLoader.py`](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/dataset/data_loader/BaseLoader.py) 预处理视频；[`neural_methods/model/EfficientPhys.py`](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/model/EfficientPhys.py)、[`DeepPhys.py`](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/model/DeepPhys.py) 实现网络；相应 `trainer/*Trainer.py` 做训练、验证、保存检查点；[`evaluation/metrics.py`](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/evaluation/metrics.py) 计算心率误差/相关。 | 数据、网络、训练、指标分开；固定配置并保存最佳检查点。 | 视频裁剪、波形标签、心率频谱和 HR 指标不适合单图质量回归；该指标文件不提供本赛题所需的 SROCC/PLCC 流程。 |
| [PhysFormer](https://github.com/ZitongYu/PhysFormer) | `train_Physformer_160_VIPL.py` 组织训练和模型保存；`Loadtemporal_data.py` 读视频片段；`model/Physformer.py` / `transformer_layer.py` 建网络；`TorchLossComputer.py` 实现波形与频率损失；`inference_OneSample_VIPL_PhysFormer.py` 推理。 | 明确区分训练、验证、推理及损失组成。 | 3D 视频 Transformer、频率标签和 >5M 参数不符合当前任务。 |
| [RhythmFormer](https://github.com/zizheng-guo/RhythmFormer) | `main.py` 入口；`dataset/data_loader/BaseLoader.py` 读视频；`neural_methods/model/RhythmFormer.py` 包含 Fusion Stem、周期模块；`neural_methods/trainer/RhythmFormerTrainer.py` 训练/验证/保存；`neural_methods/loss/TorchLossComputer.py` 组合损失；`evaluation/metrics.py` 算心率指标。 | 干扰鲁棒性思路和明确的验证/推理流程。 | 视频周期模块及 HR 频谱损失无法直接服务单图评分；128×128 MACs 不可充当本赛题 FLOPs 合规证明。 |
| [DSL-FIQA](https://github.com/DSL-FIQA/DSL-FIQA) | `train_iqa.py` 训练、Charbonnier、SciPy 的 Spearman/Pearson 验证；`models/iqa.py` 定义 ViT 与退化/区域模块；`utils/process_image.py` / `eval_process_image.py` 处理裁剪；`test.py` / `test_custom.py` 推理；`landmark_detection/landmark_detect.py` 检测关键点。 | 回归损失、SROCC/PLCC 计算、局部区域分析的实验设计；按完整验证集而非只看训练 Loss 选检查点。 | ViT-Base、Swin、关键点检测和多次裁剪造成很大计算开销；数据集与本赛题评分含义未证实一致。 |
| [ViT](https://github.com/google-research/vision_transformer) | `vit_jax/models_vit.py` 定义图像 Transformer；`vit_jax/input_pipeline.py` 读取/预处理；`vit_jax/train.py` 含分类损失和训练；`vit_jax/checkpoint.py` 管理预训练参数。 | 预训练权重加载与可复现配置的思路。 | 官方代码基于 JAX/Flax、分类任务且 ViT-Base 86M 参数；本项目使用 PyTorch 单值回归。 |

## 综合对比

| Paper | 核心思想 | Backbone | Loss | 是否直接报告 FIQA SROCC/PLCC | 轻量化判断 | 对本项目价值 |
| --- | --- | --- | --- | --- | --- | --- |
| EfficientPhys | 视频内差分估计心率 | CNN / 视频 Transformer | 负 Pearson / MSE | 否 | 参数/FLOPs 未报告 | rPPG 背景；直接价值低 |
| PhysFormer | 时间差分注意力 | 视频 Transformer | 波形相关 + 频域 | 否 | 7.03M 参数，超限 | 了解时间信号，当前不采用 |
| RhythmFormer | 周期稀疏注意力 | 视频时空网络 | 波形相关 + 频域 | 否 | 3.25M 参数但 FLOPs 口径不符 | 轻量化思路；直接价值低 |
| DSL-FIQA | 退化与局部面部区域 | ViT-Base + Swin | 对比 + Charbonnier | 是 | 全模型未报告，ViT-Base 已超限 | 直接任务启发最高，架构不可照搬 |
| ViT | 图像块注意力和预训练 | ViT-Base 等 | 分类交叉熵 | 否 | Base 86M 参数，超限 | 预训练思想 |
| DeepPhys | 外观/运动双分支注意力 | CNN | MSE | 否 | 参数/FLOPs 未报告 | rPPG 前端背景 |

## 对原模型方案的判断

1. **进入 baseline：** 轻量、带预训练权重的 MobileNetV3-Small 加单值输出，是比完整 ViT/视频 Transformer 更合理的起点；先用简单、稳健的回归损失，按整组 B 的 SROCC/PLCC 选模型。DSL-FIQA 直接支持关注退化与局部区域，代码也提供相关系数计算范例。
2. **留给消融实验：** 多尺度特征、局部面部区域、Pearson Loss、Ranking Loss 各自单独比较。消融是每次只改一个因素，看它是否真正带来收益。
3. **不符合或尚未证明符合预算：** PhysFormer 和 ViT-Base 参数已超 5M；DSL-FIQA 完整官方架构含 ViT-Base，也超限。RhythmFormer 虽报告 3.25M 参数，但其视频 MACs/帧不能证明本赛题的 224×224 单图 500 MFLOPs 限制。多个模型平均的总计算也必须计算在内。
4. **证据边界：** 六篇索引论文并未直接证明 MobileNetV3-Small、单图多尺度、Pearson Loss、Ranking Loss 的组合适用于本数据。补充的[同类 VQualA 2025 赛题论文](https://openaccess.thecvf.com/content/ICCV2025W/VQualA/papers/Ma_VQualA_2025_Challenge_on_Face_Image_Quality_Assessment_Methods_and_ICCVW_2025_paper.pdf)确实记录了 MobileNetV3-Small + L1RankLoss、MSE + Pearson 和 MobileNetV3 多层特征等**不同团队**的方案，但不能把它们拼接后当作已验证的统一模型。它们在不同训练尺寸、数据和硬件条件下产生，且有方案使用外部 GFIQA-20k 数据；本项目仍须在 B 上逐项验证，并核对比赛是否允许外部数据。
5. **更合理的调整：** 先建立可复现的轻量回归基线，再考虑复杂模块。评分中已有完全相同图片对应不同分数，稳健损失值得优先比较。身份信息缺失是评估可信度的主要限制；如果赛方后来提供身份/来源字段，应重做固定 Group Split，不能继续把现有划分解释为身份隔离。

这些建议是研究假设。PHASE 0 没有训练、模型指标或消融结论。
