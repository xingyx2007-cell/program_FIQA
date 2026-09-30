"""Diagnostic synthetic degradations on a fixed subset of Group B.

The original labels describe the original images, not the degraded variants.
This script never trains or selects a checkpoint.
"""

import argparse
import csv
import hashlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.metrics import quality_metrics
from src.model import build_model


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("Original", "Exposure 0.5x", "Gaussian blur r=2", "JPEG quality=20", "Resolution 56px")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variants(image: Image.Image) -> list[Image.Image]:
    """Degrade the exact 224x224 padded input used by Candidate P."""
    small = image.resize((56, 56), Image.Resampling.BILINEAR)
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=20, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            jpeg = compressed.convert("RGB")
    return [image, ImageEnhance.Brightness(image).enhance(0.5),
            image.filter(ImageFilter.GaussianBlur(radius=2)), jpeg,
            small.resize(image.size, Image.Resampling.BILINEAR)]


class DiagnosticDataset(Dataset):
    def __init__(self, split: Path, sample_size: int, seed: int):
        with split.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or set(rows[0]) != {"image_id", "image_path", "quality_score"}:
            raise ValueError("Unexpected Group B manifest columns")
        if not (2 <= sample_size <= len(rows)):
            raise ValueError(f"sample_size must be between 2 and {len(rows)}")
        self.population = len(rows)
        indices = np.sort(np.random.default_rng(seed).choice(len(rows), sample_size, replace=False))
        self.rows = [rows[int(index)] for index in indices]
        if len({row["image_id"] for row in self.rows}) != sample_size:
            raise ValueError("Repeated image_id in selected Group B rows")
        for row in self.rows:
            relative = Path(row["image_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "train":
                raise ValueError(f"Unsafe image path: {relative}")
            row["full_path"] = ROOT / relative
            row["label"] = float(row["quality_score"])
            if not np.isfinite(row["label"]) or not row["full_path"].is_file():
                raise ValueError(f"Missing image or invalid label: {relative}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["full_path"]) as source:
            image = ImageOps.pad(source.convert("RGB"), (224, 224),
                                 method=Image.Resampling.BILINEAR, color=(128, 128, 128))
        images = torch.stack([TF.normalize(TF.to_tensor(item), MEAN, STD)
                              for item in variants(image)])
        return images, row["label"], row["image_id"]


def save_charts(results: list[dict], directory: Path) -> None:
    names = [row["condition"] for row in results]
    positions = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(11, 5.4), layout="constrained")
    for offset, key, label, color in [(-0.24, "srocc", "SROCC", "#2563a6"),
                                      (0, "plcc", "PLCC", "#19a79b"),
                                      (0.24, "final_score", "Final Score", "#ef9e43")]:
        ax.bar(positions + offset, [row[key] for row in results], width=0.23,
               label=label, color=color)
    ax.set_xticks(positions, names, rotation=10)
    ax.set_ylabel("Correlation with original image labels")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left")
    ax.set_title("Synthetic degradation diagnostic on the same 1,000 Group B images")
    fig.savefig(directory / "robustness_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    colors = ["#64748b"] + ["#2563a6" if row["mean_delta"] >= 0 else "#dc6258"
                            for row in results[1:]]
    axes[0].bar(positions, [row["mean_delta"] for row in results], color=colors)
    axes[0].axhline(0, color="#263648", linewidth=1)
    axes[0].set_ylabel("Mean prediction change vs original")
    axes[1].bar(positions, [100 * row["drop_fraction"] for row in results],
                color="#ef9e43")
    axes[1].set_ylabel("Images with a lower predicted score (%)")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xticks(positions, names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "robustness_prediction_shift.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260930)
    args = parser.parse_args()
    lock = yaml.safe_load((ROOT / "configs/final_locked.yaml").read_text(encoding="utf-8"))
    if lock["lock_status"] != "LOCKED" or lock["candidate"] != "Candidate P":
        raise RuntimeError("Candidate P is not locked")
    if (lock["model"] != "mobilenet_v3_small" or lock["multi_scale"] or
            not lock["normalize_imagenet"] or lock["horizontal_flip"] or
            lock["augmentation"] != "none"):
        raise RuntimeError("Unexpected locked model or preprocessing")
    if lock["val_split"] != "splits/group_B.csv" or lock["image_size"] != 224:
        raise RuntimeError("This diagnostic only permits Group B and 224px input")
    checkpoint_path = ROOT / lock["checkpoint"]
    split_path = ROOT / lock["val_split"]
    if sha256(checkpoint_path) != lock["checkpoint_sha256"]:
        raise RuntimeError("Locked checkpoint SHA-256 mismatch")
    if sha256(split_path) != lock["val_split_sha256"]:
        raise RuntimeError("Locked Group B manifest SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != lock["checkpoint_epoch"]:
        raise RuntimeError("Locked checkpoint epoch mismatch")

    data = DiagnosticDataset(split_path, args.sample_size, args.sample_seed)
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(multi_scale=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    labels = []
    predictions = [[] for _ in CONDITIONS]
    with torch.inference_mode():
        for images, targets, _ids in loader:
            batch = images.shape[0]
            output = model(images.reshape(batch * len(CONDITIONS), 3, 224, 224)
                           .to(device, non_blocking=True)).reshape(batch, len(CONDITIONS))
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite prediction")
            values = output.cpu().numpy()
            labels.extend(targets.tolist())
            for index in range(len(CONDITIONS)):
                predictions[index].extend(values[:, index].tolist())

    truth = np.asarray(labels, dtype=np.float64)
    clean = np.asarray(predictions[0], dtype=np.float64)
    results = []
    for condition, values in zip(CONDITIONS, predictions):
        pred = np.asarray(values, dtype=np.float64)
        metrics = quality_metrics(truth, pred)
        change = pred - clean
        results.append({"condition": condition, "samples": len(truth),
                        "srocc": metrics["srocc"], "plcc": metrics["plcc"],
                        "final_score": metrics["final_score"],
                        "mean_prediction": float(pred.mean()),
                        "mean_delta": float(change.mean()),
                        "median_delta_vs_clean": float(np.median(change)),
                        "drop_fraction": float(np.mean(change < 0)),
                        "rank_correlation_vs_clean": float(spearmanr(clean, pred).statistic)})
    if not all(np.isfinite(value) for row in results for key, value in row.items()
               if key not in ("condition", "samples")):
        raise RuntimeError("Non-finite diagnostic metric")

    directory = ROOT / "reports"
    with (directory / "robustness_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    save_charts(results, directory)
    sample_digest = hashlib.sha256("\n".join(row["image_id"] for row in data.rows).encode()).hexdigest()
    lines = [
        "# 合成退化鲁棒性诊断（PHASE 6）", "",
        "**性质：固定 Group B 子集的非训练型诊断。** 使用 PHASE 4 锁定的 Candidate P："
        f"`{lock['checkpoint']}`，第 {checkpoint['epoch']} 轮，SHA-256 `{lock['checkpoint_sha256']}`。"
        "模型仅由 Group A 训练，Group B 曾用于选轮次和调参；本结果**不是独立泛化成绩**。"
        "没有读取 Group C 或官方无标签 `val/`，没有训练、改权重或修改提交预测。", "",
        "## 固定样本和处理方法", "",
        f"从 `splits/group_B.csv` 的 {data.population:,} 行中，"
        f"使用 NumPy `default_rng({args.sample_seed})` 无放回选取 {args.sample_size:,} 行，"
        "将索引升序排列。所有条件使用**完全相同**的图片。"
        f"选中 image_id 顺序拼接（换行分隔）的 SHA-256：`{sample_digest}`；"
        f"Group B 清单 SHA-256：`{lock['val_split_sha256']}`。", "",
        "先按正式流程读为 RGB、保持宽高比并灰色填充至 224×224；随后仅对图像应用以下**确定性**退化，"
        "最后转 Tensor 并使用 ImageNet 均值和标准差归一化。无随机增强。", "",
        "- Exposure：全图亮度乘以 0.5（低曝光）。",
        "- Blur：Pillow GaussianBlur，半径 2 像素。",
        "- JPEG Compression：Pillow JPEG 质量 20、4:2:0 色度抽样。",
        "- Resolution degradation：224×224 双线性缩小到 56×56，再双线性放大回 224×224。", "",
        "推理统一为 `model.eval()`、`torch.inference_mode()`、FP32，无测试时增强。"
        "`scripts/robustness_eval.py` 可复跑相同步骤。", "",
        "## 结果", "",
        "| 条件 | SROCC¹ | PLCC¹ | Final¹ | 平均预测变化² | 降分占比² |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(f"| {row['condition']} | {row['srocc']:.6f} | {row['plcc']:.6f} | "
                     f"{row['final_score']:.6f} | {row['mean_delta']:+.6f} | "
                     f"{100 * row['drop_fraction']:.1f}% |")
    lines += [
        "", "**主要观察：** 在这组固定图像中，模糊、JPEG 压缩和降分辨率分别使 "
        f"{100 * results[2]['drop_fraction']:.1f}%、{100 * results[3]['drop_fraction']:.1f}%、"
        f"{100 * results[4]['drop_fraction']:.1f}% 的预测下降。"
        f"相反，0.5 倍曝光的平均预测变化为 {results[1]['mean_delta']:+.6f}，"
        f"仅 {100 * results[1]['drop_fraction']:.1f}% 的图片降分。"
        "这提示模型对低曝光可能存在不理想的响应；由于缺少退化后人工评分，不能断言每张图的正确目标分数。", "",
        "", "¹ SROCC、PLCC 和 Final Score 都以**原图人工评分**为参照。合成退化改变了图像实际质量，"
        "但没有重新获得人工评分。因此这些数值只是与原标签的相关性诊断，**不能解释为退化图的准确率**，"
        "也不能与正式 B/C 测试分数直接比较。", "",
        "² 每张退化图减去**同一张原图**的模型预测，再求平均；降分占比是退化后预测低于原图的图片比例。"
        "这衡量模型的响应方向，不证明新分数与人的主观质量判断一致。", "",
        "详细数值在 [robustness_results.csv](robustness_results.csv)；"
        "答辩图为 [相关系数](robustness_metrics.png) 和 [预测变化](robustness_prediction_shift.png)。", "",
        "## 解读边界", "",
        "退化强度是人为设定的单一档位，实际摄像头还可能同时存在运动模糊、噪声、遮挡和光照变化。"
        "这里没有新的人类标注、真实设备视频或 rPPG 信号质量标签，无法证明模型能改善 rPPG。"
        "Group B 是开发集，不能用此诊断重新选择模型、Loss、退化强度或阈值。", "",
        f"运行设备：`{device}`；PyTorch `{torch.__version__}`；Pillow `{Image.__version__}`。", "",
    ]
    (directory / "ROBUSTNESS.md").write_text("\n".join(lines), encoding="utf-8")
    for row in results:
        print(row)
    print(f"Wrote {directory / 'ROBUSTNESS.md'}")


if __name__ == "__main__":
    main()
