"""Read-only audit of the FIQA images and headerless training CSV."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def inspect_image(path: Path) -> dict:
    relative = path.relative_to(ROOT).as_posix()
    try:
        data = path.read_bytes()
        digest = sha256(data).hexdigest()
        with Image.open(BytesIO(data)) as image:
            size = image.size
            mode = image.mode
            image.load()  # Decode pixels so truncated files are detected.
        return {"path": relative, "hash": digest, "size": size, "mode": mode}
    except Exception as exc:
        return {"path": relative, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    csv_path = ROOT / "train.csv"
    # The source CSV has no header. Keep its six-digit IDs as strings.
    labels = pd.read_csv(
        csv_path,
        header=None,
        names=["image_id", "quality_score"],
        dtype={"image_id": "string"},
    )
    train = sorted(path for path in (ROOT / "train").rglob("*") if path.is_file())
    val = sorted(path for path in (ROOT / "val").rglob("*") if path.is_file())
    print(f"Inspecting {len(train)} train and {len(val)} val files", flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        inspected = list(pool.map(inspect_image, train + val))
    valid = [item for item in inspected if "error" not in item]
    errors = [item for item in inspected if "error" in item]

    train_names = {path.stem for path in train}
    label_ids = set(labels["image_id"].dropna())
    missing_images = sorted(label_ids - train_names)
    missing_labels = sorted(train_names - label_ids)
    repeated_ids = labels.loc[labels.image_id.duplicated(keep=False), "image_id"].tolist()
    missing_scores = int(labels.quality_score.isna().sum())
    bad_scores = int(pd.to_numeric(labels.quality_score, errors="coerce").isna().sum())

    hashes = defaultdict(list)
    for item in valid:
        hashes[item["hash"]].append(item["path"])
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    scores_by_id = labels.set_index("image_id")["quality_score"].to_dict()
    conflicting_duplicate_labels = [
        [(path, float(scores_by_id[Path(path).stem])) for path in paths]
        for paths in duplicates
        if len({scores_by_id[Path(path).stem] for path in paths if path.startswith("train/")}) > 1
    ]
    (REPORTS / "duplicate_groups.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    duplicate_pairs = sum(len(paths) - 1 for paths in duplicates)
    cross_train_val = [paths for paths in duplicates if any(p.startswith("train/") for p in paths) and any(p.startswith("val/") for p in paths)]

    train_valid = [item for item in valid if item["path"].startswith("train/")]
    val_valid = [item for item in valid if item["path"].startswith("val/")]
    train_res = Counter(tuple(item["size"]) for item in train_valid)
    val_res = Counter(tuple(item["size"]) for item in val_valid)
    train_modes = Counter(item["mode"] for item in train_valid)
    val_modes = Counter(item["mode"] for item in val_valid)
    train_pixels = [w * h / 1_000_000 for item in train_valid for w, h in [item["size"]]]
    val_pixels = [w * h / 1_000_000 for item in val_valid for w, h in [item["size"]]]
    non_numeric_ids = int((~labels.image_id.map(str.isdigit)).sum())
    score = labels.quality_score

    plt.figure(figsize=(8, 4.5))
    plt.hist(score, bins=40, color="#3769a7", edgecolor="white")
    plt.xlabel("Quality score")
    plt.ylabel("Image count")
    plt.title("Train label distribution")
    plt.tight_layout()
    plt.savefig(REPORTS / "label_distribution.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.logspace(np.log10(min(train_pixels + val_pixels)), np.log10(max(train_pixels + val_pixels)), 45)
    ax.hist(train_pixels, bins=bins, density=True, alpha=0.7, label="Train")
    ax.hist(val_pixels, bins=bins, density=True, alpha=0.55, label="Val")
    ax.set_xscale("log")
    ax.set_xlabel("Image size (megapixels, logarithmic scale)")
    ax.set_ylabel("Density")
    ax.set_title("Image resolution distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "resolution_distribution.png", dpi=150)
    plt.close(fig)

    lines = [
        "# FIQA 数据审计",
        "",
        "源数据只读检查；未修改或删除 `train/`、`val/` 或 `train.csv`。图片已完整解码检查并按 SHA-256 比较文件内容。",
        "",
        "## 结构与对应关系",
        "",
        f"- `train/`：{len(train)} 个文件；`val/val/`：{len(val)} 个文件。",
        f"- `train.csv`：{len(labels)} 行，**无表头**。按真实内容解析为第 1 列图片编号 `image_id`、第 2 列人工质量评分 `quality_score`。这两个名称是本项目赋予的列名，不是源文件的表头。",
        f"- 训练图片扩展名：{dict(Counter(p.suffix.lower() for p in train))}；官方 val 扩展名：{dict(Counter(p.suffix.lower() for p in val))}。CSV 编号对应 `train/<image_id>.png`。",
        f"- CSV 唯一图片编号：{labels.image_id.nunique()}；重复编号行：{len(repeated_ids)}。",
        f"- 图片 ID 均为 6 个字符，其中 {non_numeric_ids} 个含非数字字符（以 `z` 开头）；读取时必须保持字符串，不能转成整数或丢掉前导零。",
        f"- CSV 有编号但无训练图片：{len(missing_images)}；训练图片无 CSV 标签：{len(missing_labels)}。",
        f"- 缺失评分：{missing_scores}；非数字评分（含缺失）：{bad_scores}。",
        f"- 损坏/无法完整解码图片：{len(errors)}。",
        f"- SHA-256 完全相同的重复内容组：{len(duplicates)} 组，冗余文件数：{duplicate_pairs}；其中 train 与 val 跨集合重复组：{len(cross_train_val)}。",
        "- 重复内容组清单：[duplicate_groups.json](duplicate_groups.json)；后续划分需让同组文件留在同一组。",
        f"- 完全相同图片但评分不同：{len(conflicting_duplicate_labels)} 组；详见下方异常明细。这些评分差异保留在原始数据中，后续训练应把它们视为标签噪声。",
        "- CSV 没有 `identity`、`subject`、`source` 等分组列；图片路径也没有提供明确人物/来源目录。因此无法做可靠的身份分组划分。文件内容去重只能识别完全相同的图片，不能证明不同图片属于不同人。",
        "",
        "## 标签统计",
        "",
        f"- 最小值：{score.min():.8f}；最大值：{score.max():.8f}；均值：{score.mean():.8f}；标准差（样本）：{score.std():.8f}。",
        f"- 分位数：1%={score.quantile(.01):.8f}，10%={score.quantile(.1):.8f}，25%={score.quantile(.25):.8f}，50%={score.median():.8f}，75%={score.quantile(.75):.8f}，90%={score.quantile(.9):.8f}，99%={score.quantile(.99):.8f}。",
        "- 标签直方图：[label_distribution.png](label_distribution.png)。",
        "",
        "## 图片属性",
        "",
        f"- 训练图片模式：{dict(train_modes)}；官方 val 图片模式：{dict(val_modes)}。其中 RGB 为三通道彩色，L 为灰度，RGBA 为带透明通道彩色。",
        f"- 训练图片不同分辨率：{len(train_res)} 种；官方 val 不同分辨率：{len(val_res)} 种。",
        f"- 训练图片像素量（百万像素）：最小 {min(train_pixels):.3f}、中位 {np.median(train_pixels):.3f}、最大 {max(train_pixels):.3f}；官方 val：最小 {min(val_pixels):.3f}、中位 {np.median(val_pixels):.3f}、最大 {max(val_pixels):.3f}。",
        "- 训练图片最常见分辨率（宽×高，张数）：" + "；".join(f"{w}×{h}: {count}" for (w, h), count in train_res.most_common(10)) + "。",
        "- 官方 val 最常见分辨率（宽×高，张数）：" + "；".join(f"{w}×{h}: {count}" for (w, h), count in val_res.most_common(10)) + "。",
        "- 分辨率图：[resolution_distribution.png](resolution_distribution.png)（以像素总量比较训练集与官方 val；横轴为对数尺度）。",
        "",
        "## 异常明细",
        "",
        f"- CSV 有编号但无训练图片（最多 20 个）：{missing_images[:20]}。",
        f"- 训练图片无 CSV 标签（最多 20 个）：{missing_labels[:20]}。",
        f"- 无法解码图片（最多 20 个）：{errors[:20]}。",
        f"- 重复内容组（最多 20 组）：{duplicates[:20]}。",
        f"- 完全相同图片的冲突评分：{conflicting_duplicate_labels}。",
        "",
        "## 审计边界",
        "",
        "本次只检查文件、像素解码、尺寸、颜色模式、标签和完全相同的文件内容；没有运行人脸识别、近似重复图片检测，也没有人工评估评分质量。官方 `val/` 无公开标签，不能用于调参。",
        "",
    ]
    (REPORTS / "DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("rows", len(labels), "train", len(train), "val", len(val), flush=True)
    print("missing_images", len(missing_images), "missing_labels", len(missing_labels), "bad_scores", bad_scores, flush=True)
    print("corrupt", len(errors), "duplicate_groups", len(duplicates), "cross_train_val", len(cross_train_val), flush=True)
    print("train_modes", dict(train_modes), "val_modes", dict(val_modes), flush=True)
    print("train_res_top", train_res.most_common(5), flush=True)
    print("score_stats", score.min(), score.max(), score.mean(), score.std(), flush=True)


if __name__ == "__main__":
    main()
