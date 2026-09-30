"""Create a training manifest without changing source images or train.csv."""

import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
REPORTS = ROOT / "reports"
SCORE_MIN = Decimal("0")
SCORE_MAX = Decimal("1")


def inspect(path: Path) -> tuple[str, str | None, str | None]:
    try:
        data = path.read_bytes()
        with Image.open(BytesIO(data)) as image:
            image.load()
            digest = sha256()
            digest.update(str(image.size).encode("ascii"))
            digest.update(image.mode.encode("ascii"))
            digest.update(image.tobytes())
        return path.stem, digest.hexdigest(), None
    except Exception as exc:
        return path.stem, None, f"{type(exc).__name__}: {exc}"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    image_paths = sorted((ROOT / "train").glob("*.png"))
    with (ROOT / "train.csv").open(newline="", encoding="utf-8-sig") as stream:
        raw_rows = list(csv.reader(stream))
    if any(len(row) != 2 for row in raw_rows):
        raise ValueError("train.csv must have exactly two fields per row")

    print(f"Checking {len(image_paths)} images", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        image_info = {stem: (digest, error) for stem, digest, error in pool.map(inspect, image_paths)}
    if len(image_info) != len(image_paths):
        raise ValueError("Repeated image filename stem in train/")

    id_counts = Counter(row[0] for row in raw_rows)
    excluded = []
    candidates = []
    csv_ids = set()
    for image_id, label_text in raw_rows:
        csv_ids.add(image_id)
        image_path = f"train/{image_id}.png"
        base = {"image_id": image_id, "image_path": image_path, "original_label": label_text}
        if id_counts[image_id] != 1:
            excluded.append({**base, "reason": "duplicate_csv_id", "duplicate_group": ""})
            continue
        if image_id not in image_info:
            excluded.append({**base, "reason": "missing_image", "duplicate_group": ""})
            continue
        digest, error = image_info[image_id]
        if error:
            excluded.append({**base, "reason": "unreadable_image", "duplicate_group": ""})
            continue
        try:
            score = Decimal(label_text)
        except InvalidOperation:
            score = Decimal("NaN")
        if not score.is_finite():
            excluded.append({**base, "reason": "missing_or_nonfinite_label", "duplicate_group": ""})
            continue
        if not SCORE_MIN <= score <= SCORE_MAX:
            excluded.append({**base, "reason": "label_outside_0_1", "duplicate_group": ""})
            continue
        candidates.append({"image_id": image_id, "image_path": image_path,
                           "quality_score": label_text, "score_decimal": score, "digest": digest})

    for image_id in sorted(set(image_info) - csv_ids):
        excluded.append({"image_id": image_id, "image_path": f"train/{image_id}.png",
                         "original_label": "", "reason": "missing_label", "duplicate_group": ""})

    by_digest = defaultdict(list)
    for row in candidates:
        by_digest[row["digest"]].append(row)
    clean = []
    for digest, rows in sorted(by_digest.items()):
        rows.sort(key=lambda row: row["image_id"])
        if len(rows) > 1 and len({row["score_decimal"] for row in rows}) > 1:
            for row in rows:
                excluded.append({"image_id": row["image_id"], "image_path": row["image_path"],
                                 "original_label": row["quality_score"],
                                 "reason": "duplicate_conflicting_score", "duplicate_group": digest})
            continue
        clean.append({key: rows[0][key] for key in ("image_id", "image_path", "quality_score")})
        for row in rows[1:]:
            excluded.append({"image_id": row["image_id"], "image_path": row["image_path"],
                             "original_label": row["quality_score"],
                             "reason": "duplicate_same_score", "duplicate_group": digest})

    clean.sort(key=lambda row: row["image_id"])
    excluded.sort(key=lambda row: (row["image_id"], row["reason"]))
    SPLITS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    write_csv(SPLITS / "clean_dataset.csv", ["image_id", "image_path", "quality_score"], clean)
    write_csv(SPLITS / "excluded_samples.csv",
              ["image_id", "image_path", "original_label", "reason", "duplicate_group"], excluded)

    reasons = Counter(row["reason"] for row in excluded)
    conflict_groups = len({row["duplicate_group"] for row in excluded
                           if row["reason"] == "duplicate_conflicting_score"})
    lines = [
        "# PHASE 1 数据清洗",
        "",
        "只生成清单；未修改或删除 `train/` 和无表头的 `train.csv`。对每张训练图完整解码，并按尺寸、颜色模式和像素内容的 SHA-256 比较；这样也能发现压缩方式不同但像素相同的图片。",
        "",
        f"- 原 CSV 行数：{len(raw_rows)}；训练图片文件数：{len(image_paths)}。",
        f"- 排除记录：{len(excluded)}；最终可训练记录：{len(clean)}。",
        f"- 原因计数：{dict(sorted(reasons.items()))}。",
        f"- 完全相同但评分冲突：{conflict_groups} 组，组内样本全部排除；不取平均值。",
        "- 完全相同且评分相同：每组保留图片编号字典序最小的一张，其他记录排除。",
        "- 合法评分检查采用 [0, 1]；源数据的实测评分全部在该区间。若赛方发布不同的正式范围，应先核对规则再调整。",
        "- 没有根据主观观感或低评分删除图片；低质量但标签有效的图片仍然保留。",
        "- `excluded_samples.csv` 保留原标签及排除原因；`duplicate_group` 是相同尺寸、模式与解码像素的完整 SHA-256 值。",
        "- 当前无 identity/subject/source 字段，无法确保同一人物的不同照片不跨组。",
        "",
        "| 图片 | 原标签 | 原因 | 重复组 SHA-256 |",
        "| --- | ---: | --- | --- |",
    ]
    for row in excluded:
        lines.append(f"| {row['image_path']} | {row['original_label']} | {row['reason']} | {row['duplicate_group']} |")
    lines += ["", "清洗结果：[clean_dataset.csv](../splits/clean_dataset.csv)、"
              "[excluded_samples.csv](../splits/excluded_samples.csv)。", ""]
    (REPORTS / "DATA_CLEANING.md").write_text("\n".join(lines), encoding="utf-8")
    print("clean", len(clean), "excluded", len(excluded), "reasons", dict(reasons), flush=True)


if __name__ == "__main__":
    main()
