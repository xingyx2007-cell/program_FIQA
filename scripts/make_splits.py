"""Create fixed, score-stratified A/B/C splits from the cleaned manifest."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260930
SPLIT_DIR = ROOT / "splits"


def main() -> None:
    clean_path = ROOT / "splits" / "clean_dataset.csv"
    source = pd.read_csv(
        clean_path,
        dtype={"image_id": "string", "quality_score": "string"},
    )
    assert list(source.columns) == ["image_id", "image_path", "quality_score"]
    assert source.image_id.is_unique and source.image_path.is_unique and source.notna().all().all()
    source["score_value"] = pd.to_numeric(source.quality_score, errors="raise")
    assert all((ROOT / path).is_file() for path in source.image_path)
    source["bin"] = pd.qcut(source.score_value, q=10, labels=False, duplicates="drop")

    ab, c = train_test_split(source, test_size=1 / 3, stratify=source.bin, random_state=SEED)
    a, b = train_test_split(ab, test_size=0.5, stratify=ab.bin, random_state=SEED)
    id_to_group = {image_id: group for frame, group in ((a, "A"), (b, "B"), (c, "C")) for image_id in frame.image_id}
    source["group"] = source.image_id.map(id_to_group)
    assert source.group.notna().all()

    SPLIT_DIR.mkdir(exist_ok=True)
    frames = {}
    for group in "ABC":
        frame = source.loc[source.group == group, ["image_id", "image_path", "quality_score"]].sort_values("image_id")
        frame.to_csv(SPLIT_DIR / f"group_{group}.csv", index=False)
        frames[group] = frame

    merged = pd.concat(frames.values(), ignore_index=True)
    assert len(merged) == len(source) == merged.image_id.nunique()
    assert set(merged.image_id) == set(source.image_id)
    assert not merged.image_path.duplicated().any()
    assert merged.set_index("image_id").quality_score.to_dict() == source.set_index("image_id").quality_score.to_dict()

    full_distribution = source.bin.value_counts(normalize=True)
    max_bin_gap = 0.0
    lines = [
        "# 固定 A/B/C 划分",
        "",
        f"随机种子：`{SEED}`。输入为 [clean_dataset.csv](../splits/clean_dataset.csv)，SHA-256：`{sha256(clean_path.read_bytes()).hexdigest()}`。划分文件使用 `image_id,image_path,quality_score`，路径相对项目根目录。",
        "",
        "源数据无身份或来源字段，无法可靠进行 identity/source Group Split。先排除冲突重复组，并只保留相同评分重复组的一个代表；再将清洗后的质量评分按十等分位分桶，按桶比例分层划分。",
        "",
        "| 组 | 样本数 | 评分均值 | 评分标准差 | 最小值 | 中位数 | 最大值 | 最大分桶比例差（相对全体） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in "ABC":
        frame = source[source.group == group]
        distribution = frame.bin.value_counts(normalize=True)
        gap = float((distribution - full_distribution).abs().max())
        max_bin_gap = max(max_bin_gap, gap)
        score = frame.score_value
        lines.append(
            f"| {group} | {len(frame)} | {score.mean():.6f} | {score.std():.6f} | "
            f"{score.min():.6f} | {score.median():.6f} | {score.max():.6f} | {gap:.4%} |"
        )
        print(group, len(frame), "mean", round(score.mean(), 6), "std", round(score.std(), 6), "max_decile_gap", round(gap, 6))
    lines += [
        "",
        f"合并后共 {len(merged)} 个唯一图片 ID，恰好覆盖完整的清洗后数据集；三组无重复 ID 或重复路径。清洗后无完全相同图片。最大分桶比例差为 {max_bin_gap:.4%}。",
        "",
        "注意：源数据没有人物身份，不能保证不同组没有同一人的不同照片。PHASE 1 的模型只可用 A 训练、B 验证；不得使用 C 或官方无标签 val。",
        "",
    ]
    assert max_bin_gap < 0.02, "Score distributions differ too much"
    (ROOT / "reports" / "SPLIT_PLAN.md").write_text("\n".join(lines), encoding="utf-8")
    print("PASS: disjoint, complete, cleaned, stratified")


if __name__ == "__main__":
    main()
