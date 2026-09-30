"""Verify the frozen clean manifests and create the full training manifest."""

import csv
from hashlib import sha256
import math
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("image_id", "image_path", "quality_score")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected manifest columns: {path}")
        return list(reader)


def main() -> None:
    clean = read_rows(ROOT / "splits/clean_dataset.csv")
    groups = {name: read_rows(ROOT / f"splits/group_{name}.csv") for name in "ABC"}
    with (ROOT / "splits/excluded_samples.csv").open(newline="", encoding="utf-8") as stream:
        excluded = list(csv.DictReader(stream))
    if len(clean) != 27679 or len(excluded) != 7:
        raise ValueError("Clean or excluded count changed since PHASE 1")
    if [len(groups[name]) for name in "ABC"] != [9226, 9226, 9227]:
        raise ValueError("Frozen A/B/C sizes changed")
    clean_by_id = {row["image_id"]: row for row in clean}
    clean_paths = {row["image_path"] for row in clean}
    if len(clean_by_id) != len(clean) or len(clean_paths) != len(clean):
        raise ValueError("Duplicate ID or path in clean manifest")
    combined = [row for name in "ABC" for row in groups[name]]
    if len(combined) != len(clean) or {row["image_id"] for row in combined} != set(clean_by_id):
        raise ValueError("A/B/C union does not match clean manifest")
    if any(row != clean_by_id[row["image_id"]] for row in combined):
        raise ValueError("A/B/C values differ from clean manifest")
    excluded_ids = {row["image_id"] for row in excluded}
    excluded_paths = {row["image_path"] for row in excluded}
    if excluded_ids & clean_by_id.keys() or excluded_paths & clean_paths:
        raise ValueError("Excluded image is present in full training manifest")
    for index, row in enumerate(clean, 1):
        score = float(row["quality_score"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score: {row['image_id']}")
        relative = Path(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "train":
            raise ValueError(f"Unsafe training image path: {relative}")
        path = ROOT / relative
        if not path.is_file() or path.stem != row["image_id"]:
            raise ValueError(f"Missing or mismatched training image: {path}")
        with Image.open(path) as image:
            image.load()
        if index % 5000 == 0:
            print(f"decoded {index}/{len(clean)}", flush=True)
    output = ROOT / "splits/final_train.csv"
    if output.exists() and read_rows(output) != clean:
        raise ValueError("Existing final_train.csv differs from the verified clean set")
    if not output.exists():
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(clean)
    print(f"PASS: clean={len(clean)}, groups={[len(groups[name]) for name in 'ABC']}, excluded={len(excluded)}")
    print(f"final_train_sha256={sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
