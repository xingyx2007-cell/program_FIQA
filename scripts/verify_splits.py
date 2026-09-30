"""Check fixed manifest integrity without opening any Group C image."""

import csv
from hashlib import sha256
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"A": 9226, "B": 9226, "C": 9227}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and set(rows[0]) == {"image_id", "image_path", "quality_score"}
    return rows


def main() -> None:
    clean_path = ROOT / "splits/clean_dataset.csv"
    clean = read(clean_path)
    clean_by_id = {row["image_id"]: row for row in clean}
    assert len(clean_by_id) == len(clean) == sum(EXPECTED.values())
    seen_ids, seen_paths = set(), set()
    for group, expected in EXPECTED.items():
        rows = read(ROOT / f"splits/group_{group}.csv")
        assert len(rows) == expected, (group, len(rows), expected)
        for row in rows:
            image_id, image_path = row["image_id"], row["image_path"]
            assert image_id not in seen_ids and image_path not in seen_paths
            assert image_id in clean_by_id and row == clean_by_id[image_id]
            assert math.isfinite(float(row["quality_score"]))
            seen_ids.add(image_id)
            seen_paths.add(image_path)
        print(f"{group}: {len(rows)} valid records")
    assert seen_ids == set(clean_by_id)
    assert len(seen_paths) == len(clean)
    print("PASS: fixed split integrity; clean SHA-256", sha256(clean_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
