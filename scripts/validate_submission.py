"""Validate the final blind-set CSV against the official image collection."""

import csv
from hashlib import sha256
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "val/val"
SUBMISSION = ROOT / "submission/predictions.csv"
OFFICIAL_FIELDS = ("图像文件名", "预测分数")


def main() -> None:
    if not SUBMISSION.is_file():
        raise FileNotFoundError(SUBMISSION)
    SUBMISSION.read_bytes().decode("utf-8")
    val_names = [path.name for path in VAL_DIR.iterdir() if path.is_file()]
    if len(val_names) != 1000 or len(set(val_names)) != 1000:
        raise ValueError("Unexpected official val file collection")
    sample = next((path for path in (ROOT / "sample_submission.csv",
                                     ROOT / "val/sample_submission.csv",
                                     ROOT / "submission/sample_submission.csv") if path.is_file()), None)
    if sample:
        with sample.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            expected_fields = tuple(reader.fieldnames or ())
            expected_order = [row[expected_fields[0]] for row in reader]
    else:
        expected_fields = OFFICIAL_FIELDS
        expected_order = sorted(val_names)
    with SUBMISSION.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, restkey="EXTRA")
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if fields != expected_fields:
        raise ValueError(f"Wrong columns: {fields!r}; expected {expected_fields!r}")
    if len(rows) != len(val_names):
        raise ValueError(f"Wrong row count: {len(rows)}")
    names = []
    for row in rows:
        if set(row) != set(expected_fields):
            raise ValueError("Missing or extra CSV field")
        name = row[fields[0]]
        if not name or name != name.strip():
            raise ValueError("Blank or whitespace-padded filename")
        try:
            value = float(row[fields[1]])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-numeric prediction for {name}") from error
        if not math.isfinite(value):
            raise ValueError(f"NaN or Inf prediction for {name}")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate image filename in submission")
    if set(names) != set(val_names):
        raise ValueError(f"Submission image mismatch: missing={set(val_names)-set(names)}, "
                         f"extra={set(names)-set(val_names)}")
    if names != expected_order:
        raise ValueError("Submission row order differs from sample or deterministic filename sort")
    print(f"PASS: rows={len(rows)} columns={fields} missing=0 extra=0 duplicates=0 "
          f"NaN=0 Inf=0 sha256={sha256(SUBMISSION.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
