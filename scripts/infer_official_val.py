"""Audit official blind images, then run the final checkpoint exactly once."""

import csv
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys

from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import build_model


VAL_DIR = ROOT / "val/val"
RUN_DIR = ROOT / "runs/final_full_train"
SUBMISSION = ROOT / "submission/predictions.csv"
OFFICIAL_FIELDS = ("图像文件名", "预测分数")


def audit_val() -> tuple[list[Path], dict[str, int]]:
    images = sorted(path for path in VAL_DIR.iterdir() if path.is_file())
    if len(images) != 1000 or any(path.suffix.lower() != ".png" for path in images):
        raise ValueError("Official val must contain exactly 1000 PNG images")
    if len({path.name.casefold() for path in images}) != len(images):
        raise ValueError("Duplicate official val filename")
    modes: Counter[str] = Counter()
    for path in images:
        with Image.open(path) as image:
            modes[image.mode] += 1
            image.load()
    return images, dict(modes)


def sample_order(images: list[Path]) -> tuple[list[str], tuple[str, str], str]:
    available = {path.name for path in images}
    for path in (ROOT / "sample_submission.csv", ROOT / "val/sample_submission.csv",
                 ROOT / "submission/sample_submission.csv"):
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                columns = tuple(reader.fieldnames or ())
                rows = list(reader)
            if len(columns) != 2 or len(rows) != len(images):
                raise ValueError("Sample submission schema or row count is invalid")
            names = [row[columns[0]] for row in rows]
            if set(names) != available or len(set(names)) != len(names):
                raise ValueError("Sample submission image set differs from official val")
            return names, columns, str(path.relative_to(ROOT))
    # The official page explicitly prints this two-column format and requires a header.
    return [path.name for path in images], OFFICIAL_FIELDS, "official webpage; filename sort"


def preprocess(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image = ImageOps.pad(image, (image_size, image_size),
                             method=Image.Resampling.BILINEAR, color=(128, 128, 128))
        return TF.normalize(TF.to_tensor(image),
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))


def main() -> None:
    if SUBMISSION.exists() or (RUN_DIR / "val_predictions_raw.csv").exists():
        raise FileExistsError("Official val inference already has output; refusing to rerun")
    images, modes = audit_val()
    order, columns, format_source = sample_order(images)
    config = yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
    checkpoint_path = RUN_DIR / "final_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != config["final_epochs"] or checkpoint["config"] != config:
        raise ValueError("Final checkpoint does not match the recorded run config")
    model = build_model(multi_scale=config["multi_scale"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictions = {}
    batch_size = int(config["batch_size"])
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_paths = images[start:start + batch_size]
            batch = torch.stack([preprocess(path, int(config["image_size"]))
                                 for path in batch_paths]).to(device)
            values = model(batch).flatten().cpu().tolist()
            for path, value in zip(batch_paths, values, strict=True):
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite official val prediction: {path.name}")
                predictions[path.name] = float(value)
    if len(predictions) != len(images):
        raise RuntimeError("A val image did not receive a prediction")
    values = list(predictions.values())
    stats = {"min": min(values), "max": max(values),
             "mean": statistics.fmean(values), "std": statistics.pstdev(values)}
    if stats["std"] < 1e-6 or max(abs(stats["min"]), abs(stats["max"])) > 1000:
        raise RuntimeError(f"Technically suspicious prediction distribution: {stats}")
    raw_path = RUN_DIR / "val_predictions_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_filename", "prediction"))
        writer.writerows((name, repr(predictions[name])) for name in order)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSION.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows((name, repr(predictions[name])) for name in order)
    summary = {"val_count": len(images), "predicted_count": len(predictions),
               "failed_count": 0, "image_modes": modes, "filename_sort": "lexicographic",
               "submission_columns": columns, "format_source": format_source,
               "prediction_stats": stats, "checkpoint_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
               "submission_sha256": sha256(SUBMISSION.read_bytes()).hexdigest()}
    (RUN_DIR / "val_inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
