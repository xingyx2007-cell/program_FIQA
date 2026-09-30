"""Collect real full-B predictions from one Phase 3 best checkpoint."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import FaceQualityDataset
from src.metrics import quality_metrics
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", choices=("phase3_ranking", "phase3_multiscale", "phase3_pearson"))
    args = parser.parse_args()
    run_dir = ROOT / "runs" / args.run_name
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if Path(config["train_split"]).name != "group_A.csv" or Path(config["val_split"]).name != "group_B.csv":
        raise ValueError("Phase 3 collection only permits A/B")
    dataset = FaceQualityDataset(ROOT / config["val_split"], config["image_size"],
                                 normalize_imagenet=config.get("normalize_imagenet", False))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"],
                        pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(multi_scale=config.get("multi_scale", False)).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    rows = []
    with torch.inference_mode():
        for images, labels, filenames in loader:
            output = model(images.to(device, non_blocking=True)).squeeze(1).cpu().tolist()
            for filename, label, prediction in zip(filenames, labels.tolist(), output):
                rows.append({"filename": filename, "ground_truth": label,
                             "prediction": prediction, "absolute_error": abs(prediction - label)})
    if len(rows) != len(dataset) or len({row["filename"] for row in rows}) != len(rows):
        raise RuntimeError("Full-B prediction count or filenames are inconsistent")
    metrics = quality_metrics([r["ground_truth"] for r in rows], [r["prediction"] for r in rows])
    for name, value in metrics.items():
        if not np.isclose(value, checkpoint["metrics"][name], atol=1e-6):
            raise RuntimeError(f"Checkpoint mismatch for {name}: {value}")
    with (run_dir / "groupB_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print({"run": args.run_name, "best_epoch": checkpoint["epoch"], "samples": len(rows), **metrics})


if __name__ == "__main__":
    main()
