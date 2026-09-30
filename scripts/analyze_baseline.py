"""Generate B-only predictions and learning curves from the saved best model."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import FaceQualityDataset
from src.metrics import quality_metrics
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/baseline_mbv3_pretrained")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if Path(config["val_split"]).name != "group_B.csv":
        raise ValueError("Only Group B is allowed for baseline analysis")

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(use_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dataset = FaceQualityDataset(ROOT / config["val_split"], config["image_size"],
                                 normalize_imagenet=config.get("normalize_imagenet", False))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config.get("num_workers", 0),
                        persistent_workers=config.get("num_workers", 0) > 0)
    rows = []
    with torch.inference_mode():
        for images, labels, filenames in loader:
            predictions = model(images.to(device)).squeeze(1).cpu().tolist()
            for filename, label, prediction in zip(filenames, labels.tolist(), predictions):
                rows.append({"filename": filename, "ground_truth": label,
                             "prediction": prediction, "absolute_error": abs(label - prediction)})
    assert len(rows) == len(dataset)
    measured = quality_metrics([row["ground_truth"] for row in rows],
                               [row["prediction"] for row in rows])
    for key in ("srocc", "plcc", "final_score", "mae", "rmse"):
        if abs(measured[key] - checkpoint["metrics"][key]) > 1e-6:
            raise RuntimeError(f"B prediction file disagrees with best checkpoint: {key}")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "baseline_groupB_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "ground_truth", "prediction", "absolute_error"])
        writer.writeheader()
        writer.writerows(rows)

    metrics = pd.read_csv(run_dir / "metrics.csv")
    best_epoch = int(checkpoint["epoch"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(metrics.epoch, metrics.train_loss, marker="o", markersize=3, label="A train loss")
    ax.plot(metrics.epoch, metrics.val_loss, marker="o", markersize=3, label="B validation loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="SmoothL1 loss", title="Baseline training and validation loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports / "baseline_training_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, label in (("srocc", "B SROCC"), ("plcc", "B PLCC"),
                          ("final_score", "B Final Score")):
        ax.plot(metrics.epoch, metrics[column], marker="o", markersize=3, label=label)
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Correlation", title="Group B validation metrics")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports / "baseline_metrics_curve.png", dpi=150)
    plt.close(fig)

    print("best_epoch", best_epoch, "B_samples", len(rows), "metrics", measured)
    print("largest_errors")
    for row in sorted(rows, key=lambda row: row["absolute_error"], reverse=True)[:15]:
        print(row)
    print("prediction_quantiles", dict(zip([0, .01, .1, .5, .9, .99, 1],
                                            np.quantile([row["prediction"] for row in rows],
                                                        [0, .01, .1, .5, .9, .99, 1]).tolist())))


if __name__ == "__main__":
    main()
