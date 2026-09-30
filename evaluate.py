"""Evaluate a saved baseline on the local B development split."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml

from src.dataset import FaceQualityDataset
from src.metrics import quality_metrics
from src.model import build_model


ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.SmoothL1Loss(reduction="sum")
    labels, predictions = [], []
    total_loss = 0.0
    for images, targets, _filenames in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(images).squeeze(1)
        total_loss += criterion(output, targets).item()
        labels.extend(targets.cpu().tolist())
        predictions.extend(output.cpu().tolist())
    result = quality_metrics(labels, predictions)
    result["val_loss"] = total_loss / len(labels)
    result["samples"] = len(labels)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_mbv3.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/baseline_smoke/best.pt")
    parser.add_argument("--smoke", action="store_true", help="Use the same fixed B subset as train.py --smoke")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if Path(config["val_split"]).name != "group_B.csv":
        raise ValueError("Baseline evaluation only permits group_B.csv")
    dataset = FaceQualityDataset(ROOT / config["val_split"], config["image_size"],
                                 normalize_imagenet=config.get("normalize_imagenet", False))
    if args.smoke:
        import numpy as np

        generator = np.random.default_rng(config["seed"] + 1)
        count = min(config["smoke_val_samples"], len(dataset))
        dataset = Subset(dataset, sorted(generator.choice(len(dataset), count, replace=False).tolist()))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(multi_scale=config.get("multi_scale", False)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(evaluate_model(model, loader, device))


if __name__ == "__main__":
    main()
