"""Run exactly one locked-checkpoint evaluation on held-out Group C."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import FaceQualityDataset
from src.metrics import quality_metrics
from src.model import build_model


def main() -> None:
    config_path = ROOT / "configs/final_locked.yaml"
    output_path = ROOT / "reports/final_groupC_predictions.csv"
    receipt_path = ROOT / "reports/groupC_evaluation_receipt.json"
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("Group C evaluation output already exists; refusing a second evaluation")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if (config.get("lock_status") != "LOCKED" or
        config.get("model") != "mobilenet_v3_small" or
        config.get("multi_scale", False) or
        config.get("group_c_evaluation_limit") != 1 or
        config.get("test_split") != "splits/group_C.csv"):
        raise ValueError("Final configuration is not a locked, plain MobileNetV3-Small C test")
    if not (ROOT / "reports/FINAL_MODEL_DECISION.md").is_file():
        raise FileNotFoundError("The pre-C model decision report is missing")
    profile = {}
    for line in (ROOT / "reports/final_model_profile.txt").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            profile[key] = value
    if (int(profile["parameters"]) > 5_000_000 or
        int(profile["conservative_flops_upper_bound"]) > 500_000_000 or
        profile["remaining_unsupported_operators"] != "{}"):
        raise ValueError("Locked model does not pass the parameter/FLOPs gate")
    for split, expected in (("train_split", "train_split_sha256"),
                            ("val_split", "val_split_sha256")):
        if sha256((ROOT / config[split]).read_bytes()).hexdigest() != config[expected]:
            raise ValueError(f"Locked manifest changed: {split}")
    original_checkpoints = {
        "Candidate B": ("runs/baseline_mbv3_pretrained/best.pt", "smoothl1"),
        "Candidate P": ("runs/phase3_pearson/best.pt", "pearson"),
    }
    if (config.get("candidate") not in original_checkpoints or config.get("seed") != 20260930 or
        (config.get("checkpoint"), config.get("loss_type")) != original_checkpoints[config["candidate"]]):
        raise ValueError("Group C must use the predeclared original-seed checkpoint")
    checkpoint_path = ROOT / config["checkpoint"]
    actual_sha = sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_sha != config["checkpoint_sha256"]:
        raise ValueError("Locked checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != config["checkpoint_epoch"]:
        raise ValueError("Locked best epoch differs from checkpoint")
    original = checkpoint["config"]
    for key in ("seed", "image_size", "batch_size", "epochs", "backbone_learning_rate",
                "head_learning_rate", "weight_decay", "warmup_epochs",
                "early_stopping_patience", "gradient_clip_norm", "use_amp",
                "use_pretrained", "pretrained_weights_path", "normalize_imagenet",
                "horizontal_flip", "num_workers", "train_split", "val_split"):
        if config[key] != original[key]:
            raise ValueError(f"Locked training setting changed: {key}")
    if config["loss_type"] != original.get("loss_type", "smoothl1"):
        raise ValueError("Locked Loss differs from checkpoint training Loss")
    if config["loss_type"] == "pearson" and (
        config["pearson_weight"] != original["pearson_weight"] or
        float(config["pearson_epsilon"]) != float(original["pearson_epsilon"])
    ):
        raise ValueError("Locked Pearson parameters differ from checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(multi_scale=False).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    dataset = FaceQualityDataset(ROOT / config["test_split"], config["image_size"],
                                 normalize_imagenet=config["normalize_imagenet"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    rows = []
    with torch.inference_mode():
        for images, labels, filenames in loader:
            predictions = model(images.to(device, non_blocking=True)).squeeze(1).cpu().tolist()
            for filename, label, prediction in zip(filenames, labels.tolist(), predictions):
                rows.append({"filename": filename, "ground_truth": label,
                             "prediction": prediction, "absolute_error": abs(prediction - label)})
    if len(rows) != len(dataset) or len({row["filename"] for row in rows}) != len(rows):
        raise RuntimeError("Group C row count or filenames are inconsistent")
    metrics = quality_metrics([row["ground_truth"] for row in rows],
                              [row["prediction"] for row in rows])
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(config_bytes).hexdigest(),
        "checkpoint": config["checkpoint"], "checkpoint_sha256": actual_sha,
        "checkpoint_epoch": checkpoint["epoch"], "candidate": config["candidate"],
        "seed": config["seed"], "samples": len(rows),
        "predictions_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "group_b_metrics": checkpoint["metrics"], "group_c_metrics": metrics,
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
