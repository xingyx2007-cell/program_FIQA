"""Train the fixed A/B MobileNetV3-Small baseline, with optional tiny preflight."""

import argparse
import csv
from hashlib import sha256
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V3_Small_Weights
import yaml

from evaluate import evaluate_model
from src.dataset import FaceQualityDataset
from src.losses import training_loss
from src.model import build_model


ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def fixed_subset(dataset: FaceQualityDataset, count: int, seed: int) -> Subset:
    generator = np.random.default_rng(seed)
    indices = sorted(generator.choice(len(dataset), min(count, len(dataset)), replace=False).tolist())
    return Subset(dataset, indices)


def make_loaders(config: dict, device: torch.device, smoke: bool):
    train_data = FaceQualityDataset(
        ROOT / config["train_split"], config["image_size"], config.get("horizontal_flip", False),
        config.get("normalize_imagenet", False),
    )
    val_data = FaceQualityDataset(
        ROOT / config["val_split"], config["image_size"],
        normalize_imagenet=config.get("normalize_imagenet", False),
    )
    if smoke:
        train_data = fixed_subset(train_data, config["smoke_train_samples"], config["seed"])
        val_data = fixed_subset(val_data, config["smoke_val_samples"], config["seed"] + 1)
    workers = config.get("num_workers", 0)
    common = {"batch_size": config["batch_size"], "num_workers": workers,
              "pin_memory": device.type == "cuda", "persistent_workers": workers > 0}
    train_loader = DataLoader(train_data, shuffle=True,
                              generator=torch.Generator().manual_seed(config["seed"]), **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    return train_loader, val_loader


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler, scaler: torch.amp.GradScaler, device: torch.device,
                clip_norm: float, use_amp: bool, config: dict) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels, _filenames in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(images).squeeze(1)
            loss = training_loss(predictions, labels, config)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    return total_loss / total_samples


def run_training(config: dict, run_dir: Path, smoke: bool, device: torch.device) -> None:
    set_seed(int(config["seed"]))
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    train_loader, val_loader = make_loaders(config, device, smoke)
    epochs = 1 if smoke else int(config["epochs"])
    logging.info("device=%s smoke=%s pretrained=%s train=%d val=%d batch=%d amp=%s",
                 device, smoke, config["use_pretrained"], len(train_loader.dataset),
                 len(val_loader.dataset), config["batch_size"], config.get("use_amp", False))
    model = build_model(config["use_pretrained"], config.get("pretrained_weights_path"),
                        config.get("multi_scale", False)).to(device)
    if config["use_pretrained"]:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        cache_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        logging.info("torchvision_weights=%s url=%s sha256=%s",
                     weights.name, weights.url, sha256(cache_file.read_bytes()).hexdigest())

    head_names = ("classifier.3.", "shallow_projection.", "mid_projection.")
    head = [parameter for name, parameter in model.named_parameters()
            if name.startswith(head_names)]
    backbone = [parameter for name, parameter in model.named_parameters()
                if not name.startswith(head_names)]
    backbone_lr = config.get("backbone_learning_rate", config.get("learning_rate"))
    head_lr = config.get("head_learning_rate", config.get("learning_rate"))
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}],
        weight_decay=config["weight_decay"],
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = min(config.get("warmup_epochs", 0) * len(train_loader), total_steps - 1)
    if warmup_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
    use_amp = bool(config.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    patience = int(config.get("early_stopping_patience", epochs))
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    start = time.perf_counter()

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["epoch", "train_loss", "val_loss", "srocc", "plcc", "final_score",
                  "mae", "rmse", "samples", "backbone_lr", "head_lr", "epoch_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, scaler, device,
                config.get("gradient_clip_norm", 1.0), use_amp, config,
            )
            results = evaluate_model(model, val_loader, device)
            score = results["final_score"]
            if not np.isfinite(score):
                raise RuntimeError(f"Non-finite B Final Score at epoch {epoch}")
            improved = score > best_score
            if improved:
                best_score, best_epoch, stale_epochs = score, epoch, 0
            else:
                stale_epochs += 1
            row = {"epoch": epoch, "train_loss": train_loss, **results,
                   "backbone_lr": optimizer.param_groups[0]["lr"],
                   "head_lr": optimizer.param_groups[1]["lr"],
                   "epoch_seconds": time.perf_counter() - epoch_start}
            writer.writerow(row)
            stream.flush()
            checkpoint = {
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                "epoch": epoch, "config": config, "metrics": results,
                "train_loss": train_loss, "best_score": best_score, "best_epoch": best_epoch,
            }
            torch.save(checkpoint, run_dir / "last.pt")
            if improved:
                torch.save(checkpoint, run_dir / "best.pt")
            logging.info(
                "epoch=%d train_loss=%.6f B_srocc=%.6f B_plcc=%.6f B_final=%.6f "
                "B_mae=%.6f B_rmse=%.6f lr=%.7f/%.7f best_epoch=%d stale=%d time=%.1fs",
                epoch, train_loss, results["srocc"], results["plcc"], score,
                results["mae"], results["rmse"], row["backbone_lr"], row["head_lr"],
                best_epoch, stale_epochs, row["epoch_seconds"],
            )
            if not smoke and stale_epochs >= patience:
                logging.info("early_stopping: %d epochs without B Final Score improvement", patience)
                break

    logging.info("training_done best_epoch=%d best_score=%.6f elapsed=%.1fs",
                 best_epoch, best_score, time.perf_counter() - start)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    restored = build_model(multi_scale=config.get("multi_scale", False)).to(device)
    restored.load_state_dict(checkpoint["model_state"])
    repeated = evaluate_model(restored, val_loader, device)
    for name in ("srocc", "plcc", "final_score", "mae", "rmse"):
        if abs(repeated[name] - checkpoint["metrics"][name]) > 1e-6:
            raise RuntimeError(f"Reloaded best checkpoint changed {name}")
    logging.info("best_checkpoint_reloaded B_metrics=%s", repeated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_mbv3.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if Path(config["train_split"]).name != "group_A.csv" or Path(config["val_split"]).name != "group_B.csv":
        raise ValueError("This baseline only permits A training and B validation")
    if int(config["epochs"]) > 30:
        raise ValueError("Maximum 30 epochs for this baseline")
    name = config.get("run_name")
    run_dir = ROOT / "runs" / (
        f"{name}_preflight" if args.smoke and name else
        "baseline_smoke" if args.smoke else name or "baseline"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and config.get("use_amp", False):
        logging.info("CUDA unavailable: AMP disabled")
    try:
        run_training(config, run_dir, args.smoke, device)
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda" or int(config["batch_size"]) != 64:
            raise
        logging.warning("CUDA out of memory at batch_size=64; restarting from initialization at 32")
        torch.cuda.empty_cache()
        reduced = dict(config)
        reduced["batch_size"] = 32
        run_training(reduced, run_dir, args.smoke, device)


if __name__ == "__main__":
    main()
