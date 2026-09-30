"""Train the locked Candidate P once on the full clean official training set."""

import csv
from hashlib import sha256
import logging
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights
import yaml

from src.dataset import FaceQualityDataset
from src.losses import pearson_loss, training_loss
from src.model import build_model
from train import set_seed


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "configs/final_locked.yaml"
MANIFEST = ROOT / "splits/final_train.csv"
RUN_DIR = ROOT / "runs/final_full_train"


def locked_config() -> dict:
    locked = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    if locked["lock_status"] != "LOCKED" or locked["candidate"] != "Candidate P":
        raise ValueError("Final model lock is absent")
    if locked["model"] != "mobilenet_v3_small" or locked["multi_scale"]:
        raise ValueError("Locked architecture changed")
    if locked["loss_type"] != "pearson" or locked["pearson_weight"] != 0.5:
        raise ValueError("Locked objective changed")
    with (ROOT / "reports/PHASE4_STABILITY.csv").open(newline="", encoding="utf-8") as stream:
        epochs = [int(row["Best Epoch"]) for row in csv.DictReader(stream) if row["Model"] == "Pearson"]
    if sorted(epochs) != [24, 25, 26]:
        raise ValueError(f"PHASE 4 best epochs changed: {epochs}")
    final_epochs = sorted(epochs)[len(epochs) // 2]
    config = dict(locked)
    config["train_split"] = "splits/final_train.csv"
    config["train_split_sha256"] = sha256(MANIFEST.read_bytes()).hexdigest()
    config["final_epochs"] = final_epochs
    unchanged_keys = set(locked) - {"train_split", "train_split_sha256"}
    if any(config[key] != locked[key] for key in unchanged_keys):
        raise ValueError("Unexpected change from locked config")
    return config


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler, scaler: torch.amp.GradScaler, device: torch.device,
                config: dict) -> tuple[float, float, float, float, int]:
    model.train()
    sums = [0.0, 0.0, 0.0]
    samples = 0
    max_gradient = 0.0
    skipped_overflow_steps = 0
    use_amp = bool(config["use_amp"] and device.type == "cuda")
    for images, labels, _names in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(images).squeeze(1)
            loss = training_loss(predictions, labels, config)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        with torch.no_grad():
            smooth = F.smooth_l1_loss(predictions.float(), labels.float())
            pearson = pearson_loss(predictions, labels, float(config["pearson_epsilon"]))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(gradient):
            if not use_amp or scaler.get_scale() >= scale_before:
                raise RuntimeError("Non-finite gradient was not handled by AMP scaler")
            skipped_overflow_steps += 1
        else:
            max_gradient = max(max_gradient, float(gradient))
        if scaler.get_scale() >= scale_before:
            scheduler.step()
        batch = len(labels)
        for index, value in enumerate((loss, smooth, pearson)):
            sums[index] += value.item() * batch
        samples += batch
    return *(total / samples for total in sums), max_gradient, skipped_overflow_steps


def main() -> None:
    config = locked_config()
    if (RUN_DIR / "final_model.pt").exists():
        raise FileExistsError("Final model already exists; refusing to overwrite")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                         handlers=[logging.FileHandler(RUN_DIR / "train.log", encoding="utf-8"),
                                   logging.StreamHandler()])
    (RUN_DIR / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FaceQualityDataset(MANIFEST, config["image_size"], config["horizontal_flip"],
                                 config["normalize_imagenet"])
    if len(dataset) != 27679:
        raise ValueError("Final dataset size changed")
    workers = int(config["num_workers"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]),
                        num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    cache_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    weight_hash = sha256(cache_file.read_bytes()).hexdigest()
    if weight_hash != config["pretrained_weights_sha256"]:
        raise ValueError("Pretrained ImageNet weights do not match the lock")
    logging.info("device=%s train=%d epochs=%d batch=%d pretrained=%s sha256=%s",
                 device, len(dataset), config["final_epochs"], config["batch_size"],
                 weights.name, weight_hash)
    model = build_model(config["use_pretrained"], config["pretrained_weights_path"],
                        config["multi_scale"]).to(device)
    head_names = ("classifier.3.", "shallow_projection.", "mid_projection.")
    head = [parameter for name, parameter in model.named_parameters()
            if name.startswith(head_names)]
    backbone = [parameter for name, parameter in model.named_parameters()
                if not name.startswith(head_names)]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": config["backbone_learning_rate"]},
         {"params": head, "lr": config["head_learning_rate"]}],
        weight_decay=config["weight_decay"])
    total_steps = int(config["final_epochs"]) * len(loader)
    warmup_steps = min(config["warmup_epochs"] * len(loader), total_steps - 1)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"] and device.type == "cuda")
    started = time.perf_counter()
    with (RUN_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("epoch", "train_loss", "smooth_l1", "pearson_loss", "backbone_lr",
                  "head_lr", "max_gradient_norm_before_clip", "amp_overflow_skipped_steps",
                  "samples", "epoch_seconds")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, int(config["final_epochs"]) + 1):
            epoch_started = time.perf_counter()
            loss, smooth, pearson, max_gradient, skipped = train_epoch(
                model, loader, optimizer, scheduler, scaler, device, config)
            row = {"epoch": epoch, "train_loss": loss, "smooth_l1": smooth,
                   "pearson_loss": pearson, "backbone_lr": optimizer.param_groups[0]["lr"],
                   "head_lr": optimizer.param_groups[1]["lr"],
                   "max_gradient_norm_before_clip": max_gradient,
                   "amp_overflow_skipped_steps": skipped, "samples": len(dataset),
                   "epoch_seconds": time.perf_counter() - epoch_started}
            writer.writerow(row)
            stream.flush()
            logging.info("epoch=%d/%d train_loss=%.6f smooth_l1=%.6f pearson=%.6f "
                         "lr=%.7f/%.7f max_grad=%.4f amp_skips=%d time=%.1fs", epoch,
                         config["final_epochs"], loss, smooth, pearson,
                         row["backbone_lr"], row["head_lr"], max_gradient, skipped,
                         row["epoch_seconds"])
    checkpoint = {"model_state": model.state_dict(), "epoch": config["final_epochs"],
                  "config": config, "train_loss": loss}
    torch.save(checkpoint, RUN_DIR / "final_model.pt")
    restored_checkpoint = torch.load(RUN_DIR / "final_model.pt", map_location="cpu",
                                     weights_only=True)
    restored = build_model(multi_scale=config["multi_scale"]).to(device)
    restored.load_state_dict(restored_checkpoint["model_state"])
    restored.eval()
    with torch.inference_mode():
        inputs = torch.stack([dataset[index][0] for index in range(16)]).to(device)
        outputs = restored(inputs).flatten()
    if not torch.isfinite(outputs).all() or torch.unique(outputs).numel() < 2:
        raise RuntimeError("Reloaded checkpoint failed training-image inference health check")
    logging.info("checkpoint_reloaded=true train_sample_outputs=%s elapsed_seconds=%.1f",
                 outputs.tolist(), time.perf_counter() - started)


if __name__ == "__main__":
    main()
