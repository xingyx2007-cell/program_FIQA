"""Small batch-only objectives for the independent Phase 3 experiments."""

import torch
from torch.nn import functional as F


def ranking_loss(predictions: torch.Tensor, targets: torch.Tensor,
                 epsilon_rank: float) -> torch.Tensor:
    """Penalize reversed pairs whose human-score gap is meaningful."""
    predictions, targets = predictions.float(), targets.float()
    true_gap = targets[:, None] - targets[None, :]
    pred_gap = predictions[:, None] - predictions[None, :]
    valid = torch.triu(torch.abs(true_gap) >= epsilon_rank, diagonal=1)
    if not valid.any():
        return predictions.sum() * 0.0
    return F.relu(-torch.sign(true_gap[valid]) * pred_gap[valid]).mean()


def pearson_loss(predictions: torch.Tensor, targets: torch.Tensor,
                 epsilon: float = 1e-8) -> torch.Tensor:
    predictions, targets = predictions.float(), targets.float()
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered.square().sum() + epsilon)
                             * (target_centered.square().sum() + epsilon))
    return 1.0 - numerator / denominator


def training_loss(predictions: torch.Tensor, targets: torch.Tensor, config: dict) -> torch.Tensor:
    base = F.smooth_l1_loss(predictions.float(), targets.float())
    loss_type = config.get("loss_type", "smoothl1")
    if loss_type == "ranking":
        return base + float(config["ranking_weight"]) * ranking_loss(
            predictions, targets, float(config["epsilon_rank"])
        )
    if loss_type == "pearson":
        return base + float(config["pearson_weight"]) * pearson_loss(
            predictions, targets, float(config.get("pearson_epsilon", 1e-8))
        )
    if loss_type != "smoothl1":
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return base
