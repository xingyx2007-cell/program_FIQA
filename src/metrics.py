"""Metrics computed across the whole validation subset."""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def quality_metrics(labels, predictions) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(truth) != len(pred) or len(truth) < 2:
        raise ValueError("Metrics need at least two matching scores")
    if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
        raise ValueError("Metrics received NaN or Inf")
    srocc = float(spearmanr(truth, pred).statistic) if np.std(truth) and np.std(pred) else float("nan")
    plcc = float(pearsonr(truth, pred).statistic) if np.std(truth) and np.std(pred) else float("nan")
    diff = pred - truth
    return {
        "srocc": srocc,
        "plcc": plcc,
        "final_score": 0.5 * (srocc + plcc),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }
