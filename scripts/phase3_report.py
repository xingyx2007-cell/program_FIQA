"""Build the four-way ablation table and label-quantile error breakdown."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import quality_metrics


def checkpoint_metrics(run_name: str) -> tuple[int, dict]:
    ckpt = torch.load(ROOT / "runs" / run_name / "best.pt", map_location="cpu", weights_only=True)
    return ckpt["epoch"], ckpt["metrics"]


def main() -> None:
    report_dir = ROOT / "reports"
    baseline_profile = {}
    for line in (report_dir / "baseline_model_profile.txt").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition(": ")
        if sep:
            baseline_profile[key] = value
    multi_profile = json.loads((report_dir / "phase3_multiscale_profile.json").read_text(encoding="utf-8"))
    experiments = [
        ("Baseline", "baseline_mbv3_pretrained", "MobileNetV3-Small", "SmoothL1",
         report_dir / "baseline_groupB_predictions.csv", int(baseline_profile["parameters"]),
         int(baseline_profile["fvcore_counted_flops"])),
        ("Ranking", "phase3_ranking", "MobileNetV3-Small", "SmoothL1 + 0.25 Ranking",
         ROOT / "runs/phase3_ranking/groupB_predictions.csv", int(baseline_profile["parameters"]),
         int(baseline_profile["fvcore_counted_flops"])),
        ("Multi-scale", "phase3_multiscale", "MobileNetV3-Small + scale fusion", "SmoothL1",
         ROOT / "runs/phase3_multiscale/groupB_predictions.csv", multi_profile["parameters"],
         multi_profile["fvcore_counted_flops"]),
        ("Pearson", "phase3_pearson", "MobileNetV3-Small", "SmoothL1 + 0.5 Pearson",
         ROOT / "runs/phase3_pearson/groupB_predictions.csv", int(baseline_profile["parameters"]),
         int(baseline_profile["fvcore_counted_flops"])),
    ]
    ablation, segments, tails = [], [], []
    reference = None
    bins = None
    tail_cutoffs = None
    for name, run_name, model, loss, prediction_path, parameters, flops in experiments:
        epoch, saved_metrics = checkpoint_metrics(run_name)
        frame = pd.read_csv(prediction_path).sort_values("filename").reset_index(drop=True)
        if reference is None:
            reference = frame[["filename", "ground_truth"]].copy()
            bins = pd.qcut(reference["ground_truth"], q=3, labels=["Low", "Middle", "High"])
            tail_cutoffs = reference["ground_truth"].quantile([0.1, 0.9]).tolist()
        else:
            if not frame["filename"].equals(reference["filename"]) or not np.allclose(
                frame["ground_truth"], reference["ground_truth"], atol=1e-7
            ):
                raise RuntimeError(f"{name}: B sample IDs or labels differ from Baseline")
        metrics = quality_metrics(frame["ground_truth"], frame["prediction"])
        for metric, value in metrics.items():
            if not np.isclose(value, saved_metrics[metric], atol=1e-6):
                raise RuntimeError(f"{name}: {metric} differs from best.pt")
        ablation.append({"Experiment": name, "Model": model, "Loss": loss,
                         "SROCC": metrics["srocc"], "PLCC": metrics["plcc"],
                         "Final Score": metrics["final_score"], "MAE": metrics["mae"],
                         "RMSE": metrics["rmse"], "Params": parameters, "FLOPs": flops,
                         "Best Epoch": epoch, "B Samples": len(frame)})
        error = frame["prediction"] - frame["ground_truth"]
        for band in ("Low", "Middle", "High"):
            mask = bins == band
            band_error = error[mask]
            segments.append({"Experiment": name, "Band": band, "Samples": int(mask.sum()),
                             "Label Min": float(frame.loc[mask, "ground_truth"].min()),
                             "Label Max": float(frame.loc[mask, "ground_truth"].max()),
                             "MAE": float(band_error.abs().mean()),
                             "Mean Signed Error": float(band_error.mean()),
                             "Overestimate Fraction": float((band_error > 0).mean()),
                             "Underestimate Fraction": float((band_error < 0).mean())})
        for tail, mask in (("Bottom 10%", frame["ground_truth"] <= tail_cutoffs[0]),
                           ("Top 10%", frame["ground_truth"] >= tail_cutoffs[1])):
            tail_error = error[mask]
            tails.append({"Experiment": name, "Tail": tail, "Samples": int(mask.sum()),
                          "Cutoff": tail_cutoffs[0] if tail == "Bottom 10%" else tail_cutoffs[1],
                          "MAE": float(tail_error.abs().mean()),
                          "Mean Signed Error": float(tail_error.mean()),
                          "Overestimate Fraction": float((tail_error > 0).mean()),
                          "Underestimate Fraction": float((tail_error < 0).mean())})
    pd.DataFrame(ablation).to_csv(report_dir / "ablation_phase3.csv", index=False, float_format="%.10f")
    pd.DataFrame(segments).to_csv(report_dir / "phase3_segment_metrics.csv", index=False, float_format="%.10f")
    pd.DataFrame(tails).to_csv(report_dir / "phase3_tail_metrics.csv", index=False, float_format="%.10f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    names = [row["Experiment"] for row in ablation]
    x = np.arange(len(names))
    width = 0.25
    for key in ("SROCC", "PLCC", "Final Score"):
        axes[0].plot(x, [row[key] for row in ablation], marker="o", linestyle="none",
                     markersize=8, label=key)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0.91, 0.95)
    axes[0].set_title("Full Group B correlation")
    axes[0].legend()
    for index, band in enumerate(("Low", "Middle", "High")):
        axes[1].bar(x + (index - 1) * width,
                    [next(s["MAE"] for s in segments if s["Experiment"] == name and s["Band"] == band)
                     for name in names], width, label=band)
    axes[1].set_xticks(x, names)
    axes[1].set_title("MAE by B label quantile")
    axes[1].legend()
    fig.savefig(report_dir / "ablation_phase3.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(ablation).to_string(index=False))
    print(pd.DataFrame(segments).to_string(index=False))
    print(pd.DataFrame(tails).to_string(index=False))


if __name__ == "__main__":
    main()
