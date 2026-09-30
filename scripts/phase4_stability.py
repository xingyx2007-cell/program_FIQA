"""Verify the six fixed A/B runs and write the Phase 4 stability tables."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260930, 20261001, 20261002)
METRICS = ("srocc", "plcc", "final_score", "mae", "rmse")
OFFICIAL_WEIGHT_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
RUNS = {
    "Baseline": {20260930: "baseline_mbv3_pretrained",
                 20261001: "phase4_baseline_seed20261001",
                 20261002: "phase4_baseline_seed20261002"},
    "Pearson": {20260930: "phase3_pearson",
                20261001: "phase4_pearson_seed20261001",
                20261002: "phase4_pearson_seed20261002"},
}


def main() -> None:
    rows = []
    baseline_config = yaml.safe_load((ROOT / "configs/baseline_mbv3_pretrained.yaml").read_text(encoding="utf-8"))
    common_keys = [key for key in baseline_config if key not in ("run_name", "seed")]
    for model, runs_by_seed in RUNS.items():
        for seed in SEEDS:
            run_dir = ROOT / "runs" / runs_by_seed[seed]
            if f"sha256={OFFICIAL_WEIGHT_SHA256}" not in (run_dir / "train.log").read_text(encoding="utf-8"):
                raise RuntimeError(f"Official pretrained weight source not verified: {run_dir}")
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            config = checkpoint["config"]
            if config["seed"] != seed or any(config[key] != baseline_config[key] for key in common_keys):
                raise RuntimeError(f"Configuration mismatch: {run_dir}")
            if bool(config.get("multi_scale", False)) or config.get("loss_type", "smoothl1") != (
                "pearson" if model == "Pearson" else "smoothl1"
            ):
                raise RuntimeError(f"Model or loss mismatch: {run_dir}")
            if model == "Pearson" and (config.get("pearson_weight") != 0.5 or
                                       float(config.get("pearson_epsilon")) != 1e-8):
                raise RuntimeError(f"Pearson loss parameters differ: {run_dir}")
            history = pd.read_csv(run_dir / "metrics.csv")
            if not 1 <= len(history) <= 30 or not np.isfinite(
                history.select_dtypes(include="number").to_numpy()
            ).all():
                raise RuntimeError(f"Incomplete or non-finite training history: {run_dir}")
            best_row = history.loc[history["final_score"].idxmax()]
            if checkpoint["epoch"] != int(best_row["epoch"]):
                raise RuntimeError(f"Best epoch mismatch: {run_dir}")
            for metric in METRICS:
                if not np.isclose(checkpoint["metrics"][metric], best_row[metric], atol=1e-7):
                    raise RuntimeError(f"Saved metric differs from epoch record: {run_dir}: {metric}")
            rows.append({"Model": model, "Seed": seed, "SROCC": checkpoint["metrics"]["srocc"],
                         "PLCC": checkpoint["metrics"]["plcc"],
                         "Final Score": checkpoint["metrics"]["final_score"],
                         "MAE": checkpoint["metrics"]["mae"],
                         "RMSE": checkpoint["metrics"]["rmse"],
                         "Best Epoch": checkpoint["epoch"], "Run": runs_by_seed[seed]})
    frame = pd.DataFrame(rows)
    report_dir = ROOT / "reports"
    frame.to_csv(report_dir / "PHASE4_STABILITY.csv", index=False, float_format="%.10f")
    summary = frame.groupby("Model", sort=False).agg(
        **{f"Mean {metric}": (metric, "mean") for metric in ("SROCC", "PLCC", "Final Score", "MAE", "RMSE")},
        **{f"Std {metric}": (metric, "std") for metric in ("SROCC", "PLCC", "Final Score", "MAE", "RMSE")},
    ).reset_index()
    summary.to_csv(report_dir / "PHASE4_STABILITY_SUMMARY.csv", index=False, float_format="%.10f")
    paired = frame.pivot(index="Seed", columns="Model", values="Final Score")
    deltas = paired["Pearson"] - paired["Baseline"]
    lines = [
        "# PHASE 4：多随机种子稳定性复核", "",
        "只使用固定 Group A 训练和 Group B 验证。三个共同种子为 20260930、20261001、20261002。",
        "原种子的成绩直接读取 PHASE 2/3 已保存检查点；新增种子分别独立从相同官方 ImageNet 权重初始化。",
        "两候选除 Loss 外的配置逐项相同，最多训练 30 轮并使用相同早停规则；最佳轮次由完整 B 的 Final Score 确定。", "",
        "**mean** 是三次分数的平均值；**std** 是三次结果围绕平均值的离散程度。",
        "这里使用样本标准差（`ddof=1`）。三个种子能初步检查方向，但不足以保证未来所有训练都相同。", "",
        "## 每个种子的真实 B 组结果", "",
        "| Model | Seed | SROCC | PLCC | Final Score | MAE | RMSE | 最佳轮次 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {Model} | {Seed} | {SROCC:.6f} | {PLCC:.6f} | {Final Score:.6f} | "
                     "{MAE:.6f} | {RMSE:.6f} | {Best Epoch} |".format(**row))
    lines += ["", "## 平均值与波动", "",
              "| Model | SROCC mean ± std | PLCC mean ± std | Final Score mean ± std | MAE mean ± std | RMSE mean ± std |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in summary.iterrows():
        lines.append("| {Model} | {Mean SROCC:.6f} ± {Std SROCC:.6f} | "
                     "{Mean PLCC:.6f} ± {Std PLCC:.6f} | "
                     "{Mean Final Score:.6f} ± {Std Final Score:.6f} | "
                     "{Mean MAE:.6f} ± {Std MAE:.6f} | "
                     "{Mean RMSE:.6f} ± {Std RMSE:.6f} |".format(**row))
    lines += ["", "## 同种子成对比较", "",
              "| Seed | Pearson Final − Baseline Final |", "| ---: | ---: |"]
    for seed, delta in deltas.items():
        lines.append(f"| {seed} | {delta:+.6f} |")
    lines += ["", f"三组配对差值的平均数为 **{deltas.mean():+.6f}**；"
              f"Pearson 在 **{int((deltas > 0).sum())}/3** 个种子中领先。",
              "两种结构相同，参数量和推理 FLOPs 相同；本报告只描述 A/B 稳定性，未读取 Group C。", ""]
    (report_dir / "PHASE4_STABILITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame.to_string(index=False))
    print(summary.to_string(index=False))
    print("paired_delta_final", deltas.to_dict())


if __name__ == "__main__":
    main()
