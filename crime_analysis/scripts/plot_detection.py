"""
Anomaly detection plots — DET curve, ROC curve, NDCF sensitivity.

輸入一個或多個 pilot_stats.json（需帶 escalation_score + ground_truth）。

用法：
  cd crime_analysis

  # 單一實驗
  python -m scripts.plot_detection outputs/pilot_stats.json

  # 多模型
  python -m scripts.plot_detection \\
      Ours=outputs/pilot_stats.json \\
      Gemini=outputs/experiments/gemini_baseline/pilot_stats.json

  # 使用不同 score 欄位（例如用 VLM confidence 當 anomaly proxy）
  python -m scripts.plot_detection outputs/pilot_stats.json --score-key confidence

產出：
  detection.png — 左: DET 曲線（thesis 主圖）, 中: ROC, 右: NDCF sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from crime_analysis.evaluation.detection_metrics import (  # noqa: E402
    auroc, det_points, roc_points, minimum_ndcf, ndcf_sensitivity,
    binary_task_from_stats, DEFAULT_COST_RATIOS,
)


def _parse_entry(arg: str) -> Tuple[str, Path]:
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), Path(path.strip())
    p = Path(arg)
    return p.stem, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+",
                    help="Inputs as LABEL=PATH (or bare PATH).")
    ap.add_argument("--score-key", default="escalation_score",
                    help="JSON key to use as anomaly score (default: escalation_score)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ratios", default="1,2,5,10",
                    help="Comma-separated C_miss values at C_fa=1 for sensitivity plot")
    args = ap.parse_args()

    parsed = [_parse_entry(e) for e in args.entries]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        return 2

    c_miss_values = [float(x) for x in args.ratios.split(",")]
    cost_ratios = [(c, 1.0) for c in c_miss_values]

    fig, (ax_det, ax_roc, ax_sens) = plt.subplots(1, 3, figsize=(17, 5))

    all_good = False
    for label, path in parsed:
        if not path.exists():
            print(f"[skip] {path} missing", file=sys.stderr)
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        scores, labels = binary_task_from_stats(entries, score_key=args.score_key)
        if len(scores) == 0:
            print(f"[skip] {label}: no '{args.score_key}' field", file=sys.stderr)
            continue
        if labels.min() == labels.max():
            print(f"[skip] {label}: no Normal samples — run with --n-normal > 0",
                  file=sys.stderr)
            continue

        auc = auroc(scores, labels)
        min_cost_51, tau_51 = minimum_ndcf(scores, labels, c_miss=5, c_fa=1)
        sens = ndcf_sensitivity(scores, labels, cost_ratios=cost_ratios)

        # DET
        det = det_points(scores, labels)
        ax_det.plot(det["fpr"], det["fnr"],
                    label=f"{label}  AUROC={auc:.3f}")

        # ROC
        roc = roc_points(scores, labels)
        ax_roc.plot(roc["fpr"], roc["tpr"],
                    label=f"{label}  AUROC={auc:.3f}")

        # Sensitivity
        ratios = [r["c_miss"] for r in sens]
        values = [r["ndcf"] for r in sens]
        ax_sens.plot(ratios, values, marker="o", label=label)

        all_good = True
        print(f"\n=== {label}  (n_anomaly={int(labels.sum())}, "
              f"n_normal={int((labels == 0).sum())}) ===")
        print(f"  AUROC                       = {auc:.3f}")
        print(f"  min NDCF @ C_miss=5, C_fa=1 = {min_cost_51:.3f}  (τ={tau_51:.3f})")
        print("  NDCF sensitivity (optimal threshold per ratio):")
        for row in sens:
            print(f"    C_miss={row['c_miss']:>4.1f}  NDCF={row['ndcf']:.3f}  "
                  f"τ={row['threshold']:.3f}")

    if not all_good:
        print("No usable data to plot.", file=sys.stderr)
        return 1

    # DET axes: log-scale, standard presentation
    ax_det.set_xscale("log")
    ax_det.set_yscale("log")
    ax_det.set_xlabel("False Positive Rate (log)")
    ax_det.set_ylabel("False Negative Rate (log)")
    ax_det.set_title("DET curve (lower-left = better)")
    ax_det.grid(True, which="both", alpha=0.3)
    ax_det.legend(fontsize=9)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curve")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.legend(fontsize=9)
    ax_roc.grid(alpha=0.3)

    ax_sens.set_xlabel("C_miss / C_fa ratio")
    ax_sens.set_ylabel("min NDCF")
    ax_sens.set_xscale("log")
    ax_sens.set_title("NDCF sensitivity (optimal τ per ratio)")
    ax_sens.grid(True, alpha=0.3)
    ax_sens.legend(fontsize=9)

    out = args.out or parsed[0][1].parent / "detection.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
