"""Unit tests for evaluation.detection_metrics — no GPU, no data."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from crime_analysis.evaluation.detection_metrics import (  # noqa: E402
    auroc, ndcf, minimum_ndcf, det_points, roc_points,
    ndcf_sensitivity, binary_task_from_stats, DEFAULT_COST_RATIOS,
)


def run():
    cases = []

    # ── 1. AUROC: perfect separation → 1.0 ──
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
    labels = [0,   0,   0,   1,   1,   1]
    assert auroc(scores, labels) == 1.0, "perfect auroc"
    cases.append("AUROC perfect = 1.0")

    # ── 2. AUROC: anti-correlated → 0.0 ──
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    labels = [0,   0,   0,   1,   1,   1]
    assert auroc(scores, labels) == 0.0, "anti auroc"
    cases.append("AUROC anti = 0.0")

    # ── 3. AUROC: degenerate (all same class) → 0.5 ──
    assert auroc([0.1, 0.2, 0.3], [1, 1, 1]) == 0.5, "degenerate auroc"
    cases.append("AUROC all positive → 0.5")

    # ── 4. NDCF: perfect system at threshold between classes → 0.0 ──
    # 3 pos at 0.9, 3 neg at 0.1, threshold=0.5 → P_miss=0, P_fa=0
    scores = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    labels = [1,   1,   1,   0,   0,   0]
    assert ndcf(scores, labels, 0.5) == 0.0, "perfect ndcf"
    cases.append("NDCF perfect = 0.0")

    # ── 5. NDCF: do-nothing (threshold=∞, never flag) → reject everything  ──
    #   P_miss=1, P_fa=0 → numerator = C_miss·P_target·1
    #   Normalized by min(C_miss·P_target, C_fa·(1-P_target))
    #   With default (5, 1, 0.5): numer = 2.5; denom = min(2.5, 0.5) = 0.5 → 5.0
    val = ndcf(scores, labels, threshold=np.inf)
    assert abs(val - 5.0) < 1e-9, f"do-nothing NDCF should be 5.0, got {val}"
    cases.append("NDCF never-flag (5:1) = 5.0")

    # ── 6. NDCF at symmetric cost (1:1): equal weight ──
    val_sym = ndcf(scores, labels, threshold=np.inf, c_miss=1, c_fa=1)
    assert abs(val_sym - 1.0) < 1e-9, f"1:1 never-flag NDCF should be 1.0, got {val_sym}"
    cases.append("NDCF never-flag (1:1) = 1.0")

    # ── 7. minimum_ndcf picks a threshold achieving 0 on separable data ──
    min_ndcf, tau = minimum_ndcf(scores, labels)
    assert min_ndcf == 0.0, f"min_ndcf should be 0 on separable data, got {min_ndcf}"
    assert 0.1 <= tau <= 0.9, f"optimal threshold should be between classes, got {tau}"
    cases.append(f"minimum_ndcf perfect: 0.0 @ τ={tau:.2f}")

    # ── 8. minimum_ndcf on noisy data — optimum is strictly positive ──
    rng = np.random.default_rng(42)
    pos_scores = rng.normal(0.8, 0.2, 40)
    neg_scores = rng.normal(0.4, 0.2, 40)
    s = np.concatenate([pos_scores, neg_scores])
    y = np.concatenate([np.ones(40), np.zeros(40)]).astype(int)
    min_ndcf, tau = minimum_ndcf(s, y)
    assert 0.0 < min_ndcf < 1.0, f"noisy min_ndcf should be in (0, 1), got {min_ndcf}"
    cases.append(f"minimum_ndcf noisy: {min_ndcf:.3f} @ τ={tau:.2f}")

    # ── 9. Sensitivity sweep: NDCF monotonically changes with C_miss ──
    ratios = [(1, 1), (2, 1), (5, 1), (10, 1), (20, 1)]
    scores = [0.9, 0.8, 0.7, 0.4, 0.5, 0.1]
    labels = [1,   1,   0,   1,   0,   0]
    rows = ndcf_sensitivity(scores, labels, threshold=0.65, cost_ratios=ratios)
    ndcf_vals = [r["ndcf"] for r in rows]
    # With threshold=0.65: pos(0.7 flagged, 0.4 missed, 0.9/0.8 flagged) → miss=1/3
    # neg(0.5/0.1 correctly not flagged, 0.7... wait 0.7 is labeled 0 above so fa=1/3)
    # Higher C_miss amplifies misses more → NDCF grows with ratio if P_miss > 0
    assert all(not math.isnan(v) for v in ndcf_vals), "no NaN in sensitivity"
    assert ndcf_vals[-1] >= ndcf_vals[0], \
        f"NDCF should grow with C_miss when misses exist: {ndcf_vals}"
    cases.append(f"sensitivity monotone: {[round(v,2) for v in ndcf_vals]}")

    # ── 10. Sensitivity at optimal threshold per ratio ──
    rows_opt = ndcf_sensitivity(scores, labels, cost_ratios=ratios)
    for r in rows_opt:
        assert r["mode"] == "optimal"
    # Optimal NDCF at any ratio ≤ fixed-threshold NDCF at that ratio
    for r_opt, r_fix in zip(rows_opt, rows):
        assert r_opt["ndcf"] <= r_fix["ndcf"] + 1e-9, \
            f"optimal {r_opt['ndcf']} must ≤ fixed {r_fix['ndcf']}"
    cases.append("sensitivity optimal ≤ fixed")

    # ── 11. DET/ROC arrays have matching lengths ──
    det = det_points(scores, labels)
    roc = roc_points(scores, labels)
    assert len(det["fpr"]) == len(det["fnr"]) == len(det["thresholds"])
    assert len(roc["fpr"]) == len(roc["tpr"]) == len(roc["thresholds"])
    assert len(det["fpr"]) > 0
    cases.append(f"DET/ROC arrays aligned: det={len(det['fpr'])} roc={len(roc['fpr'])}")

    # ── 12. binary_task_from_stats extracts correctly ──
    stats = [
        {"ground_truth": "Robbery", "escalation_score": 0.8},
        {"ground_truth": "Normal",  "escalation_score": 0.2},
        {"ground_truth": "Abuse",   "escalation_score": 0.6},
        {"ground_truth": "Normal",  "escalation_score": 0.3},
        {"ground_truth": "Robbery"},  # missing score → skipped
    ]
    scores, labels = binary_task_from_stats(stats)
    assert len(scores) == 4 and len(labels) == 4, "skip missing score"
    assert list(labels) == [1, 0, 1, 0], f"labels wrong: {labels}"
    assert list(scores) == [0.8, 0.2, 0.6, 0.3]
    cases.append("binary_task skips missing")

    # ── 13. Default cost ratios documented ──
    assert DEFAULT_COST_RATIOS == ((1,1),(2,1),(5,1),(10,1))
    cases.append("default ratios = 1/2/5/10:1")

    print(f"Ran {len(cases)} detection-metric tests:")
    for c in cases:
        print(f"  ✓ {c}")
    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    run()
