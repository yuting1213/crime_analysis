"""
Anomaly detection metrics — binary "crime vs normal" evaluation.

Wraps sklearn for AUROC / DET curves, adds:
  - NDCF (Normalized Detection Cost Function, NIST SRE-style)
  - NDCF sensitivity sweep across cost ratios (thesis uses this to avoid
    committing to a single arbitrary C_miss/C_fa)
  - minimum_ndcf — finds the threshold τ that minimises NDCF, used to
    calibrate the 2-stage pipeline's anomaly gate.

Input convention throughout:
  scores : array-like of anomaly scores (higher = more anomalous)
  labels : array-like of 0/1, where 1 = Crime (positive/target), 0 = Normal
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Core scalar metrics
# ─────────────────────────────────────────────────────────────

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under ROC curve. Returns 0.5 if trivial / ill-defined."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=int)
    if y.min() == y.max():
        return 0.5  # degenerate — all one class
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def ndcf(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    *,
    c_miss: float = 5.0,
    c_fa: float = 1.0,
    p_target: float = 0.5,
) -> float:
    """
    Normalized Detection Cost Function evaluated at *threshold*.

        NDCF = (C_miss · P_target · P_miss + C_fa · (1 − P_target) · P_fa)
                / min(C_miss · P_target, C_fa · (1 − P_target))

    Returns 1.0 for a "do nothing" system, 0.0 for perfect.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)

    pos = y == 1
    neg = y == 0
    p_miss = float((pred[pos] == 0).mean()) if pos.any() else 0.0
    p_fa = float((pred[neg] == 1).mean()) if neg.any() else 0.0

    numer = c_miss * p_target * p_miss + c_fa * (1.0 - p_target) * p_fa
    denom = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return numer / denom if denom > 0 else float("nan")


def minimum_ndcf(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    c_miss: float = 5.0,
    c_fa: float = 1.0,
    p_target: float = 0.5,
) -> Tuple[float, float]:
    """
    Sweep thresholds (all unique score values + ±∞) and return (min_ndcf,
    optimal_threshold).
    """
    s = np.asarray(scores, dtype=float)
    thresholds = np.concatenate(
        [[-np.inf], np.unique(s), [np.inf]]
    )
    best = (float("inf"), float("nan"))
    for tau in thresholds:
        cost = ndcf(scores, labels, tau, c_miss=c_miss, c_fa=c_fa, p_target=p_target)
        if cost < best[0]:
            best = (cost, float(tau))
    return best


# ─────────────────────────────────────────────────────────────
# Curves
# ─────────────────────────────────────────────────────────────

def det_points(
    scores: Sequence[float], labels: Sequence[int]
) -> Dict[str, np.ndarray]:
    """
    DET curve points. Returns dict with:
      fpr, fnr  — matched arrays sorted by threshold ascending
      thresholds — corresponding thresholds (high → low)
    """
    from sklearn.metrics import det_curve

    y = np.asarray(labels, dtype=int)
    if y.min() == y.max():
        return {"fpr": np.array([]), "fnr": np.array([]), "thresholds": np.array([])}
    fpr, fnr, thresholds = det_curve(y, np.asarray(scores, dtype=float))
    return {"fpr": fpr, "fnr": fnr, "thresholds": thresholds}


def roc_points(
    scores: Sequence[float], labels: Sequence[int]
) -> Dict[str, np.ndarray]:
    """ROC curve (fpr, tpr) — companion to det_points."""
    from sklearn.metrics import roc_curve

    y = np.asarray(labels, dtype=int)
    if y.min() == y.max():
        return {"fpr": np.array([]), "tpr": np.array([]), "thresholds": np.array([])}
    fpr, tpr, thresholds = roc_curve(y, np.asarray(scores, dtype=float))
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


# ─────────────────────────────────────────────────────────────
# Sensitivity analysis (thesis thrust)
# ─────────────────────────────────────────────────────────────

DEFAULT_COST_RATIOS: Tuple[Tuple[float, float], ...] = (
    (1.0, 1.0), (2.0, 1.0), (5.0, 1.0), (10.0, 1.0),
)


def ndcf_sensitivity(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float | None = None,
    cost_ratios: Sequence[Tuple[float, float]] = DEFAULT_COST_RATIOS,
    p_target: float = 0.5,
) -> List[Dict]:
    """
    Compute NDCF at each (C_miss, C_fa) ratio.

    If *threshold* is None, each ratio uses its own optimum threshold
    (min_ndcf sweep) — this is the "best achievable" curve.
    If *threshold* is fixed, we evaluate every ratio at that single
    operating point — this is the "deployment" curve.

    Returns a list of dicts:
        [{c_miss, c_fa, ratio, ndcf, threshold, mode}, ...]
    """
    rows = []
    for c_miss, c_fa in cost_ratios:
        if threshold is None:
            val, tau = minimum_ndcf(
                scores, labels, c_miss=c_miss, c_fa=c_fa, p_target=p_target
            )
            mode = "optimal"
        else:
            val = ndcf(
                scores, labels, threshold,
                c_miss=c_miss, c_fa=c_fa, p_target=p_target,
            )
            tau = threshold
            mode = "fixed"
        rows.append({
            "c_miss": float(c_miss),
            "c_fa": float(c_fa),
            "ratio": float(c_miss / c_fa) if c_fa > 0 else float("inf"),
            "ndcf": float(val),
            "threshold": float(tau),
            "mode": mode,
        })
    return rows


# ─────────────────────────────────────────────────────────────
# Convenience: build binary task from pilot_stats entries
# ─────────────────────────────────────────────────────────────

def binary_task_from_stats(
    case_stats: Sequence[Dict],
    *,
    score_key: str = "escalation_score",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (scores, labels) arrays from case_stats entries.

    Each entry must carry:
      - ground_truth : str (== "Normal" for negatives, anything else = positive)
      - <score_key>  : float anomaly score (default: escalation_score)

    Entries missing the score are skipped.
    """
    scores, labels = [], []
    for s in case_stats:
        if score_key not in s:
            continue
        scores.append(float(s[score_key]))
        labels.append(0 if s.get("ground_truth") == "Normal" else 1)
    return np.asarray(scores), np.asarray(labels, dtype=int)
