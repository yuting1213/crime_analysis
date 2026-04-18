"""
從 VLM 診斷 JSON 計算 per-class bias correction priors。

公式：bias[c] = log(empirical_rate[c] / uniform_rate)

在 _classify_diagnostics 中，把這個值從 logit 減去 → 過度預測的類別被壓低、
沒人預測的類別被抬高。效果類似 Temperature-scaled balanced classifier。

用法：
    cd crime_analysis
    python -m scripts.compute_bias_priors \\
        --diagnostic outputs/vlm_diagnostic_qwen3vl.json \\
        --output data/bias_priors_qwen3vl.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def compute(diagnostic: list, smoothing: float = 0.5) -> dict:
    """
    Parameters
    ----------
    diagnostic : list of dict with "predicted" key
    smoothing : float
        Additive smoothing for categories never predicted. 0.5 = half-count,
        avoids log(0) and keeps correction finite.

    Returns
    -------
    dict: {cat: bias_value}
        Positive value → the model over-predicted this class; subtract from
        logit to suppress. Negative value → under-predicted; subtract (i.e.
        add in magnitude) to boost.
    """
    n = len(diagnostic)
    if n == 0:
        return {cat: 0.0 for cat in CRIME_CATEGORIES}

    uniform = 1.0 / len(CRIME_CATEGORIES)
    counts = Counter(entry["predicted"] for entry in diagnostic)
    total = n + smoothing * len(CRIME_CATEGORIES)

    bias = {}
    for cat in CRIME_CATEGORIES:
        empirical = (counts.get(cat, 0) + smoothing) / total
        bias[cat] = math.log(empirical / uniform)
    return bias


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic", required=True, type=Path,
                    help="Path to vlm_diagnostic_*.json")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write bias_priors JSON")
    ap.add_argument("--smoothing", type=float, default=0.5)
    args = ap.parse_args()

    entries = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    bias = compute(entries, smoothing=args.smoothing)
    counts = Counter(e["predicted"] for e in entries)
    n = len(entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bias, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 顯示分析
    print(f"Diagnostic: {args.diagnostic.name}  (n={n})")
    print(f"{'category':<16} {'count':>5} {'rate':>6} {'bias':>7}")
    for cat in CRIME_CATEGORIES:
        c = counts.get(cat, 0)
        rate = c / n if n else 0
        print(f"{cat:<16} {c:>5} {rate:>6.1%} {bias[cat]:>+7.3f}")

    print(f"\nOver-predicted (bias > 0):",
          sorted([c for c in CRIME_CATEGORIES if bias[c] > 0.3],
                 key=lambda c: -bias[c]))
    print(f"Under-predicted (bias < 0):",
          sorted([c for c in CRIME_CATEGORIES if bias[c] < -0.3],
                 key=lambda c: bias[c]))
    print(f"\nWrote → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
