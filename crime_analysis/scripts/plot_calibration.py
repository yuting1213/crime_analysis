"""
Reliability diagram + Expected Calibration Error（ECE）。

讀 pilot_stats.json 裡每 case 的 confidence / correct，分 bin 後：
  - 畫出 (mean confidence) vs (empirical accuracy) 的 reliability diagram
  - 疊 confidence 直方圖
  - 輸出 ECE（權重為 bin sample 數）

跑過第一次 pilot 拿到真實 confidence 後，thresholds (HARD=0.75 / SOFT=0.70)
可以根據此圖直接調整：找出 accuracy > 0.8 的最小 confidence 當 HARD 門檻。

用法：
  cd crime_analysis
  python -m scripts.plot_calibration outputs/pilot_stats.json

  # 多模型
  python -m scripts.plot_calibration \\
      Ours=outputs/pilot_stats.json \\
      Gemini=outputs/experiments/gemini_baseline/pilot_stats.json \\
      --out outputs/calibration.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple


def _parse_entry(arg: str) -> Tuple[str, Path]:
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), Path(path.strip())
    p = Path(arg)
    return p.stem, p


def load_confidence_pairs(path: Path) -> List[Tuple[float, bool]]:
    """Return (confidence, correct) list from pilot_stats.json style JSON."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for e in entries:
        if "confidence" not in e or "correct" not in e:
            continue
        out.append((float(e["confidence"]), bool(e["correct"])))
    return out


def compute_bins(pairs: List[Tuple[float, bool]], n_bins: int = 10):
    """
    Equal-width bins over [0, 1].
    Returns (bin_centers, bin_confs, bin_accs, bin_counts, ece).
    """
    if not pairs:
        return [], [], [], [], 0.0
    edges = [i / n_bins for i in range(n_bins + 1)]
    bin_confs = [0.0] * n_bins
    bin_accs = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for conf, correct in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_confs[idx] += conf
        bin_accs[idx] += 1.0 if correct else 0.0
        bin_counts[idx] += 1

    centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(n_bins)]
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_confs[i] /= bin_counts[i]
            bin_accs[i] /= bin_counts[i]
        else:
            bin_confs[i] = centers[i]
            bin_accs[i] = math.nan

    total = len(pairs)
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total) * abs(bin_accs[i] - bin_confs[i])

    return centers, bin_confs, bin_accs, bin_counts, ece


def suggest_thresholds(pairs: List[Tuple[float, bool]],
                       target_hard: float = 0.85,
                       target_soft: float = 0.70):
    """
    For each confidence quantile, compute empirical accuracy of samples with
    confidence ≥ threshold. Returns (hard_thr, soft_thr) — smallest threshold
    where cumulative-above accuracy crosses the target.
    """
    if not pairs:
        return None, None
    sorted_pairs = sorted(pairs, key=lambda x: -x[0])  # high → low
    n = len(sorted_pairs)
    running_correct = 0
    hard_thr = soft_thr = None
    for i, (conf, correct) in enumerate(sorted_pairs, 1):
        running_correct += int(correct)
        acc = running_correct / i
        if hard_thr is None and acc >= target_hard and i >= max(3, n // 4):
            hard_thr = conf
        if soft_thr is None and acc >= target_soft and i >= max(3, n // 3):
            soft_thr = conf
    return hard_thr, soft_thr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+",
                    help="Inputs as LABEL=PATH (or bare PATH).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    parsed = [_parse_entry(e) for e in args.entries]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        return 2

    n_plots = len(parsed)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(
        rows * 2, cols,
        figsize=(5.5 * cols, 5 * rows),
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1] * rows},
    )

    for i, (label, path) in enumerate(parsed):
        if not path.exists():
            print(f"[skip] {path} missing", file=sys.stderr)
            continue
        pairs = load_confidence_pairs(path)
        if not pairs:
            print(f"[skip] {path} has no confidence/correct pairs", file=sys.stderr)
            continue

        centers, bin_confs, bin_accs, bin_counts, ece = compute_bins(pairs, args.bins)
        hard_thr, soft_thr = suggest_thresholds(pairs)
        unique_confs = len(set(round(c, 3) for c, _ in pairs))

        # Top row: reliability diagram
        row = (i // cols) * 2
        col = i % cols
        ax_rel = axes[row][col]
        ax_rel.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        valid = [(c, a, n) for c, a, n in zip(bin_confs, bin_accs, bin_counts)
                 if n > 0 and not math.isnan(a)]
        if valid:
            x = [v[0] for v in valid]
            y = [v[1] for v in valid]
            sizes = [30 + 8 * v[2] for v in valid]
            ax_rel.scatter(x, y, s=sizes, alpha=0.65, edgecolor="navy")
            for cx, cy, cn in zip(x, y, [v[2] for v in valid]):
                ax_rel.annotate(f"n={cn}", (cx, cy), fontsize=7,
                                xytext=(3, 3), textcoords="offset points")
        overall_acc = sum(int(c) for _, c in pairs) / len(pairs)
        mean_conf = sum(conf for conf, _ in pairs) / len(pairs)
        ax_rel.axhline(overall_acc, color="green", ls=":", alpha=0.4,
                       label=f"acc={overall_acc:.2f}")
        ax_rel.axvline(mean_conf, color="red", ls=":", alpha=0.4,
                       label=f"conf={mean_conf:.2f}")
        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1.02)
        ax_rel.set_xlabel("Predicted confidence")
        ax_rel.set_ylabel("Empirical accuracy")
        degen = " [DEGENERATE — need re-run]" if unique_confs <= 2 else ""
        ax_rel.set_title(f"{label}  ECE={ece:.3f}  (n={len(pairs)}){degen}",
                         fontsize=10)
        ax_rel.legend(loc="upper left", fontsize=8)
        ax_rel.grid(alpha=0.3)

        # Bottom row: confidence histogram
        ax_hist = axes[row + 1][col]
        confs = [c for c, _ in pairs]
        ax_hist.hist(confs, bins=args.bins, range=(0, 1),
                     color="steelblue", alpha=0.7, edgecolor="black")
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel("Confidence")
        ax_hist.set_ylabel("Count")

        # Terminal report
        print(f"\n=== {label}  (n={len(pairs)}) ===")
        print(f"  accuracy      = {overall_acc:.3f}")
        print(f"  mean conf     = {mean_conf:.3f}")
        print(f"  ECE (10 bins) = {ece:.3f}")
        print(f"  unique conf   = {unique_confs}"
              + (" ← 目前硬編 0.7，修 confidence 後才有區分" if unique_confs <= 2 else ""))
        if hard_thr is not None:
            print(f"  HARD 建議     = {hard_thr:.3f}  (accuracy ≥ 0.85 above)")
        if soft_thr is not None:
            print(f"  SOFT 建議     = {soft_thr:.3f}  (accuracy ≥ 0.70 above)")

    # Blank unused
    for k in range(n_plots, rows * cols):
        r = (k // cols) * 2
        c = k % cols
        axes[r][c].axis("off")
        axes[r + 1][c].axis("off")

    out = args.out or parsed[0][1].parent / "calibration.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
