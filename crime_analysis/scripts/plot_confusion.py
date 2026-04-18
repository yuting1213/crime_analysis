"""
Confusion matrix + per-class metrics 繪圖工具。

接受多個 JSON（pilot_stats / vlm_diagnostic 格式），為每份產 row-normalized CM
（看 recall）+ 精準率/召回率/F1 表；若傳 >1 份則再畫一張 side-by-side 比較圖
供 thesis 直接用。

用法：
  cd crime_analysis
  # 單一實驗（label 從檔名自動推）
  python -m scripts.plot_confusion outputs/pilot_stats.json

  # 多模型比較（thesis 主圖）：LABEL=PATH 指定顯示名稱
  python -m scripts.plot_confusion \\
      Ours=outputs/pilot_stats.json \\
      Gemini=outputs/experiments/gemini_baseline/pilot_stats.json \\
      "FT 32B"=outputs/vlm_diagnostic_qwen3vl32b_finetuned.json \\
      --out outputs/confusion_compare.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def load_predictions(path: Path) -> List[Tuple[str, str]]:
    """Return list of (ground_truth, predicted) tuples."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [(e["ground_truth"], e["predicted"]) for e in entries]


def build_matrix(pairs: List[Tuple[str, str]], cats: List[str]):
    """Return raw confusion matrix (numpy) of shape (n_cats, n_cats)."""
    import numpy as np

    idx = {c: i for i, c in enumerate(cats)}
    cm = np.zeros((len(cats), len(cats)), dtype=int)
    for gt, pred in pairs:
        if gt in idx and pred in idx:
            cm[idx[gt], idx[pred]] += 1
    return cm


def per_class_metrics(cm) -> Dict[str, Dict[str, float]]:
    """Return {cat: {precision, recall, f1, support}} from confusion matrix."""
    import numpy as np

    n = cm.shape[0]
    metrics = {}
    for i, cat in enumerate(CATEGORIES[:n]):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[cat] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support,
        }
    return metrics


def draw_single(cm, ax, title: str, cats: List[str], normalize: bool = True) -> None:
    """Draw a single confusion matrix onto *ax*."""
    import numpy as np

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.where(row_sum > 0, cm / row_sum, 0.0)
        fmt = "{:.2f}"
        vmax = 1.0
    else:
        data = cm
        fmt = "{:d}"
        vmax = cm.max() if cm.max() > 0 else 1

    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)
    n = len(cats)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cats, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title, fontsize=11)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if val > 0:
                color = "white" if val > vmax * 0.55 else "black"
                ax.text(j, i, fmt.format(val if normalize else int(val)),
                        ha="center", va="center", color=color, fontsize=7)
    return im


def print_metrics_table(name: str, metrics: Dict[str, Dict[str, float]], n_total: int) -> None:
    """Write a terminal-friendly per-class metrics table."""
    print(f"\n=== {name}  (n={n_total}) ===")
    print(f"{'category':<16} {'P':>6} {'R':>6} {'F1':>6} {'support':>7}")
    p_sum = r_sum = f_sum = 0.0
    k = 0
    for cat, m in metrics.items():
        if m["support"] == 0:
            continue
        print(f"{cat:<16} {m['precision']:>6.2f} {m['recall']:>6.2f} "
              f"{m['f1']:>6.2f} {m['support']:>7d}")
        p_sum += m["precision"]; r_sum += m["recall"]; f_sum += m["f1"]
        k += 1
    if k:
        print(f"{'macro avg':<16} {p_sum / k:>6.2f} {r_sum / k:>6.2f} "
              f"{f_sum / k:>6.2f} {'':>7}")


def _parse_entry(arg: str) -> Tuple[str, Path]:
    """Accept either `LABEL=PATH` or bare `PATH`."""
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), Path(path.strip())
    p = Path(arg)
    return p.stem, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+",
                    help="Inputs as LABEL=PATH (or bare PATH to use filename).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG; default is alongside the first JSON.")
    ap.add_argument("--raw", action="store_true",
                    help="Show raw counts instead of row-normalized recall.")
    args = ap.parse_args()

    parsed = [_parse_entry(e) for e in args.entries]
    labels = [lbl for lbl, _ in parsed]
    paths = [p for _, p in parsed]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        return 2

    n_plots = len(paths)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows), squeeze=False)

    last_im = None
    for i, (path, label) in enumerate(zip(paths, labels)):
        if not path.exists():
            print(f"[skip] {path} missing", file=sys.stderr)
            continue
        pairs = load_predictions(path)
        # Use categories actually present in the file (handles Normal or subsets)
        cats_present = [c for c in CATEGORIES if any(gt == c or p == c for gt, p in pairs)]
        if not cats_present:
            cats_present = CATEGORIES
        cm = build_matrix(pairs, cats_present)
        metrics = per_class_metrics(cm)

        correct = sum(1 for gt, p in pairs if gt == p)
        title = f"{label}  —  {correct}/{len(pairs)} ({100 * correct / len(pairs):.1f}%)"
        ax = axes[i // cols][i % cols]
        last_im = draw_single(cm, ax, title, cats_present, normalize=not args.raw)

        print_metrics_table(label, metrics, len(pairs))

    # Blank-out unused subplots
    for k in range(len(paths), rows * cols):
        axes[k // cols][k % cols].axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.8)
        cbar.set_label("recall (row-normalised)" if not args.raw else "count")

    out = args.out or paths[0].parent / "confusion_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
