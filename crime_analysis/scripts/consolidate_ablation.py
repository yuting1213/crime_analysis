"""
把 5 個 ablation variant + baseline（+ Gemini / standalone 32B FT 若有）的
pilot_stats.json 合成一張論文級對照表。

用法：
    cd crime_analysis

    # 假設所有 ablation 都在 outputs/ablations/ 下，baseline 是 pilot_v2
    python -m scripts.consolidate_ablation \\
        --root outputs/ablations \\
        --baseline outputs/pilot_v2 \\
        --extra Gemini=outputs/experiments/gemini_baseline \\
        --extra "FT 32B"=outputs/vlm_diagnostic_qwen3vl32b_finetuned.json

產出：
    outputs/ablations/ablation_table.md   # Markdown（對話／README 用）
    outputs/ablations/ablation_table.tex  # LaTeX（thesis 直接貼）
    outputs/ablations/ablation_diff.png   # Bar chart：各 variant 的 delta
    outputs/ablations/ablation_summary.json  # 結構化結果備查

指標欄：
    n, accuracy, macro-F1,
    AUROC, min-NDCF(5:1), min-NDCF(10:1),
    mean turns, mean Rcons, mean Rlegal_gt, ECE
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from crime_analysis.evaluation.detection_metrics import (  # noqa: E402
    auroc, minimum_ndcf, binary_task_from_stats,
)


CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def _load_stats(path: Path) -> List[Dict]:
    """Load pilot_stats.json (or vlm_diagnostic_*.json)."""
    if path.is_dir():
        path = path / "pilot_stats.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _macro_f1(stats: List[Dict]) -> float:
    """Per-class F1 averaged (macro)."""
    cats = sorted({s.get("ground_truth", "") for s in stats if s.get("ground_truth")})
    if not cats:
        return 0.0
    f1s = []
    for cat in cats:
        tp = sum(1 for s in stats if s.get("ground_truth") == cat
                 and s.get("predicted") == cat)
        fn = sum(1 for s in stats if s.get("ground_truth") == cat
                 and s.get("predicted") != cat)
        fp = sum(1 for s in stats if s.get("ground_truth") != cat
                 and s.get("predicted") == cat)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _ece(stats: List[Dict], n_bins: int = 10) -> float:
    """Expected Calibration Error over confidence."""
    valid = [(float(s["confidence"]), bool(s.get("correct", False)))
             for s in stats if "confidence" in s]
    if not valid:
        return 0.0
    bin_sums = [[0.0, 0.0, 0] for _ in range(n_bins)]  # [conf, acc, count]
    for c, corr in valid:
        idx = min(int(c * n_bins), n_bins - 1)
        bin_sums[idx][0] += c
        bin_sums[idx][1] += 1.0 if corr else 0.0
        bin_sums[idx][2] += 1
    n = len(valid)
    ece = 0.0
    for conf_sum, acc_sum, cnt in bin_sums:
        if cnt > 0:
            mean_conf = conf_sum / cnt
            mean_acc = acc_sum / cnt
            ece += (cnt / n) * abs(mean_acc - mean_conf)
    return ece


def _compute_row(name: str, stats: List[Dict]) -> Dict:
    """Return one row of the big comparison table."""
    if not stats:
        return {"name": name, "n": 0}

    correct = sum(1 for s in stats if s.get("correct"))
    turns = [s.get("total_turns", 0) for s in stats]
    rcons = [s.get("rcons", 0.0) for s in stats]
    rlegal_gt = [s.get("rlegal_gt", 0.0) for s in stats if "rlegal_gt" in s]

    # Detection metrics (if we have Normal samples + escalation_score)
    has_score = any("escalation_score" in s for s in stats)
    has_normal = any(s.get("ground_truth") == "Normal" for s in stats)
    auc = min_ndcf_51 = min_ndcf_101 = math.nan
    if has_score and has_normal:
        scores, labels = binary_task_from_stats(stats)
        if len(scores) > 0 and labels.min() != labels.max():
            auc = auroc(scores, labels)
            min_ndcf_51, _ = minimum_ndcf(scores, labels, c_miss=5, c_fa=1)
            min_ndcf_101, _ = minimum_ndcf(scores, labels, c_miss=10, c_fa=1)

    return {
        "name": name,
        "n": len(stats),
        "accuracy": correct / len(stats) if stats else 0.0,
        "macro_f1": _macro_f1(stats),
        "auroc": auc,
        "min_ndcf_5_1": min_ndcf_51,
        "min_ndcf_10_1": min_ndcf_101,
        "mean_turns": sum(turns) / len(turns) if turns else 0.0,
        "mean_rcons": sum(rcons) / len(rcons) if rcons else 0.0,
        "mean_rlegal_gt": sum(rlegal_gt) / len(rlegal_gt) if rlegal_gt else 0.0,
        "ece": _ece(stats),
    }


def _format_cell(value, width: int = 7, precision: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.{precision}f}"
    return str(value)


def _write_markdown(rows: List[Dict], out_path: Path) -> None:
    cols = [
        ("name",           "variant"),
        ("n",              "n"),
        ("accuracy",       "acc"),
        ("macro_f1",       "macro-F1"),
        ("auroc",          "AUROC"),
        ("min_ndcf_5_1",   "NDCF(5:1)"),
        ("min_ndcf_10_1",  "NDCF(10:1)"),
        ("mean_turns",     "turns"),
        ("mean_rcons",     "Rcons"),
        ("mean_rlegal_gt", "Rlegal_GT"),
        ("ece",            "ECE"),
    ]
    header = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for row in rows:
        cells = [_format_cell(row.get(k)) for k, _ in cols]
        lines.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(rows: List[Dict], out_path: Path) -> None:
    cols = [
        ("name", "Variant", "l"),
        ("accuracy", "Acc", "r"),
        ("macro_f1", "F1", "r"),
        ("auroc", "AUROC", "r"),
        ("min_ndcf_5_1", "NDCF$_{5:1}$", "r"),
        ("mean_rlegal_gt", "$R_{legal}^{GT}$", "r"),
        ("ece", "ECE", "r"),
    ]
    spec = "".join(c[2] for c in cols)
    head = " & ".join(c[1] for c in cols) + " \\\\\n\\midrule"
    body_lines = []
    for row in rows:
        cells = [row.get("name", "")] + [_format_cell(row.get(k)) for k, _, _ in cols[1:]]
        # escape underscores for LaTeX
        cells = [c.replace("_", "\\_") if isinstance(c, str) else c for c in cells]
        body_lines.append(" & ".join(cells) + " \\\\")
    body = "\n".join(body_lines)
    out = (
        f"\\begin{{tabular}}{{{spec}}}\n"
        "\\toprule\n"
        f"{head}\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    out_path.write_text(out, encoding="utf-8")


def _plot_diff(rows: List[Dict], out_path: Path, baseline_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    base = next((r for r in rows if r["name"] == baseline_name), None)
    if base is None or base.get("n") == 0:
        return

    metrics = [("accuracy", "acc"), ("macro_f1", "F1"),
               ("auroc", "AUROC"), ("mean_rlegal_gt", "Rlegal_GT")]
    variants = [r for r in rows if r["name"] != baseline_name and r.get("n", 0) > 0]
    if not variants:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    for ax, (key, label) in zip(axes, metrics):
        names = [r["name"] for r in variants]
        base_val = base.get(key) or 0.0
        vals = [(r.get(key) or 0.0) - base_val for r in variants]
        bars = ax.bar(names, vals)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(f"{label} (Δ vs {baseline_name})")
        ax.tick_params(axis="x", rotation=35)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:+.2f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_extra(arg: str) -> Tuple[str, Path]:
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), Path(path.strip())
    p = Path(arg)
    return p.stem, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="ablation 根目錄；掃描 <root>/<variant>/pilot_stats.json")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="baseline pilot 目錄 (e.g. outputs/pilot_v2)")
    ap.add_argument("--extra", action="append", default=[],
                    help="額外模型 LABEL=PATH（例如 Gemini=.../pilot_stats.json）")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-tex", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    rows: List[Dict] = []

    if args.baseline:
        bstats = _load_stats(args.baseline)
        rows.append(_compute_row("baseline", bstats))

    # Scan ablation root
    if args.root.exists():
        for sub in sorted(args.root.iterdir()):
            if sub.is_dir():
                stats = _load_stats(sub)
                if stats:
                    rows.append(_compute_row(sub.name, stats))

    # Extra external stats（Gemini / standalone FT / etc.）
    for arg in args.extra:
        label, path = _parse_extra(arg)
        stats = _load_stats(path)
        if stats:
            rows.append(_compute_row(label, stats))

    if not rows:
        print("[consolidate] 沒找到任何 pilot_stats.json，請檢查 --root / --baseline", file=sys.stderr)
        return 1

    # Print to terminal
    print(f"\n{'variant':<22} {'n':>4} {'acc':>6} {'F1':>6} {'AUROC':>6} "
          f"{'NDCF5':>6} {'Rleg':>6} {'ECE':>6}")
    print("-" * 75)
    for r in rows:
        print(
            f"{r['name']:<22} {r.get('n', 0):>4} "
            f"{_format_cell(r.get('accuracy'), precision=3):>6} "
            f"{_format_cell(r.get('macro_f1'), precision=3):>6} "
            f"{_format_cell(r.get('auroc'), precision=3):>6} "
            f"{_format_cell(r.get('min_ndcf_5_1'), precision=2):>6} "
            f"{_format_cell(r.get('mean_rlegal_gt'), precision=3):>6} "
            f"{_format_cell(r.get('ece'), precision=3):>6}"
        )

    # Write outputs
    args.root.mkdir(parents=True, exist_ok=True)
    out_md = args.out_md or args.root / "ablation_table.md"
    out_tex = args.out_tex or args.root / "ablation_table.tex"
    out_png = args.out_png or args.root / "ablation_diff.png"
    out_json = args.out_json or args.root / "ablation_summary.json"

    _write_markdown(rows, out_md)
    _write_latex(rows, out_tex)
    _plot_diff(rows, out_png, baseline_name="baseline")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved:\n  {out_md}\n  {out_tex}\n  {out_png}\n  {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
