"""
實驗結果分析腳本

用法：
    cd crime_analysis
    python -m scripts.analyze_results --experiments_dir outputs/experiments

功能：
    1. 跨實驗比較表格（Racc, Rcons, Rlegal, Rcost）
    2. 統計檢定（Paired t-test, Cohen's d）
    3. Radar chart（6 組 × 5 指標）
    4. Per-category breakdown
"""
import argparse
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_experiment(exp_dir: Path) -> Optional[List[Dict]]:
    """載入單一實驗的 pilot_stats.json。"""
    stats_path = exp_dir / "pilot_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(stats: List[Dict]) -> Dict:
    """從 pilot_stats 計算彙總指標。"""
    n = len(stats)
    if n == 0:
        return {}

    correct = sum(1 for s in stats if s.get("correct", False))
    racc = correct / n

    rcons_list = [s.get("rcons", 0.0) for s in stats]
    rlegal_list = [s.get("rlegal", 0.0) for s in stats]
    rcost_list = [s.get("rcost", 0.0) for s in stats]
    conf_list = [s.get("confidence", 0.0) for s in stats]
    turns_list = [s.get("total_turns", 0) for s in stats]

    return {
        "n": n,
        "Racc": round(racc, 4),
        "Rcons": round(statistics.mean(rcons_list), 4),
        "Rlegal": round(statistics.mean(rlegal_list), 4),
        "Rcost": round(statistics.mean(rcost_list), 4),
        "Confidence": round(statistics.mean(conf_list), 4),
        "Turns": round(statistics.mean(turns_list), 2),
        # 保留原始列表供統計檢定
        "_rcons": rcons_list,
        "_rlegal": rlegal_list,
        "_rcost": rcost_list,
    }


def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test，回傳 (t_stat, p_value)。"""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 1.0

    diffs = [a[i] - b[i] for i in range(n)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs) if len(diffs) > 1 else 1e-8
    t_stat = mean_d / (std_d / math.sqrt(n))

    # 近似 p-value（two-tailed, using normal approx for large n）
    # 精確版需要 scipy.stats.t，這裡用簡易近似
    try:
        from scipy import stats as sp_stats
        p_value = sp_stats.t.sf(abs(t_stat), df=n - 1) * 2
    except ImportError:
        # 簡易近似：|t| > 2 → p < 0.05
        p_value = 2 * math.exp(-0.5 * t_stat ** 2) if abs(t_stat) < 10 else 0.0

    return round(t_stat, 4), round(p_value, 6)


def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d 效果量。"""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0

    mean_a = statistics.mean(a[:n])
    mean_b = statistics.mean(b[:n])
    pooled_std = math.sqrt(
        (statistics.variance(a[:n]) + statistics.variance(b[:n])) / 2
    )
    if pooled_std < 1e-8:
        return 0.0

    return round((mean_a - mean_b) / pooled_std, 4)


def print_comparison_table(all_metrics: Dict[str, Dict]):
    """印出跨實驗比較表格。"""
    print("\n" + "=" * 90)
    print("  Cross-Experiment Comparison")
    print("=" * 90)

    header = f"{'Experiment':20s} | {'N':>4s} | {'Racc':>7s} | {'Rcons':>7s} | {'Rlegal':>7s} | {'Rcost':>7s} | {'Conf':>7s} | {'Turns':>5s}"
    print(header)
    print("-" * 90)

    for name, m in all_metrics.items():
        print(
            f"{name:20s} | {m['n']:4d} | {m['Racc']:7.1%} | {m['Rcons']:7.3f} | "
            f"{m['Rlegal']:7.3f} | {m['Rcost']:7.3f} | {m['Confidence']:7.3f} | {m['Turns']:5.1f}"
        )


def print_statistical_tests(all_metrics: Dict[str, Dict]):
    """印出 Ours vs 各消融的統計檢定。"""
    if "ours" not in all_metrics:
        print("\n[SKIP] 'ours' 實驗未找到，跳過統計檢定")
        return

    ours = all_metrics["ours"]

    print("\n" + "=" * 90)
    print("  Statistical Tests (Ours vs Ablation)")
    print("=" * 90)
    print(f"{'Comparison':30s} | {'Metric':>8s} | {'Δ Mean':>8s} | {'t-stat':>8s} | {'p-value':>10s} | {'Cohen d':>8s} | {'Sig':>3s}")
    print("-" * 90)

    for name, m in all_metrics.items():
        if name == "ours":
            continue

        for metric in ["_rcons", "_rlegal", "_rcost"]:
            label = metric.replace("_", "")
            a = ours.get(metric, [])
            b = m.get(metric, [])
            if not a or not b:
                continue

            n = min(len(a), len(b))
            delta = statistics.mean(a[:n]) - statistics.mean(b[:n])
            t, p = paired_ttest(a, b)
            d = cohens_d(a, b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            print(
                f"{'ours vs ' + name:30s} | {label:>8s} | {delta:+8.4f} | {t:8.4f} | {p:10.6f} | {d:8.4f} | {sig:>3s}"
            )


def save_radar_chart(all_metrics: Dict[str, Dict], output_dir: Path):
    """Radar chart（5 指標 × N 組）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib 未安裝，跳過 radar chart")
        return

    metrics = ["Racc", "Rcons", "Rlegal", "Confidence", "Turns_inv"]
    labels = ["Racc", "Rcons", "Rlegal", "Confidence", "1-Rcost"]

    # 準備數據（Rcost 取反，越小越好）
    data = {}
    for name, m in all_metrics.items():
        data[name] = [
            m.get("Racc", 0),
            m.get("Rcons", 0),
            m.get("Rlegal", 0),
            m.get("Confidence", 0),
            1 - m.get("Rcost", 0),
        ]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))

    for (name, values), color in zip(data.items(), colors):
        values_closed = values + values[:1]
        ax.plot(angles, values_closed, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values_closed, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_title("Experiment Comparison (Radar Chart)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    chart_path = output_dir / "radar_chart.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Radar chart → {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="實驗結果分析")
    parser.add_argument(
        "--experiments_dir", default="./outputs/experiments",
        help="實驗結果根目錄",
    )
    args = parser.parse_args()

    exp_base = Path(args.experiments_dir)
    if not exp_base.exists():
        logger.error(f"實驗目錄不存在：{exp_base}")
        return

    # 載入所有實驗
    all_metrics = {}
    for exp_dir in sorted(exp_base.iterdir()):
        if not exp_dir.is_dir():
            continue
        stats = load_experiment(exp_dir)
        if stats is None:
            logger.warning(f"跳過 {exp_dir.name}（無 pilot_stats.json）")
            continue

        metrics = compute_metrics(stats)
        if metrics:
            all_metrics[exp_dir.name] = metrics
            logger.info(f"載入 {exp_dir.name}: {metrics['n']} samples, Racc={metrics['Racc']:.1%}")

    if not all_metrics:
        logger.error("無實驗結果可分析")
        return

    # 輸出分析
    print_comparison_table(all_metrics)
    print_statistical_tests(all_metrics)
    save_radar_chart(all_metrics, exp_base)

    # 存 JSON 總結
    summary = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               for k, v in all_metrics.items()}
    with open(exp_base / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"比較總結 → {exp_base / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
