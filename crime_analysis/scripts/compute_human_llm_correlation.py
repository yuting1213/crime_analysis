"""
compute_human_llm_correlation.py — 人類 vs LLM 裁判的一致性分析

讀人類填完的 human_scores.csv + cross_eval 的 per_report.json，計算：
  1. ICC(2,k) — 人類 vs 各 LLM 裁判（事件摘要、overall 等）
  2. Pearson / Spearman 相關
  3. 人類間 ICC（若有多位評分者）

用法：
    cd crime_analysis
    python -m scripts.compute_human_llm_correlation \\
        --human outputs/cross_eval/pilot_v5/human_sample/human_scores.csv \\
        --cross-eval outputs/cross_eval/pilot_v5/per_report.json \\
        --output outputs/cross_eval/pilot_v5/human_sample/validation.json
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.llm_judge import RUBRIC_QUESTIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_human_scores(path: Path) -> Dict:
    """
    human_scores.csv 格式：sample_idx, video_id, model, q_key, score
    多位評分者合併在同一檔 → 加 rater 欄位也可；此版本只支援單評分者。
    回傳 {(video_id, model): {q_key: normalized_score}}.
    """
    scores = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            model = row["model"]
            q_key = row["q_key"]
            raw = float(row["score"])
            max_score = RUBRIC_QUESTIONS[q_key]["max_score"] or 4
            scores[(vid, model)][q_key] = raw / max_score
    return dict(scores)


def load_llm_scores(cross_eval_path: Path) -> Dict:
    """
    從 per_report.json 抽出每個裁判的 q_norm。
    回傳：
      {(video_id, model): {judge: {q_key: norm, ..., "overall": 0-1}}}.

    Schema 對應 run_cross_evaluation.py 寫入的結構：
      entry["qwen_scores"]   = {judge_key: {q_norm, overall, ...}}
      entry["gemini_scores"] = {judge_key: {q_norm, overall, ...}}
    """
    data = json.loads(cross_eval_path.read_text(encoding="utf-8"))
    result = {}
    for entry in data:
        vid = entry["video_id"]
        for model in ("qwen", "gemini"):
            key = f"{model}_scores"   # run_cross_evaluation 的實際欄位名
            if key not in entry:
                continue
            judges_data = {}
            for judge_name, judge_result in entry[key].items():
                judges_data[judge_name] = {
                    "q_norm": judge_result.get("q_norm", {}),
                    "overall": judge_result.get("overall", 0),
                }
            result[(vid, model)] = judges_data
    return result


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    import statistics
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = (sum((a - mx) ** 2 for a in xs)) ** 0.5
    dy = (sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / (dx * dy) if dx and dy else None


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman 相關：rank 後做 Pearson。"""
    if len(xs) < 2:
        return None
    def rankdata(vals):
        sorted_pairs = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0] * len(vals)
        for rank, (orig_idx, _) in enumerate(sorted_pairs):
            ranks[orig_idx] = rank + 1
        return ranks
    return pearson(rankdata(xs), rankdata(ys))


def icc_2_1(ratings: List[List[float]]) -> Optional[float]:
    """
    ICC(2,1) — two-way random effects, single rater, absolute agreement.
    ratings: n_items x k_raters matrix.
    公式參考：Shrout & Fleiss 1979.
    """
    if not ratings or len(ratings) < 2:
        return None
    n = len(ratings)
    k = len(ratings[0])
    if k < 2:
        return None

    grand = sum(sum(row) for row in ratings) / (n * k)
    row_means = [sum(row) / k for row in ratings]
    col_means = [sum(ratings[i][j] for i in range(n)) / n for j in range(k)]

    MS_row = k * sum((rm - grand) ** 2 for rm in row_means) / (n - 1)
    MS_col = n * sum((cm - grand) ** 2 for cm in col_means) / (k - 1) if k > 1 else 0
    MS_error = 0
    for i in range(n):
        for j in range(k):
            MS_error += (ratings[i][j] - row_means[i] - col_means[j] + grand) ** 2
    MS_error /= (n - 1) * (k - 1) if (n - 1) * (k - 1) else 1

    denom = MS_row + (k - 1) * MS_error + k * (MS_col - MS_error) / n
    if denom == 0:
        return None
    return (MS_row - MS_error) / denom


def interpret_icc(icc: float) -> str:
    """Koo & Li 2016 standard."""
    if icc is None:
        return "N/A"
    if icc < 0.5:  return "poor (< 0.5)"
    if icc < 0.75: return "moderate (0.5–0.75)"
    if icc < 0.9:  return "good (0.75–0.9)"
    return "excellent (≥ 0.9)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True, help="human_scores.csv")
    parser.add_argument("--cross-eval", required=True, help="per_report.json from run_cross_evaluation")
    parser.add_argument("--output", required=True, help="validation.json output path")
    args = parser.parse_args()

    human = load_human_scores(Path(args.human))
    llm = load_llm_scores(Path(args.cross_eval))
    logger.info(f"Loaded {len(human)} human entries, {len(llm)} LLM entries")

    # Intersect: keys where both human + LLM have all 7 q's scored
    common = []
    for key in sorted(set(human) & set(llm)):
        h = human[key]
        if len(h) < len(RUBRIC_QUESTIONS):
            continue
        common.append(key)
    logger.info(f"  {len(common)} samples with full human + LLM coverage")

    # Compute overall from human (mean of 7 q_norm)
    human_overall = {k: sum(human[k].values()) / len(human[k]) for k in common}

    # For each judge, compute correlation between human_overall and judge's overall
    report = {
        "n_samples": len(common),
        "by_judge": {},
    }

    # Collect all judges present
    judges_set = set()
    for key in common:
        judges_set |= set(llm[key].keys())

    for judge in sorted(judges_set):
        h_vals, l_vals = [], []
        for key in common:
            if judge not in llm[key]:
                continue
            h_vals.append(human_overall[key])
            l_vals.append(llm[key][judge]["overall"])
        if len(h_vals) < 2:
            continue

        # ICC(2,1) between human and this judge (2-rater)
        ratings = [[h, l] for h, l in zip(h_vals, l_vals)]
        icc = icc_2_1(ratings)
        p = pearson(h_vals, l_vals)
        s = spearman(h_vals, l_vals)

        report["by_judge"][judge] = {
            "n": len(h_vals),
            "icc_2_1": round(icc, 4) if icc is not None else None,
            "icc_interpretation": interpret_icc(icc),
            "pearson": round(p, 4) if p is not None else None,
            "spearman": round(s, 4) if s is not None else None,
        }

    # Per-question correlation (Q-by-Q human vs judge)
    report["by_question"] = {}
    for q_key in RUBRIC_QUESTIONS:
        per_judge = {}
        for judge in sorted(judges_set):
            h_vals, l_vals = [], []
            for key in common:
                if judge not in llm[key]:
                    continue
                if q_key not in human[key] or q_key not in llm[key][judge]["q_norm"]:
                    continue
                h_vals.append(human[key][q_key])
                l_vals.append(llm[key][judge]["q_norm"][q_key])
            if len(h_vals) >= 2:
                per_judge[judge] = {
                    "n": len(h_vals),
                    "pearson": round(pearson(h_vals, l_vals) or 0, 4),
                }
        report["by_question"][q_key] = per_judge

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Pretty print
    print("\n" + "=" * 72)
    print(" Human vs LLM Judge — ICC / Pearson / Spearman")
    print("=" * 72)
    print(f"\nn_samples (full coverage): {report['n_samples']}\n")
    print(f"{'Judge':<18} {'n':>4} {'ICC':>8} {'Pearson':>10} {'Spearman':>10}  Interpretation")
    print("-" * 72)
    for judge, data in report["by_judge"].items():
        print(
            f"{judge:<18} {data['n']:>4} "
            f"{data['icc_2_1'] if data['icc_2_1'] is not None else '—':>8} "
            f"{data['pearson'] if data['pearson'] is not None else '—':>10} "
            f"{data['spearman'] if data['spearman'] is not None else '—':>10}  "
            f"{data['icc_interpretation']}"
        )

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
