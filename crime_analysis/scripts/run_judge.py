"""
LLM-as-Judge 批次評分腳本

用法：
    cd crime_analysis
    # 推薦：把 ANTHROPIC_API_KEY（或其他 judge 所需 key）寫進 crime_analysis/.env

    # 評分單一實驗的報告
    python -m scripts.run_judge --reports_dir outputs/pilot_reports

    # 比較兩組報告（Pairwise）
    python -m scripts.run_judge --pairwise \
        --reports_a outputs/pilot_reports \
        --reports_b outputs/experiments/gemini_baseline/pilot_reports

    # 指定 Judge 模型
    python -m scripts.run_judge --reports_dir outputs/pilot_reports --judge claude-sonnet-4-20250514
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 自動從 .env 載入 API keys；shell exports 仍優先
try:
    from env_loader import load_dotenv
    load_dotenv()
except ImportError:
    pass

from evaluation.llm_judge import LLMJudge, RUBRIC_DIMENSIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_reports(reports_dir: Path) -> List[Dict]:
    """載入目錄下所有 .txt 報告。"""
    reports = []
    for f in sorted(reports_dir.glob("*.txt")):
        content = f.read_text(encoding="utf-8")
        video_id = f.stem

        # 解析 ground truth 和 predicted
        gt = ""
        pred = ""
        for line in content.split("\n"):
            if "正確答案：" in line:
                gt = line.split("正確答案：")[1].strip()
            elif "模型判定：" in line:
                pred = line.split("模型判定：")[1].split()[0].strip()

        # 提取鑑定報告內容（第一個 section）
        report_text = ""
        in_report = False
        for line in content.split("\n"):
            if "一、鑑定報告" in line:
                in_report = True
                continue
            elif "一-b、" in line or "二、行為分析" in line:
                in_report = False
            elif in_report:
                report_text += line + "\n"

        reports.append({
            "video_id": video_id,
            "ground_truth": gt,
            "predicted": pred,
            "report_text": report_text.strip(),
            "file_path": str(f),
        })
    return reports


def run_rubric(judge: LLMJudge, reports: List[Dict], output_dir: Path):
    """Rubric 評分所有報告。"""
    results = []
    for i, r in enumerate(reports):
        if not r["report_text"]:
            logger.warning(f"  [{i+1}/{len(reports)}] {r['video_id']} 無報告內容，跳過")
            continue

        logger.info(f"  [{i+1}/{len(reports)}] {r['video_id']} ({r['ground_truth']})")
        prompt = f"犯罪類別：{r['predicted']}，正確答案：{r['ground_truth']}，影片 ID：{r['video_id']}"

        try:
            scores = judge.rubric_score(prompt, r["report_text"])
            scores["video_id"] = r["video_id"]
            scores["ground_truth"] = r["ground_truth"]
            scores["predicted"] = r["predicted"]
            results.append(scores)

            dims = scores["dimension_scores"]
            logger.info(
                f"    LC={dims.get('logical_consistency', 0)} "
                f"LEG={dims.get('legal_coverage', 0)} "
                f"EV={dims.get('evidence_citation', 0)} "
                f"CR={dims.get('causal_reasoning', 0)} "
                f"UM={dims.get('uncertainty_marking', 0)} "
                f"| overall={scores['overall_score']:.1f}"
            )
        except Exception as e:
            logger.error(f"    評分失敗：{e}")

        time.sleep(1)  # Rate limiting

    # 儲存結果
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "judge_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 摘要
    if results:
        print_summary(results, output_dir)

    return results


def run_pairwise(judge: LLMJudge, reports_a: List[Dict], reports_b: List[Dict], output_dir: Path):
    """Pairwise 比較兩組報告。"""
    # 以 video_id 配對
    b_map = {r["video_id"]: r for r in reports_b}
    results = []

    for i, a in enumerate(reports_a):
        b = b_map.get(a["video_id"])
        if not b or not a["report_text"] or not b["report_text"]:
            continue

        logger.info(f"  [{i+1}/{len(reports_a)}] {a['video_id']} A vs B")
        prompt = f"犯罪類別：{a['ground_truth']}，影片 ID：{a['video_id']}"

        try:
            result = judge.pairwise_compare(
                prompt, a["report_text"], b["report_text"],
                video_id=a["video_id"],
                double_check=True,
            )
            results.append(result)
            logger.info(f"    Winner={result['winner']} A={result['score_a']:.1f} B={result['score_b']:.1f}")
        except Exception as e:
            logger.error(f"    比較失敗：{e}")

        time.sleep(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pairwise_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 摘要
    if results:
        a_wins = sum(1 for r in results if r["winner"] == "A")
        b_wins = sum(1 for r in results if r["winner"] == "B")
        ties = sum(1 for r in results if r["winner"] == "tie")
        consistent = sum(1 for r in results if r.get("is_consistent", True))
        print(f"\nPairwise: A wins={a_wins}, B wins={b_wins}, tie={ties}")
        print(f"Position Bias 一致率: {consistent}/{len(results)} ({100*consistent/len(results):.0f}%)")

    return results


def print_summary(results: List[Dict], output_dir: Path):
    """印出 + 儲存評分摘要。"""
    n = len(results)
    print(f"\n{'='*60}")
    print(f"  LLM-as-Judge Rubric Summary ({n} reports)")
    print(f"{'='*60}")

    for dim in RUBRIC_DIMENSIONS:
        vals = [r["dimension_scores"].get(dim, 0) for r in results]
        avg = sum(vals) / len(vals)
        print(f"  {dim:25s}: {avg:.2f} / 5.0")

    overalls = [r["overall_score"] for r in results]
    avg_overall = sum(overalls) / len(overalls)
    print(f"  {'overall':25s}: {avg_overall:.2f} / 5.0")
    print(f"  {'total (×5)':25s}: {avg_overall * 5:.1f} / 25.0")

    stable = sum(1 for r in results if r.get("is_stable", True))
    print(f"  穩定率: {stable}/{n} ({100*stable/n:.0f}%)")

    # 存摘要
    summary = {
        "n_reports": n,
        "per_dimension": {
            dim: round(sum(r["dimension_scores"].get(dim, 0) for r in results) / n, 3)
            for dim in RUBRIC_DIMENSIONS
        },
        "overall": round(avg_overall, 3),
        "total_25": round(avg_overall * 5, 1),
        "stability_rate": round(stable / n, 3),
    }
    with open(output_dir / "judge_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"摘要 → {output_dir / 'judge_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge 批次評分")
    parser.add_argument("--reports_dir", type=str, help="報告目錄（Rubric 評分）")
    parser.add_argument("--pairwise", action="store_true", help="Pairwise 比較模式")
    parser.add_argument("--reports_a", type=str, help="Pairwise A 組報告目錄")
    parser.add_argument("--reports_b", type=str, help="Pairwise B 組報告目錄")
    parser.add_argument("--judge", type=str, default=None, help="Judge 模型名稱")
    parser.add_argument("--budget", type=float, default=20.0, help="預算上限 USD（預設 $20）")
    parser.add_argument("--output_dir", type=str, default="./outputs/judge_results")
    args = parser.parse_args()

    judge = LLMJudge(judge_model=args.judge, budget_limit_usd=args.budget)
    logger.info(f"Judge: {judge.judge_model} | Budget: ${args.budget}")

    if args.pairwise:
        if not args.reports_a or not args.reports_b:
            logger.error("Pairwise 模式需要 --reports_a 和 --reports_b")
            sys.exit(1)
        reports_a = load_reports(Path(args.reports_a))
        reports_b = load_reports(Path(args.reports_b))
        logger.info(f"Pairwise: A={len(reports_a)} reports, B={len(reports_b)} reports")
        run_pairwise(judge, reports_a, reports_b, Path(args.output_dir))
    else:
        if not args.reports_dir:
            logger.error("需要 --reports_dir")
            sys.exit(1)
        reports = load_reports(Path(args.reports_dir))
        logger.info(f"Rubric: {len(reports)} reports from {args.reports_dir}")
        run_rubric(judge, reports, Path(args.output_dir))

    # Token 使用摘要
    summary = judge.token_summary
    print(f"\n{'='*60}")
    print(f"  Token Usage Summary")
    print(f"{'='*60}")
    print(f"  API calls:      {summary['total_calls']}")
    print(f"  Input tokens:   {summary['total_input_tokens']:,}")
    print(f"  Output tokens:  {summary['total_output_tokens']:,}")
    print(f"  Total cost:     ${summary['total_cost_usd']:.4f}")
    print(f"  Budget remaining: ${summary['budget_remaining_usd']:.4f} / ${summary['budget_limit_usd']}")

    # 存 token 使用記錄
    with open(Path(args.output_dir) / "token_usage.json", "w") as f:
        json.dump(summary, f, indent=2)
