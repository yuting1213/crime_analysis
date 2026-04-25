"""
Cross-Evaluation — 對稱 LLM-as-a-Judge

設計：兩組報告（Qwen / Gemini）都由同一對「第三方裁判」評分，取平均。
預設裁判：Claude + OpenAI（都是第三方，避免 self-enhancement bias）。
可用 CLI / ENV 換成其他 judge 組合。

使用 7 題 Rubric（含 Q7 情景事實吻合度，UCA 當 GT），
每題正規化到 0–1 後平均得 overall（0–1）。

Open-book γ：裁判 prompt 同時附
  1. UCA 情景描述（事實 GT）
  2. 報告實際引用的法條全文 + 構成要件

用法：
    # 預設（Claude + OpenAI）
    python -m scripts.run_cross_evaluation \\
        --qwen-reports outputs/pilot_v5/pilot_reports \\
        --gemini-reports outputs/experiments/gemini_baseline/pilot_reports \\
        --output-dir outputs/cross_eval/pilot_v5

    # 自訂裁判（逗號分隔，格式任意）
    python -m scripts.run_cross_evaluation ... \\
        --judges claude-sonnet-4-20250514,gpt-4o

    # 只用一個裁判（smoke test）
    python -m scripts.run_cross_evaluation ... --judges gemini-2.5-flash --n-samples 3
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from env_loader import load_dotenv
    load_dotenv()
except ImportError:
    pass

from evaluation.llm_judge import LLMJudge, RUBRIC_QUESTIONS
from rag.rag_module import LEGAL_ELEMENTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Judge defaults ──────────────────────────────────────────
# 預設用兩個第三方 judge 平均；兩者都跟 Qwen/Gemini 無利害關係，避 self-bias。
import os as _os
DEFAULT_JUDGES = [
    _os.environ.get("JUDGE_A", "claude-sonnet-4-20250514"),
    _os.environ.get("JUDGE_B", "gpt-4o"),
]


# ── Legal article DB ────────────────────────────────────────

def _load_legal_db() -> Dict[str, Dict]:
    path = Path("data/rag/laws/criminal_code.json")
    if not path.exists():
        logger.warning(f"criminal_code.json not found at {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["article_id"]: entry for entry in data}


def _load_uca_scenarios(split: str = "Test") -> Dict[str, str]:
    from scripts.pilot_experiment import load_uca_annotations
    annots = load_uca_annotations(split)
    return {
        vid: " ".join(ann.get("sentences", []))
        for vid, ann in annots.items()
    }


# ── Report loaders ──────────────────────────────────────────

REPORT_SECTION_RE = re.compile(
    r"一、鑑定報告[^\n]*\n"
    r"─+\n\n"
    r"(.+?)"
    r"(?=─+\n\s*一-b|=+$|\Z)",
    re.DOTALL,
)
PREDICTED_RE = re.compile(r"模型判定：(\w+)")
GT_RE = re.compile(r"正確答案：(\w+)")


def load_report(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m_pred = PREDICTED_RE.search(text)
    m_gt = GT_RE.search(text)
    m_body = REPORT_SECTION_RE.search(text)
    if not m_body:
        return None
    return {
        "video_id": path.stem,
        "ground_truth": m_gt.group(1) if m_gt else "Unknown",
        "predicted": m_pred.group(1) if m_pred else "Unknown",
        "report_text": m_body.group(1).strip(),
    }


def load_reports_dir(reports_dir: Path) -> List[Dict]:
    if not reports_dir.exists():
        logger.error(f"Reports dir 不存在：{reports_dir}")
        return []
    out = []
    for fpath in sorted(reports_dir.glob("*.txt")):
        r = load_report(fpath)
        if r:
            out.append(r)
    logger.info(f"從 {reports_dir} 載入 {len(out)} 份報告")
    return out


# ── Open-book context ──────────────────────────────────────

# Arabic 條號：第 123 條 / 第 185-1 條 / 第 277-A 條 / 第 320-SH 條
# 後綴允許數字或字母（字母通常是修法版本代號如 A/F/SH/ST）
CITED_ARTICLE_ARABIC_RE = re.compile(
    r"第\s*(\d+(?:\s*-\s*[A-Za-z\d]+)?)\s*條"
)
# 中文數字條號：第六條 / 第二十五條 / 第一百八十五條
CITED_ARTICLE_CN_RE = re.compile(
    r"第\s*([〇零一二三四五六七八九十百千]+)\s*條"
)

_CN_DIGIT = {
    "〇": 0, "零": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}


def _cn_to_arabic(s: str) -> Optional[int]:
    """中文數字 → int（支援至千；回傳 None 代表無法轉）."""
    if not s:
        return None
    s = s.strip()
    # 單字 (一 ~ 九)
    if s in _CN_DIGIT:
        return _CN_DIGIT[s]
    if s == "十":
        return 10
    # 千
    if "千" in s:
        parts = s.split("千", 1)
        hd = _CN_DIGIT.get(parts[0], 1) if parts[0] else 1
        rest = _cn_to_arabic(parts[1]) if parts[1] else 0
        return hd * 1000 + (rest or 0) if rest is not None else None
    # 百
    if "百" in s:
        parts = s.split("百", 1)
        hd = _CN_DIGIT.get(parts[0], 1) if parts[0] else 1
        if not parts[1]:
            return hd * 100
        # 一百〇五 / 一百零五 形式
        rest = parts[1].lstrip("〇零")
        rest_val = _cn_to_arabic(rest) if rest else 0
        return hd * 100 + (rest_val or 0) if rest_val is not None else None
    # [N]十[M]
    if "十" in s:
        parts = s.split("十", 1)
        tens = _CN_DIGIT.get(parts[0], 1) if parts[0] else 1
        ones = _CN_DIGIT.get(parts[1], 0) if parts[1] else 0
        return tens * 10 + ones
    return None


def build_open_book_context(
    video_id: str,
    report_text: str,
    predicted_category: str,
    legal_db: Dict,
    scenarios: Dict[str, str],
) -> Dict:
    scenario = scenarios.get(video_id, "")

    # 1. Arabic 條號（含 185-3、277-A 等帶後綴）
    cited_ids = set()
    for m in CITED_ARTICLE_ARABIC_RE.finditer(report_text):
        # 正規化：去空白、連字號前後
        raw = re.sub(r"\s+", "", m.group(1))
        cited_ids.add(raw)

    # 2. 中文數字條號 → 轉 Arabic 再查表
    for m in CITED_ARTICLE_CN_RE.finditer(report_text):
        cn = m.group(1)
        n = _cn_to_arabic(cn)
        if n is not None:
            cited_ids.add(str(n))

    articles = []
    for article_id in cited_ids:
        entry = legal_db.get(article_id)
        if not entry:
            continue
        # 用 crime_category 查要件；兼顧 non-刑法（e.g. 家暴法）標籤
        cat = entry.get("crime_category", predicted_category)
        elements = LEGAL_ELEMENTS.get(cat, [])
        law_name = entry.get("law_name", "刑法")
        articles.append({
            "article": f"{law_name}第 {article_id} 條（{entry.get('chapter', '')}）",
            "full_text": entry.get("content", ""),
            "elements": elements,
        })
    return {"scenario_description": scenario, "cited_articles": articles}


# ── Cross-evaluation core ───────────────────────────────────

def _short_name(judge_model: str) -> str:
    """Short key for judge_model string (e.g. 'claude-sonnet-4-...' → 'claude')."""
    m = judge_model.lower()
    if "claude" in m: return "claude"
    if "gemini" in m: return "gemini"
    if "gpt" in m or "openai" in m: return "openai"
    return m.split("-")[0]


def _unique_judge_keys(judge_models: List[str]) -> List[str]:
    """
    產出獨一的 judge key。若家族短名（claude/gemini/openai）有衝突，
    退而取 `family_nextToken`（e.g. claude_sonnet, claude_haiku）。
    """
    shorts = [_short_name(m) for m in judge_models]
    if len(set(shorts)) == len(shorts):
        return shorts
    result = []
    for model in judge_models:
        parts = model.lower().split("-")
        # 取 family + 下一個有意義的 token
        key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        # 若還是衝突，追加 index
        if key in result:
            key = f"{key}_{len(result)}"
        result.append(key)
    return result


def evaluate_report(
    judge: LLMJudge,
    report: Dict,
    context: Dict,
) -> Dict:
    case_prompt = f"{report['ground_truth']}（模型判定：{report['predicted']}）"
    result = judge.rubric_score(
        prompt=case_prompt,
        report=report["report_text"],
        open_book_context=context,
        double_check=False,
    )
    return {
        "judge_model": judge.judge_model,
        "q_scores": result["q_scores"],
        "q_norm": result["q_norm"],
        "q_max": result["q_max"],
        "overall": result["overall"],
        "feedback": result["feedback"],
    }


def run_cross_eval(
    model_reports: Dict[str, List[Dict]],    # {"qwen": [...], "gemini": [...]}
    judges: List[LLMJudge],
    legal_db: Dict,
    scenarios: Dict[str, str],
    out_dir: Path,
    n_samples: Optional[int] = None,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按 video_id 配對
    by_vid: Dict[str, Dict[str, Dict]] = {}
    for model_key, reports in model_reports.items():
        for r in reports:
            by_vid.setdefault(r["video_id"], {})[model_key] = r
    common_ids = sorted(
        vid for vid, d in by_vid.items()
        if all(model_key in d for model_key in model_reports)
    )
    logger.info(f"Common video_ids across all models: {len(common_ids)}")

    if n_samples:
        common_ids = common_ids[:n_samples]
        logger.info(f"  limiting to first {n_samples} for smoke test")

    judge_keys = _unique_judge_keys([j.judge_model for j in judges])
    per_report = []
    for i, vid in enumerate(common_ids, 1):
        logger.info(f"[{i}/{len(common_ids)}] {vid}")
        entry = {"video_id": vid}

        for model_key, r in by_vid[vid].items():
            ctx = build_open_book_context(
                vid, r["report_text"], r["predicted"], legal_db, scenarios,
            )
            entry[f"{model_key}_gt"] = r["ground_truth"]
            entry[f"{model_key}_predicted"] = r["predicted"]
            entry[f"{model_key}_scores"] = {
                jkey: evaluate_report(j, r, ctx)
                for jkey, j in zip(judge_keys, judges)
            }

        per_report.append(entry)
        # 漸進儲存
        (out_dir / "per_report.json").write_text(
            json.dumps(per_report, ensure_ascii=False, indent=2)
        )

    return {"per_report": per_report, "n_common": len(common_ids), "judge_keys": judge_keys}


# ── Aggregation ─────────────────────────────────────────────

def summarize(per_report: List[Dict], judge_keys: List[str], models: List[str], out_dir: Path):
    import statistics

    def mean_std(xs):
        if not xs:
            return {"mean": 0, "std": 0, "n": 0}
        return {
            "mean": round(statistics.mean(xs), 4),
            "std":  round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs),
        }

    def pearson(xs, ys):
        if len(xs) < 2:
            return None
        mx = statistics.mean(xs); my = statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = (sum((a - mx) ** 2 for a in xs)) ** 0.5
        dy = (sum((b - my) ** 2 for b in ys)) ** 0.5
        return round(num / (dx * dy), 4) if dx and dy else None

    # Per-model × per-judge overall
    per_model_summary = {}
    for m in models:
        judge_scores = {j: [] for j in judge_keys}
        averaged = []
        for r in per_report:
            scores = r.get(f"{m}_scores")
            if not scores:
                continue
            for j in judge_keys:
                if j in scores:
                    judge_scores[j].append(scores[j]["overall"])
            # 取所有裁判平均當 overall
            judge_vals = [scores[j]["overall"] for j in judge_keys if j in scores]
            if judge_vals:
                averaged.append(sum(judge_vals) / len(judge_vals))
        per_model_summary[m] = {
            "by_judge": {j: mean_std(xs) for j, xs in judge_scores.items()},
            "averaged": mean_std(averaged),
        }

    # Cross-judge agreement: 每個 model 內，兩裁判間 Pearson
    agreement = {}
    for m in models:
        if len(judge_keys) < 2:
            continue
        xs_by_j = {j: [] for j in judge_keys}
        for r in per_report:
            scores = r.get(f"{m}_scores")
            if not scores:
                continue
            vals = {j: scores[j]["overall"] for j in judge_keys if j in scores}
            if len(vals) == len(judge_keys):
                for j, v in vals.items():
                    xs_by_j[j].append(v)
        # Compute all pairs
        for i in range(len(judge_keys)):
            for k in range(i + 1, len(judge_keys)):
                j1, j2 = judge_keys[i], judge_keys[k]
                r_val = pearson(xs_by_j[j1], xs_by_j[j2])
                agreement[f"{m}__{j1}_vs_{j2}"] = r_val

    # Cross-model comparison — 看 averaged overall
    # 注意：若某 judge 在某 entry 缺席（e.g. API 失敗），分母要用實際 judge 數，
    # 否則分數會被稀釋。以 pair-wise 比較為準，兩個 model 都至少要有 1 個 judge 分數。
    cross_model = {}
    if len(models) == 2:
        m1, m2 = models
        xs_m1, xs_m2 = [], []
        for r in per_report:
            if f"{m1}_scores" not in r or f"{m2}_scores" not in r:
                continue
            judges_m1 = [j for j in judge_keys if j in r[f"{m1}_scores"]]
            judges_m2 = [j for j in judge_keys if j in r[f"{m2}_scores"]]
            if not judges_m1 or not judges_m2:
                continue
            v1 = sum(r[f"{m1}_scores"][j]["overall"] for j in judges_m1) / len(judges_m1)
            v2 = sum(r[f"{m2}_scores"][j]["overall"] for j in judges_m2) / len(judges_m2)
            xs_m1.append(v1); xs_m2.append(v2)
        cross_model = {
            f"{m1}_mean": mean_std(xs_m1),
            f"{m2}_mean": mean_std(xs_m2),
            f"{m1}_minus_{m2}": (
                round(statistics.mean(xs_m1) - statistics.mean(xs_m2), 4)
                if xs_m1 and xs_m2 else None
            ),
        }

    summary = {
        "n_reports": len(per_report),
        "judges": judge_keys,
        "per_model": per_model_summary,
        "cross_judge_agreement_pearson": agreement,
        "cross_model_comparison": cross_model,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Human-readable
    lines = [
        "=" * 76,
        " Cross-Evaluation Summary (symmetric 2-judge)",
        "=" * 76,
        f"\nn_reports: {summary['n_reports']}",
        f"judges:    {judge_keys}",
        "\n── Per-model overall (0–1 scale) ──",
    ]
    for m, d in per_model_summary.items():
        lines.append(f"\n  {m}:")
        for j, s in d["by_judge"].items():
            lines.append(f"    {j:<10} mean={s['mean']:.3f}  std={s['std']:.3f}  n={s['n']}")
        a = d["averaged"]
        lines.append(f"    {'averaged':<10} mean={a['mean']:.3f}  std={a['std']:.3f}  n={a['n']}")

    if agreement:
        lines.append("\n── Cross-judge agreement (Pearson r) ──")
        for k, v in agreement.items():
            lines.append(f"  {k}: {v if v is not None else 'N/A'}")

    if cross_model:
        lines.append("\n── Cross-model comparison (averaged over judges) ──")
        for k, v in cross_model.items():
            if isinstance(v, dict):
                lines.append(f"  {k:<24} mean={v['mean']:.3f}  n={v['n']}")
            else:
                lines.append(f"  {k:<24} = {v}")

    (out_dir / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-reports", required=True)
    parser.add_argument("--gemini-reports",
                        default="outputs/experiments/gemini_baseline/pilot_reports")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--split", default="Test")
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--judges", default=None,
                        help="逗號分隔的 judge model 名稱；預設取 $JUDGE_A, $JUDGE_B (fallback claude+openai)")
    args = parser.parse_args()

    judge_models = (
        [x.strip() for x in args.judges.split(",")]
        if args.judges else DEFAULT_JUDGES
    )
    logger.info(f"Judges: {judge_models}")

    qwen_reports = load_reports_dir(Path(args.qwen_reports))
    gemini_reports = load_reports_dir(Path(args.gemini_reports))
    legal_db = _load_legal_db()
    scenarios = _load_uca_scenarios(args.split)

    if not qwen_reports or not gemini_reports:
        logger.error("至少一組報告為空，abort")
        sys.exit(1)

    judges = [
        LLMJudge(judge_model=m, budget_limit_usd=args.budget)
        for m in judge_models
    ]

    out_dir = Path(args.output_dir)
    model_reports = {"qwen": qwen_reports, "gemini": gemini_reports}
    results = run_cross_eval(
        model_reports, judges, legal_db, scenarios,
        out_dir, n_samples=args.n_samples,
    )

    summarize(
        results["per_report"],
        results["judge_keys"],
        list(model_reports.keys()),
        out_dir,
    )

    logger.info(f"\n✅ Done. Outputs → {out_dir}/")
    unique_keys = _unique_judge_keys([j.judge_model for j in judges])
    for key, j in zip(unique_keys, judges):
        s = j.token_summary
        logger.info(f"  [{key}] {s['total_calls']} calls, ${s['total_cost_usd']:.3f}")


if __name__ == "__main__":
    main()
