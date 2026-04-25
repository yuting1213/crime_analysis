"""
sample_for_human_eval.py — 從 cross_eval 結果抽取人工評分樣本

輸出：
  human_sample/sample_list.csv       — 受試者要評的 40 份報告 (20 部 × 2 model)
  human_sample/questionnaire.html    — 自含 rubric + open-book γ 參考資料的 HTML 問卷
  human_sample/human_scores_blank.csv — 空白評分表（人工填完後交給 compute_human_llm_correlation.py）

用法：
    cd crime_analysis
    python -m scripts.sample_for_human_eval \\
        --cross-eval-dir outputs/cross_eval/pilot_v5 \\
        --qwen-reports outputs/pilot_v5/pilot_reports \\
        --gemini-reports outputs/experiments/gemini_baseline/pilot_reports \\
        --per-category 2 \\
        --seed 42
"""
import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.llm_judge import RUBRIC_QUESTIONS
from scripts.run_cross_evaluation import (
    load_reports_dir, build_open_book_context,
    _load_legal_db, _load_uca_scenarios,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# UCF 10 crime 類別（排除 non-crime 與 Normal）
CRIME_CATEGORIES = [
    "Abuse", "Arson", "Assault", "Burglary", "Fighting",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def stratified_sample(
    qwen_reports: List[Dict],
    gemini_reports: List[Dict],
    per_category: int,
    seed: int,
) -> List[str]:
    """每類別抽 per_category 部影片，確保兩個 model 都有該 video_id 的報告。"""
    random.seed(seed)
    by_cat_qwen = defaultdict(list)
    by_cat_gemini = defaultdict(list)
    for r in qwen_reports:
        by_cat_qwen[r["ground_truth"]].append(r["video_id"])
    for r in gemini_reports:
        by_cat_gemini[r["ground_truth"]].append(r["video_id"])

    selected = []
    for cat in CRIME_CATEGORIES:
        common = sorted(set(by_cat_qwen.get(cat, [])) & set(by_cat_gemini.get(cat, [])))
        if not common:
            logger.warning(f"  {cat}: no common video_ids between Qwen & Gemini → skip")
            continue
        picks = random.sample(common, min(per_category, len(common)))
        selected.extend(picks)
        logger.info(f"  {cat}: picked {len(picks)} / {len(common)} common")
    return selected


def render_questionnaire(
    samples: List[Dict],
    out_path: Path,
):
    """Render self-contained HTML questionnaire. Submit = download CSV."""
    # Build question HTML block
    q_blocks = []
    for q_key, q in RUBRIC_QUESTIONS.items():
        max_score = q["max_score"] if q["max_score"] is not None else 4
        options = "".join(
            f'<option value="{i}">{i}</option>'
            for i in range(max_score + 1)
        )
        q_blocks.append(f"""
          <div class="q">
            <label><b>{q_key}</b> (0–{max_score}) — {q['description']}</label>
            <select name="{{prefix}}_{q_key}" required>
              <option value="">--</option>{options}
            </select>
          </div>
        """)
    q_html = "".join(q_blocks)

    # Per-sample cards
    cards = []
    for idx, s in enumerate(samples):
        scenario = (s.get("scenario") or "")[:500]
        ref_arts = s.get("cited_articles", [])
        ref_html = ""
        if ref_arts:
            items = [f"<li><b>{a['article']}</b><br><pre>{a['full_text'][:400]}</pre>"
                     f"<br>要件：{'、'.join(a['elements']) if a['elements'] else '（未知）'}</li>"
                     for a in ref_arts]
            ref_html = f"<p><b>報告引用的法條</b></p><ul>{''.join(items)}</ul>"

        card = f"""
        <section>
          <h3>#{idx+1} — {s['video_id']}  （{s['model']}）</h3>
          <p><b>GT 類別：</b>{s['ground_truth']}  <b>模型判定：</b>{s['predicted']}</p>
          <details open>
            <summary><b>【參考資料】</b>（點擊摺疊）</summary>
            <p><b>UCA 情景描述：</b><br>{scenario}</p>
            {ref_html}
          </details>
          <details>
            <summary><b>【待評報告】</b>（點擊展開）</summary>
            <pre style="white-space:pre-wrap;font-family:monospace;">{s['report_text']}</pre>
          </details>
          <div class="questions">{q_html.replace('{prefix}', f'sample_{idx}')}</div>
        </section>
        """
        cards.append(card)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>LLM-as-a-Judge 人工評分問卷</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 2em auto; padding: 1em; }}
  section {{ border: 1px solid #ccc; padding: 1em; margin: 1em 0; border-radius: 6px; }}
  pre {{ background: #f5f5f5; padding: 0.5em; overflow-x: auto; }}
  .q {{ margin: 0.5em 0; }}
  details summary {{ cursor: pointer; margin: 0.5em 0; }}
  button {{ font-size: 1.1em; padding: 0.5em 2em; }}
</style>
</head>
<body>
<h1>鑑定報告人工評分（7 題 Rubric）</h1>
<p>請依下方 rubric 判準評分：<code>raw 分數 0 ~ 該題最高分</code>。完成後按下方按鈕下載 CSV 繳交。</p>
<form id="evalForm">
{''.join(cards)}
<button type="button" onclick="downloadCsv()">下載評分 CSV</button>
</form>
<script>
function downloadCsv() {{
    const form = document.getElementById('evalForm');
    const data = new FormData(form);
    const rows = [['sample_idx','video_id','model','q_key','score']];
    const samples = {json.dumps([(s['video_id'], s['model']) for s in samples])};
    samples.forEach((tup, idx) => {{
        const [vid, model] = tup;
        const qKeys = {json.dumps(list(RUBRIC_QUESTIONS.keys()))};
        qKeys.forEach(q => {{
            const v = data.get(`sample_${{idx}}_${{q}}`);
            if (v !== null && v !== '') rows.push([idx, vid, model, q, v]);
        }});
    }});
    const csv = rows.map(r => r.join(',')).join('\\n');
    const blob = new Blob([csv], {{type: 'text/csv'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'human_scores.csv';
    a.click();
}}
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-reports", required=True)
    parser.add_argument("--gemini-reports",
                        default="outputs/experiments/gemini_baseline/pilot_reports")
    parser.add_argument("--cross-eval-dir", required=True,
                        help="Dir containing per_report.json (used to pull LLM scores for later correlation)")
    parser.add_argument("--per-category", type=int, default=2,
                        help="Videos per crime category (default 2 → 20 total; × 2 models = 40 reports)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="Default: <cross-eval-dir>/human_sample/")
    parser.add_argument("--split", default="Test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir or (Path(args.cross_eval_dir) / "human_sample"))
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_reports = load_reports_dir(Path(args.qwen_reports))
    gemini_reports = load_reports_dir(Path(args.gemini_reports))
    legal_db = _load_legal_db()
    scenarios = _load_uca_scenarios(args.split)

    video_ids = stratified_sample(
        qwen_reports, gemini_reports,
        per_category=args.per_category, seed=args.seed,
    )
    logger.info(f"Selected {len(video_ids)} video_ids → {len(video_ids)*2} reports")

    # Build sample entries (add open-book context for HTML rendering)
    by_vid_q = {r["video_id"]: r for r in qwen_reports}
    by_vid_g = {r["video_id"]: r for r in gemini_reports}

    samples = []
    for vid in video_ids:
        for model, r in (("qwen", by_vid_q[vid]), ("gemini", by_vid_g[vid])):
            ctx = build_open_book_context(
                vid, r["report_text"], r["predicted"], legal_db, scenarios,
            )
            samples.append({
                "video_id": vid,
                "model": model,
                "ground_truth": r["ground_truth"],
                "predicted": r["predicted"],
                "report_text": r["report_text"],
                "scenario": ctx["scenario_description"],
                "cited_articles": ctx["cited_articles"],
            })

    # sample_list.csv
    with (out_dir / "sample_list.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "video_id", "model", "ground_truth", "predicted"])
        for i, s in enumerate(samples):
            w.writerow([i, s["video_id"], s["model"], s["ground_truth"], s["predicted"]])

    # blank human_scores.csv (submit template)
    with (out_dir / "human_scores_blank.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "video_id", "model", "q_key", "score"])

    # questionnaire.html
    render_questionnaire(samples, out_dir / "questionnaire.html")

    logger.info(f"\n✅ Done. Files in {out_dir}:")
    logger.info(f"  sample_list.csv          - {len(samples)} rows")
    logger.info(f"  questionnaire.html       - self-contained 問卷（請人類評分者用瀏覽器打開填完下載 CSV）")
    logger.info(f"  human_scores_blank.csv   - template for submission")


if __name__ == "__main__":
    main()
