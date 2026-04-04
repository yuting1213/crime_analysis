"""
Pilot Experiment — 小樣本校準實驗

資料來源：
  UCF-Crime 影片 + UCA 時序文字標註

目的：
  1. 確認 Reflector 信心門檻（目前 HARD=0.75 / SOFT=0.70）
  2. 確認 Rcost 的 threshold_low / threshold_high（目前 3 / 8）
  3. 觀察 Rlegal 分布，驗證 LEGAL_ELEMENTS 的覆蓋率是否合理
  4. 驗證 UCA 標註是否能有效輔助分析

使用方式：
  cd crime_analysis
  python -m scripts.pilot_experiment --n_samples 30

產出：
  outputs/pilot_stats.json — 每個案例的詳細指標
  outputs/pilot_summary.txt — 統計摘要與建議門檻
"""
import argparse
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 路徑設定 ─────────────────────────────────────────────────
# 相對於 repo root（碩論/）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts → crime_analysis → 碩論
UCA_ROOT = REPO_ROOT / "UCA"
VIDEOS_DIR = UCA_ROOT / "UCF_Crimes" / "UCF_Crimes" / "Videos"

# 犯罪類別（不含 Normal）
CRIME_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


# ── 標註載入 ─────────────────────────────────────────────────

def load_uca_annotations(split: str = "Test") -> Dict:
    """
    載入 UCA 時序文字標註。

    Args:
        split: "Train" | "Test" | "Val"

    Returns:
        {
            "Abuse001_x264": {
                "duration": 91.0,
                "timestamps": [[0.0, 5.3], ...],
                "sentences": ["A woman ...", ...],
            },
            ...
        }
    """
    json_path = UCA_ROOT / f"UCFCrime_{split}.json"
    if not json_path.exists():
        logger.warning(f"UCA 標註不存在：{json_path}")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"UCA {split} 標註：{len(data)} 部影片")
    return data


def video_id_to_category(video_id: str) -> str:
    """
    從 video_id 提取犯罪類別。

    "Abuse001_x264" → "Abuse"
    "RoadAccidents045_x264" → "RoadAccidents"
    "Normal_Videos_003_x264" → "Normal"
    """
    if video_id.startswith("Normal"):
        return "Normal"
    match = re.match(r"([A-Za-z]+)\d+", video_id)
    return match.group(1) if match else "Normal"


# ── 樣本載入 ─────────────────────────────────────────────────

def load_pilot_samples(
    n_samples: int = 30,
    split: str = "Test",
    include_normal: bool = False,
) -> List[Dict]:
    """
    從 UCA 標註 + UCF-Crime 影片目錄載入 pilot 樣本。

    策略：每個犯罪類別各取 ceil(n_samples / 13) 個，
    確保 violent / property / public_safety 都有覆蓋。
    """
    annotations = load_uca_annotations(split)
    if not annotations:
        return []

    per_cat = max(1, -(-n_samples // len(CRIME_CATEGORIES)))
    category_counts: Dict[str, int] = {}
    samples = []

    for video_id, ann in annotations.items():
        cat = video_id_to_category(video_id)

        if cat == "Normal" and not include_normal:
            continue
        if cat not in CRIME_CATEGORIES and cat != "Normal":
            continue
        if category_counts.get(cat, 0) >= per_cat:
            continue

        # 找影片檔案
        video_path = VIDEOS_DIR / cat / f"{video_id}.mp4"
        if not video_path.exists():
            # 嘗試 Normal 的路徑
            for normal_dir in ["Training_Normal_Videos_Anomaly",
                               "Testing_Normal_Videos_Anomaly",
                               "z_Normal_Videos_event"]:
                alt = VIDEOS_DIR / normal_dir / f"{video_id}.mp4"
                if alt.exists():
                    video_path = alt
                    break
            if not video_path.exists():
                continue

        # 組裝 UCA 標註為 metadata
        fps = 25.0  # UCF-Crime 預設
        metadata = {
            "fps": fps,
            "video_id": video_id,
            "duration": ann["duration"],
            # 將 UCA 時序描述轉為結構化 metadata
            "uca_segments": [
                {
                    "start": ts[0],
                    "end": ts[1],
                    "description": sent,
                    "start_frame": int(ts[0] * fps),
                    "end_frame": int(ts[1] * fps),
                }
                for ts, sent in zip(ann["timestamps"], ann["sentences"])
            ],
        }

        samples.append({
            "video_path": str(video_path),
            "video_id": video_id,
            "ground_truth": cat,
            "metadata": metadata,
        })
        category_counts[cat] = category_counts.get(cat, 0) + 1

    samples = samples[:n_samples]
    logger.info(
        f"Pilot 樣本：{len(samples)} 個影片（目標 {n_samples}）\n"
        f"  類別分布：{category_counts}"
    )
    return samples


# ── 幀抽取 ───────────────────────────────────────────────────

def extract_frames(video_path: str, n_frames: int = 32) -> List:
    """從影片均勻抽取 n_frames 幀。"""
    try:
        import cv2
    except ImportError:
        logger.warning("cv2 未安裝，使用 placeholder 幀")
        return [None] * n_frames

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total <= 0:
        cap.release()
        return [None] * n_frames

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame if ret else None)
    cap.release()

    return frames


# ── 執行實驗 ─────────────────────────────────────────────────

def run_pilot(samples: List[Dict], output_dir: str) -> Dict:
    """對每個 pilot 樣本執行 pipeline.analyze()，收集統計數據。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from pipeline import CrimeAnalysisPipeline

    pipeline = CrimeAnalysisPipeline()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    case_stats = []

    for i, sample in enumerate(samples):
        video_id = sample["video_id"]
        gt = sample["ground_truth"]
        logger.info(f"[Pilot {i+1}/{len(samples)}] {video_id} ({gt})")

        frames = extract_frames(sample["video_path"])
        metadata = sample["metadata"]

        try:
            result = pipeline.analyze(frames, metadata)
        except Exception as e:
            logger.error(f"  分析失敗：{e}")
            continue

        # 收集 UCA 標註統計
        uca_segments = metadata.get("uca_segments", [])
        n_uca_sentences = len(uca_segments)

        stat = {
            "video_id": video_id,
            "ground_truth": gt,
            "predicted": result.get("final_category", "Normal"),
            "correct": result.get("final_category") == gt,
            "confidence": result.get("fact_finding", {}).get("confidence", 0.0),
            "rcons": result.get("rcons", 0.0),
            "rlegal": result.get("rlegal", 0.0),
            "rcost": result.get("rcost", 0.0),
            "total_turns": result.get("total_turns", 0),
            "conflict_type": result.get("conflict_type", "NONE"),
            "is_convergent": result.get("is_convergent", False),
            "duration": metadata.get("duration", 0.0),
            "n_uca_sentences": n_uca_sentences,
        }
        case_stats.append(stat)

        # ── 輸出完整鑑定報告 ────────────────────────────
        _save_case_report(result, video_id, gt, metadata, output_path)

    # ── 統計分析 ─────────────────────────────────────────
    if not case_stats:
        logger.warning("無有效結果")
        return {}

    return _compute_summary(case_stats, output_path)


def _save_case_report(
    result: Dict,
    video_id: str,
    ground_truth: str,
    metadata: Dict,
    output_dir: Path,
):
    """將單一案例的完整鑑定報告寫入 outputs/pilot_reports/{video_id}.txt"""
    report_dir = output_dir / "pilot_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{video_id}.txt"

    predicted = result.get("final_category", "Normal")
    correct = "✓" if predicted == ground_truth else "✗"
    confidence = result.get("fact_finding", {}).get("confidence", 0.0)
    report_text = result.get("fact_finding", {}).get("description", "（無報告）")
    rationale = result.get("fact_finding", {}).get("rationale", "")

    # 行為分析
    behavior = result.get("behavior_analysis", {})
    causal_chain = behavior.get("causal_chain", "")
    escalation = behavior.get("escalation_score", 0.0)
    pre_crime = behavior.get("pre_crime_indicators", [])
    post_crime = behavior.get("post_crime_indicators", [])

    # 法律分類
    legal = result.get("legal_classification", {})
    articles = legal.get("applicable_articles", [])
    elements = legal.get("elements_covered", [])
    rlegal = result.get("rlegal", 0.0)

    # 不確定性
    uncertainty = result.get("uncertainty_notes", {})

    # UCA 標註
    uca_segments = metadata.get("uca_segments", [])

    lines = [
        "=" * 70,
        f"  鑑定報告：{video_id}",
        "=" * 70,
        "",
        f"  正確答案：{ground_truth}",
        f"  模型判定：{predicted}  {correct}",
        f"  信心程度：{confidence:.2%}",
        f"  報告生成：{result.get('report_generation_method', 'unknown')}",
        "",
        "─" * 70,
        "  一、Qwen3-8B 鑑定報告",
        "─" * 70,
        "",
        report_text,
        "",
        "─" * 70,
        "  一-b、ActionEmotion Agent 行為摘要",
        "─" * 70,
        "",
        rationale or "（無）",
        "",
        "─" * 70,
        "  二、行為分析",
        "─" * 70,
        "",
        f"  因果鏈：{causal_chain}",
        f"  升溫分數：{escalation:.2f}",
        f"  事前指標：{'; '.join(pre_crime) if pre_crime else '（無）'}",
        f"  事後指標：{'; '.join(post_crime) if post_crime else '（無）'}",
        "",
        "─" * 70,
        "  三、法律適用",
        "─" * 70,
        "",
        f"  適用法條：{', '.join(articles) if articles else '（無）'}",
        f"  構成要件：{', '.join(elements) if elements else '（無）'}",
        f"  Rlegal：{rlegal:.3f}",
        "",
        "─" * 70,
        "  四、系統指標",
        "─" * 70,
        "",
        f"  Rcons：{result.get('rcons', 0.0):.3f}",
        f"  Rcost：{result.get('rcost', 0.0):.3f}",
        f"  對話輪次：{result.get('total_turns', 0)}",
        f"  衝突類型：{result.get('conflict_type', 'NONE')}",
        f"  是否收斂：{result.get('is_convergent', False)}",
        "",
        "─" * 70,
        "  五、不確定性與限制",
        "─" * 70,
        "",
    ]

    for key, label in [
        ("low_confidence_items", "低信心項目"),
        ("conflicting_evidence", "衝突證據"),
        ("insufficient_evidence", "證據不足"),
    ]:
        items = uncertainty.get(key, [])
        if items:
            lines.append(f"  【{label}】")
            for item in items:
                lines.append(f"    - {item}")
        else:
            lines.append(f"  【{label}】（無）")

    if uca_segments:
        lines.extend([
            "",
            "─" * 70,
            "  附錄：UCA 時序標註（Ground Truth）",
            "─" * 70,
            "",
        ])
        for seg in uca_segments:
            lines.append(
                f"  [{seg.get('start', 0):.1f}s – {seg.get('end', 0):.1f}s] "
                f"{seg.get('description', '')}"
            )

    lines.append("")
    lines.append("=" * 70)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"  報告 → {report_path}")


def _compute_summary(case_stats: List[Dict], output_path: Path) -> Dict:
    """計算統計摘要 + 門檻建議。"""

    turns_list = [s["total_turns"] for s in case_stats]
    conf_correct = [s["confidence"] for s in case_stats if s["correct"]]
    conf_wrong = [s["confidence"] for s in case_stats if not s["correct"]]
    rlegal_list = [s["rlegal"] for s in case_stats]
    rcons_list = [s["rcons"] for s in case_stats]

    # 按犯罪類別分組統計
    cat_stats: Dict[str, Dict] = {}
    for s in case_stats:
        cat = s["ground_truth"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0, "conf_list": []}
        cat_stats[cat]["total"] += 1
        if s["correct"]:
            cat_stats[cat]["correct"] += 1
        cat_stats[cat]["conf_list"].append(s["confidence"])

    per_cat_acc = {
        cat: d["correct"] / d["total"] if d["total"] > 0 else 0.0
        for cat, d in cat_stats.items()
    }

    summary = {
        "n_samples": len(case_stats),
        "accuracy": sum(1 for s in case_stats if s["correct"]) / len(case_stats),
        "per_category_accuracy": per_cat_acc,
        "turns": {
            "mean": statistics.mean(turns_list),
            "median": statistics.median(turns_list),
            "stdev": statistics.stdev(turns_list) if len(turns_list) > 1 else 0,
            "min": min(turns_list),
            "max": max(turns_list),
            "p25": sorted(turns_list)[len(turns_list) // 4],
            "p75": sorted(turns_list)[3 * len(turns_list) // 4],
        },
        "confidence_correct": {
            "mean": statistics.mean(conf_correct) if conf_correct else 0,
            "p25": sorted(conf_correct)[len(conf_correct) // 4] if len(conf_correct) > 3 else 0,
            "p75": sorted(conf_correct)[3 * len(conf_correct) // 4] if len(conf_correct) > 3 else 0,
        },
        "confidence_wrong": {
            "mean": statistics.mean(conf_wrong) if conf_wrong else 0,
        },
        "rlegal": {
            "mean": statistics.mean(rlegal_list),
            "min": min(rlegal_list),
            "max": max(rlegal_list),
        },
        "rcons": {
            "mean": statistics.mean(rcons_list),
        },
        "conflict_distribution": {
            ct: sum(1 for s in case_stats if s["conflict_type"] == ct)
            for ct in ["NONE", "SOFT", "HARD"]
        },
    }

    # ── 門檻建議 ─────────────────────────────────────────
    suggestions = []

    if len(conf_correct) > 3:
        sorted_conf = sorted(conf_correct)
        suggested_hard = sorted_conf[3 * len(sorted_conf) // 4]
        suggested_soft = sorted_conf[len(sorted_conf) // 2]
        suggestions.append(
            f"CONFIDENCE_HARD_THRESHOLD 建議值：{suggested_hard:.2f}"
            f"（正確分類 p75）（目前 0.75）"
        )
        suggestions.append(
            f"CONFIDENCE_SOFT_THRESHOLD 建議值：{suggested_soft:.2f}"
            f"（正確分類 p50）（目前 0.70）"
        )

    sorted_turns = sorted(turns_list)
    suggested_low = sorted_turns[len(sorted_turns) // 2]
    suggested_high = sorted_turns[3 * len(sorted_turns) // 4]
    suggestions.append(
        f"Rcost threshold_low 建議值：{suggested_low}（turns median）（目前 3）"
    )
    suggestions.append(
        f"Rcost threshold_high 建議值：{suggested_high}（turns p75）（目前 8）"
    )

    summary["suggestions"] = suggestions

    # ── 輸出 ─────────────────────────────────────────────
    stats_path = output_path / "pilot_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(case_stats, f, ensure_ascii=False, indent=2)

    summary_path = output_path / "pilot_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Pilot Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"樣本數：{summary['n_samples']}\n")
        f.write(f"整體準確率：{summary['accuracy']:.1%}\n\n")

        f.write("── 各類別準確率 ──\n")
        for cat, acc in sorted(per_cat_acc.items()):
            cs = cat_stats[cat]
            f.write(f"  {cat:<16} {acc:>6.1%} ({cs['correct']}/{cs['total']})\n")

        f.write("\n── 對話輪次統計 ──\n")
        t = summary["turns"]
        f.write(f"  mean={t['mean']:.1f}  median={t['median']}  "
                f"stdev={t['stdev']:.1f}  range=[{t['min']}, {t['max']}]\n")
        f.write(f"  p25={t['p25']}  p75={t['p75']}\n\n")

        f.write("── 信心分數統計 ──\n")
        cc = summary["confidence_correct"]
        f.write(f"  正確分類：mean={cc['mean']:.3f}  p25={cc['p25']:.3f}  p75={cc['p75']:.3f}\n")
        cw = summary["confidence_wrong"]
        f.write(f"  錯誤分類：mean={cw['mean']:.3f}\n\n")

        f.write("── Rlegal 統計 ──\n")
        rl = summary["rlegal"]
        f.write(f"  mean={rl['mean']:.3f}  range=[{rl['min']:.3f}, {rl['max']:.3f}]\n\n")

        f.write("── Rcons 統計 ──\n")
        f.write(f"  mean={summary['rcons']['mean']:.3f}\n\n")

        f.write("── 衝突分布 ──\n")
        for ct, n in summary["conflict_distribution"].items():
            f.write(f"  {ct}: {n}\n")

        f.write("\n── 門檻建議 ──\n")
        for s in suggestions:
            f.write(f"  • {s}\n")

    # 也存 JSON 版 summary
    summary_json_path = output_path / "pilot_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"統計結果 → {stats_path}")
    logger.info(f"摘要報告 → {summary_path}")
    logger.info(f"摘要 JSON → {summary_json_path}")

    return summary


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot Experiment（小樣本校準）")
    parser.add_argument(
        "--n_samples", type=int, default=30,
        help="總樣本數（會均勻分配到各犯罪類別）",
    )
    parser.add_argument(
        "--split", default="Test", choices=["Train", "Test", "Val"],
        help="使用哪個 split 的標註（預設 Test）",
    )
    parser.add_argument(
        "--output_dir", default="./outputs",
        help="輸出目錄",
    )
    parser.add_argument(
        "--include_normal", action="store_true",
        help="是否包含 Normal 影片",
    )
    args = parser.parse_args()

    samples = load_pilot_samples(
        n_samples=args.n_samples,
        split=args.split,
        include_normal=args.include_normal,
    )
    if samples:
        run_pilot(samples, args.output_dir)
    else:
        logger.error("無法載入樣本，請確認 UCA 資料集路徑")
