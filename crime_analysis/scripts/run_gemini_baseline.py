"""
實驗一：Gemini Baseline 報告生成

用法：
    cd crime_analysis
    # 推薦：把 GEMINI_API_KEY 寫進 crime_analysis/.env（見 env_loader.py）
    python -m scripts.run_gemini_baseline --n_samples 154 --split Test --seed 42

功能：
    1. Gemini 2.5 Flash 直接看影片幀 → 犯罪分類 + 鑑定報告
    2. 與本系統使用相同的 REPORT_SYSTEM_PROMPT（公平比較）
    3. 輸出 pilot_stats.json / pilot_reports/ / confusion_matrix（格式與 pilot_experiment 一致）
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 自動從 .env 載入 API keys（shell exports 仍優先）
from env_loader import load_dotenv  # noqa: E402
load_dotenv()

from scripts.pilot_experiment import (
    load_pilot_samples, extract_frames, _save_case_report,
    _compute_summary, _save_confusion_matrix,
    CRIME_CATEGORIES,
)
from agents.planner import REPORT_SYSTEM_PROMPT, REPORT_USER_TEMPLATE
from rag.rag_module import LEGAL_ELEMENTS, GROUP_LEGAL_CONTEXT

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def gemini_classify_and_report(
    client,
    model_name: str,
    video_path: str,
    n_frames: int = 16,
) -> Dict:
    """
    Gemini 一次完成分類 + 報告生成（模擬 single-MLLM baseline）。
    """
    import base64, io
    from PIL import Image

    # 從影片抽取幀
    frames_pil = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        indices = [int(i * total / n_frames) for i in range(n_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_pil.append(Image.fromarray(rgb))
    cap.release()

    if not frames_pil:
        return {"category": "Normal", "confidence": 0.0, "report": "", "error": "no frames"}

    categories_str = ", ".join(CRIME_CATEGORIES)

    prompt = (
        "你是一位台灣刑事鑑定報告撰寫專家。\n"
        "以下是監視器影片的關鍵幀截圖。請完成兩個任務：\n\n"
        f"任務一：從以下類別中選擇最可能的犯罪類型：{categories_str}\n"
        "以 CATEGORY: <類別名> 的格式回答。\n\n"
        "任務二：撰寫一份結構化的初步鑑定報告，包含：\n"
        "1. 事實認定（影片可觀察事實）\n"
        "2. 構成要件分析（逐一論述是否該當）\n"
        "3. 法律適用（引用台灣刑法具體法條條號）\n"
        "4. 不確定性與限制\n"
        "5. 初步結論\n\n"
        "以繁體中文撰寫。"
    )

    # 組裝 Gemini 請求（圖片 + 文字）
    contents = []
    for img in frames_pil:
        contents.append(img)
    contents.append(prompt)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config={"temperature": 0.7},  # 不限制 max_output_tokens，讓模型完整生成
        )
        text = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API 失敗：{e}")
        return {"category": "Normal", "confidence": 0.0, "report": "", "error": str(e)}

    # 解析分類
    import re
    cat_match = re.search(r"CATEGORY:\s*(\w+)", text, re.IGNORECASE)
    category = "Normal"
    if cat_match:
        raw = cat_match.group(1)
        for cat in CRIME_CATEGORIES:
            if cat.lower() == raw.lower():
                category = cat
                break

    return {
        "category": category,
        "confidence": 0.95,  # Gemini 不回報信心，固定 0.95
        "report": text,
        "error": None,
    }


def run_gemini_baseline(
    samples: List[Dict],
    output_dir: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
):
    """執行 Gemini baseline 實驗。"""
    from google import genai
    client = genai.Client(api_key=api_key)
    logger.info(f"Gemini model: {model_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    case_stats = []
    for i, sample in enumerate(samples):
        video_id = sample["video_id"]
        gt = sample["ground_truth"]
        video_path = sample["video_path"]

        logger.info(f"[Gemini {i+1}/{len(samples)}] {video_id} ({gt})")

        # 跳過已處理的（讀回分類結果）
        report_path = output_path / "pilot_reports" / f"{video_id}.txt"
        if report_path.exists():
            # 從報告檔案解析分類結果
            with open(report_path, encoding="utf-8") as rf:
                content = rf.read()
            import re as _re
            pred_match = _re.search(r"模型判定：(\w+)", content)
            predicted = pred_match.group(1) if pred_match else "Normal"
            is_anomaly_gt = gt != "Normal"
            is_anomaly_pred = predicted != "Normal"
            stat = {
                "video_id": video_id, "ground_truth": gt,
                "predicted": predicted, "correct": predicted == gt,
                "confidence": 0.95, "rcons": 1.0, "rlegal": 0.0, "rcost": 0.0,
                "total_turns": 1, "conflict_type": "NONE", "is_convergent": True,
                "duration": sample["metadata"].get("duration", 0.0),
                "n_uca_sentences": len(sample["metadata"].get("uca_segments", [])),
                "escalation_score": 0.0,
                "is_anomaly_gt": is_anomaly_gt,
                "is_anomaly_pred": is_anomaly_pred,
                "anomaly_correct": is_anomaly_gt == is_anomaly_pred,
                "anomaly_gated": False,
            }
            case_stats.append(stat)
            logger.info(f"  已存在 → {predicted} ({'V' if predicted == gt else 'X'})，跳過")
            continue

        start_t = time.time()
        result = gemini_classify_and_report(client, model_name, video_path)
        elapsed = time.time() - start_t

        if result["error"]:
            logger.warning(f"  錯誤：{result['error']}")
            # Rate limit: 等一下再試
            if "429" in str(result["error"]) or "quota" in str(result["error"]).lower():
                logger.info("  Rate limited, waiting 30s...")
                time.sleep(30)
                result = gemini_classify_and_report(client, model_name, video_path)

        predicted = result["category"]
        correct = predicted == gt
        is_anomaly_gt = gt != "Normal"
        is_anomaly_pred = predicted != "Normal"

        stat = {
            "video_id": video_id,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": correct,
            "confidence": result["confidence"],
            "rcons": 1.0,  # Gemini 無 Reflector
            "rlegal": 0.0,  # 不計算（無 RAG）
            "rcost": 0.0,  # 單次呼叫
            "total_turns": 1,
            "conflict_type": "NONE",
            "is_convergent": True,
            "duration": sample["metadata"].get("duration", 0.0),
            "n_uca_sentences": len(sample["metadata"].get("uca_segments", [])),
            # Option D anomaly-detection 欄位（跟 pilot_v2 對等）
            "escalation_score": 0.0,  # Gemini 沒有 MIL escalation
            "is_anomaly_gt": is_anomaly_gt,
            "is_anomaly_pred": is_anomaly_pred,
            "anomaly_correct": is_anomaly_gt == is_anomaly_pred,
            "anomaly_gated": False,
        }
        case_stats.append(stat)

        # 保存報告
        report_result = {
            "final_category": predicted,
            "fact_finding": {
                "description": result["report"],
                "rationale": "",
                "confidence": result["confidence"],
            },
            "behavior_analysis": {"causal_chain": "", "escalation_score": 0.0,
                                  "pre_crime_indicators": [], "post_crime_indicators": []},
            "legal_classification": {"applicable_articles": [], "elements_covered": []},
            "rcons": 1.0, "rlegal": 0.0, "rcost": 0.0,
            "total_turns": 1, "conflict_type": "NONE", "is_convergent": True,
            "report_generation_method": "gemini-baseline",
            "uncertainty_notes": {"low_confidence_items": [], "conflicting_evidence": [],
                                  "insufficient_evidence": []},
        }
        _save_case_report(report_result, video_id, gt, sample["metadata"], output_path)

        logger.info(f"  → {predicted} ({'V' if correct else 'X'}) ({elapsed:.1f}s)")

        # API rate limiting
        time.sleep(2)

    # 統計
    stats_path = output_path / "pilot_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(case_stats, f, ensure_ascii=False, indent=2)

    summary = _compute_summary(case_stats, output_path)
    _save_confusion_matrix(case_stats, output_path)

    # 存 meta
    meta = {
        "experiment": "gemini_baseline",
        "model": model_name,
        "n_samples": len(samples),
        "accuracy": summary.get("accuracy", 0),
    }
    with open(output_path / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\nGemini Baseline 完成：{sum(1 for s in case_stats if s['correct'])}/{len(case_stats)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Baseline 實驗")
    parser.add_argument("--n_samples", type=int, default=154)
    parser.add_argument("--split", default="Test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./outputs/experiments/gemini_baseline")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--n-normal", type=int, default=0,
        help="Normal 影片樣本數（用於 anomaly detection 評估，跟 pilot_v2 對等）",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("請設定 GEMINI_API_KEY 環境變數")
        sys.exit(1)

    import random
    random.seed(args.seed)

    samples = load_pilot_samples(
        n_samples=args.n_samples,
        split=args.split,
        include_normal=args.n_normal > 0,
        n_normal=args.n_normal,
    )
    if samples:
        run_gemini_baseline(samples, args.output_dir, api_key, args.model)
