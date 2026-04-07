"""
用 Ollama 測試 VLM 分類能力（支援大模型量化版）

用法：
    ollama pull qwen3-vl:32b
    cd crime_analysis
    python -m scripts.test_ollama_classify --model qwen3-vl:32b --n_samples 13
    python -m scripts.test_ollama_classify --model qwen3-vl:8b --n_samples 13
"""
import argparse
import base64
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pilot_experiment import load_pilot_samples, CRIME_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

CLASSIFY_PROMPT = (
    "You are a forensic surveillance video analyst.\n"
    "Look at these frames from a CCTV video and determine what crime is occurring.\n\n"
    "Choose ONE category from: {categories}\n\n"
    "Category definitions:\n"
    "- Assault: One person attacking another (one-sided)\n"
    "- Robbery: Forcibly taking someone's belongings\n"
    "- Stealing: Secretly taking items\n"
    "- Shoplifting: Concealing store merchandise\n"
    "- Burglary: Breaking into a building/car\n"
    "- Fighting: Mutual physical combat (both sides)\n"
    "- Arson: Deliberately setting fire\n"
    "- Explosion: Sudden blast with smoke/debris\n"
    "- RoadAccidents: Vehicle collision\n"
    "- Vandalism: Deliberately damaging property\n"
    "- Abuse: Sustained harm to vulnerable person\n"
    "- Shooting: Gunfire, weapon visible\n"
    "- Arrest: Law enforcement restraining suspect\n\n"
    "Reply with ONLY: CATEGORY: <name>"
)


def extract_frames_base64(video_path: str, n: int = 8) -> List[str]:
    """從影片抽幀，回傳 base64 encoded JPEG list。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * total / n) for i in range(n)]
    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_b64.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return frames_b64


def classify_ollama(model: str, frames_b64: List[str]) -> str:
    """用 Ollama API 分類。"""
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CRIME_CATEGORIES))

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": frames_b64,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 64,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()
    return result.get("message", {}).get("content", "")


def parse_category(response: str) -> str:
    import re
    cat_match = re.search(r"CATEGORY:\s*(\w+)", response, re.IGNORECASE)
    if cat_match:
        raw = cat_match.group(1)
        for cat in CRIME_CATEGORIES:
            if cat.lower() == raw.lower():
                return cat
    for cat in CRIME_CATEGORIES:
        if cat.lower() in response.lower():
            return cat
    return "Normal"


def main():
    parser = argparse.ArgumentParser(description="Ollama VLM 分類測試")
    parser.add_argument("--model", default="qwen3-vl:32b")
    parser.add_argument("--n_samples", type=int, default=13)
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    samples = load_pilot_samples(n_samples=args.n_samples, split="Test")

    results = []
    for i, s in enumerate(samples):
        vid = s["video_id"]
        gt = s["ground_truth"]
        logger.info(f"[{args.model} {i+1}/{len(samples)}] {vid} ({gt})")

        frames_b64 = extract_frames_base64(s["video_path"], n=args.n_frames)
        if not frames_b64:
            logger.warning("  無法讀取影片")
            continue

        try:
            start = time.time()
            response = classify_ollama(args.model, frames_b64)
            elapsed = time.time() - start
            predicted = parse_category(response)
        except Exception as e:
            logger.error(f"  失敗：{e}")
            predicted = "Normal"
            elapsed = 0

        correct = predicted == gt
        results.append({
            "video_id": vid, "ground_truth": gt,
            "predicted": predicted, "correct": correct,
            "response": response[:100], "elapsed": round(elapsed, 1),
        })
        ok = "V" if correct else "X"
        logger.info(f"  → {predicted} {ok} ({elapsed:.1f}s)")

    c = sum(1 for r in results if r["correct"])
    print(f"\n{args.model}: {c}/{len(results)} ({100*c/len(results):.1f}%)")

    cats = sorted(set(r["ground_truth"] for r in results))
    for cat in cats:
        items = [r for r in results if r["ground_truth"] == cat]
        cc = sum(1 for r in items if r["correct"])
        print(f"  {cat:15s}: {cc}/{len(items)}")

    out_path = Path(f"outputs/vlm_diagnostic_ollama_{args.model.replace(':', '_')}.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果 → {out_path}")


if __name__ == "__main__":
    main()
