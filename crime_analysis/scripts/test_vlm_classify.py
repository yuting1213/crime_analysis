"""
VLM 分類能力診斷腳本 — 測試不同 VLM 的 zero-shot 分類準確率

用法：
    cd crime_analysis
    # Qwen3-VL-8B（目前使用的）
    python -m scripts.test_vlm_classify --model qwen3vl

    # InternVL3-8B（對照組）
    python -m scripts.test_vlm_classify --model internvl3

    # 比較
    python -m scripts.test_vlm_classify --model both

目的：確認分類瓶頸在 VLM 模型還是 framework。
不經過任何 Agent/RAG/Reflector，純粹測 VLM 的分類能力。
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pilot_experiment import load_pilot_samples, CRIME_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 分類 Prompt（兩個模型共用）──────────────────────────────

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


def extract_frames(video_path: str, n: int = 16) -> List[Image.Image]:
    """從影片均勻抽取 n 幀，回傳 PIL Image list。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def parse_category(response: str) -> str:
    """從 VLM 回應解析類別。"""
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


# ── Qwen3-VL ────────────────────────────────────────────

def load_qwen3vl():
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    return model, processor


def classify_qwen3vl(model, processor, frames: List[Image.Image]) -> str:
    import torch, re
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CRIME_CATEGORIES))
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in frames],
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.1, do_sample=False)
    response = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
    return response


# ── InternVL3 ────────────────────────────────────────────

def load_internvl3():
    import torch
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL3-8B-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-8B-hf", trust_remote_code=True)
    return model, processor


def classify_internvl3(model, processor, frames: List[Image.Image]) -> str:
    import torch
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CRIME_CATEGORIES))
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in frames],
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.1, do_sample=False)
    response = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response


# ── Gemma 4 ──────────────────────────────────────────────

def load_gemma4():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-E4B-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
    return model, processor


def classify_gemma4(model, processor, frames: List[Image.Image]) -> str:
    import torch
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CRIME_CATEGORIES))
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in frames],
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    response = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response


# ── 主程式 ────────────────────────────────────────────────

def run_test(model_name: str, samples: List[Dict]) -> List[Dict]:
    """跑單一 VLM 的分類測試。"""
    import gc, torch

    if model_name == "qwen3vl":
        logger.info("載入 Qwen3-VL-8B-Instruct...")
        model, processor = load_qwen3vl()
        classify_fn = classify_qwen3vl
    elif model_name == "internvl3":
        logger.info("載入 InternVL3-8B-hf...")
        model, processor = load_internvl3()
        classify_fn = classify_internvl3
    elif model_name == "gemma4":
        logger.info("載入 Gemma-4-E4B-it...")
        model, processor = load_gemma4()
        classify_fn = classify_gemma4
    else:
        raise ValueError(f"Unknown model: {model_name}")

    results = []
    for i, sample in enumerate(samples):
        vid = sample["video_id"]
        gt = sample["ground_truth"]
        logger.info(f"[{model_name} {i+1}/{len(samples)}] {vid} ({gt})")

        frames = extract_frames(sample["video_path"], n=8)
        if not frames:
            logger.warning(f"  無法讀取影片")
            continue

        try:
            response = classify_fn(model, processor, frames)
            predicted = parse_category(response)
        except Exception as e:
            logger.error(f"  失敗：{e}")
            predicted = "Normal"
            response = str(e)

        correct = predicted == gt
        results.append({
            "video_id": vid, "ground_truth": gt,
            "predicted": predicted, "correct": correct,
            "response": response[:100],
        })
        logger.info(f"  → {predicted} ({'V' if correct else 'X'})")

    # 清理 VRAM
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_results(name: str, results: List[Dict]):
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  {name}: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")

    cats = sorted(set(r["ground_truth"] for r in results))
    for cat in cats:
        items = [r for r in results if r["ground_truth"] == cat]
        c = sum(1 for r in items if r["correct"])
        print(f"  {cat:15s}: {c}/{len(items)}")

    pred_dist = Counter(r["predicted"] for r in results)
    print(f"\n  Prediction dist: {dict(sorted(pred_dist.items()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM 分類診斷")
    parser.add_argument("--model", choices=["qwen3vl", "internvl3", "gemma4", "all"], default="all")
    parser.add_argument("--n_samples", type=int, default=13, help="每類 1 個 = 13")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    samples = load_pilot_samples(n_samples=args.n_samples, split="Test")

    models_to_test = [args.model] if args.model not in ("both", "all") else ["qwen3vl", "internvl3", "gemma4"]
    all_results = {}

    for m in models_to_test:
        results = run_test(m, samples)
        all_results[m] = results
        print_results(m, results)

        # 存結果
        out_path = Path(f"outputs/vlm_diagnostic_{m}.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 比較
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  比較")
        print(f"{'='*60}")
        for m, results in all_results.items():
            c = sum(1 for r in results if r["correct"])
            print(f"  {m:15s}: {c}/{len(results)} ({100*c/len(results):.1f}%)")
