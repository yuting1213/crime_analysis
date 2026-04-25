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
from scripts.pilot_experiment import load_pilot_samples, UCF_CATEGORIES as _UCF_CATEGORIES

# 14 類分類目標（13 crime + Normal）— 與 pilot_v4 之 planner.VLM_CATEGORIES 一致
# 不含 Normal 時 VLM 無法正確拒答，導致 Precision 只有 ~0.5（所有 Normal 被判成 crime）
CRIME_CATEGORIES = list(_UCF_CATEGORIES) + ["Normal"]
# Stage-2 候選 — 兩階段流程已在 Stage 1 確認 anomaly，Stage 2 不再含 Normal
ANOMALY_CATEGORIES = list(_UCF_CATEGORIES)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 分類 Prompt（兩個模型共用）──────────────────────────────

CLASSIFY_PROMPT = (
    "You are a forensic surveillance video analyst.\n"
    "Look at these frames from a CCTV video and determine what category the scene falls into.\n"
    "If the footage shows no criminal, hazardous, or law-enforcement activity, reply Normal.\n\n"
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
    "- Arrest: Law enforcement restraining suspect\n"
    "- Normal: No criminal, hazardous, or law-enforcement activity — everyday routine scene\n\n"
    "Reply with ONLY: CATEGORY: <name>"
)

# NO-Normal prompt variant — restores the April 9 prompt style (before Fix #1
# added Normal). Forces model to pick from 13 crime categories. Used to test
# whether `Normal` escape hatch is what triggers safety-alignment collapse on
# Gemma-4 / InternVL3 (both predict 100% Normal under CLASSIFY_PROMPT).
# Wording deliberately presupposes a crime ("determine what crime is occurring").
CLASSIFY_PROMPT_NO_NORMAL = (
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

# ── Two-stage prompts（adapted from senior thesis 附錄二）────────────────
# Stage 1：偏向異常的二元偵測。只要符合任一條 ANOMALY CRITERIA 就回 yes。
# 設計重點：與單階段 prompt 相反，這裡的預設立場是「異常」，把 Normal 當成最後
# 才考慮的選項，藉此補回單階段 prompt 中過度偏向 Normal 造成的 recall 損失。
BINARY_PROMPT = (
    "You are an AI security analyst. Determine whether the provided video frames "
    "contain any anomalous or suspicious activity.\n\n"
    "ANOMALY CRITERIA — any single trigger warrants a positive detection:\n"
    "1. Overt incidents — physical attack, fighting, gunfire/weapon visible, "
    "arson/fire, explosion/blast, vehicle collision, arrest/restraint by officers, "
    "breaking into a building or vehicle, deliberate property damage.\n"
    "2. Subtle theft (Shoplifting / Stealing) — concealment of merchandise into "
    "bags/pockets/clothing, repeated furtive glances at staff or cameras, leaving "
    "the area without paying, unusual lingering combined with handling items.\n"
    "3. General deviations — people in unusual or unsafe positions, objects "
    "clearly misplaced or used improperly, visible damage to property, movement "
    "patterns inconsistent with the environment.\n\n"
    "If ANY of the above is present, reply exactly 'yes'. Otherwise reply exactly "
    "'no'. No explanations, no extra text."
)

# Stage 2：已確認異常，從 13 類中選一類。不含 Normal 避免再度逃避。
STAGE2_PROMPT = (
    "An upstream detector has confirmed this video contains anomalous activity. "
    "Your task is to identify which SINGLE category best fits the observed event.\n\n"
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

# Stage 2 解析失敗時的 fallback — 保留 Stage 1 的 binary 訊號（!= Normal）
# 但 14-way 會被計為錯。Vandalism 是最廣義的財產損害類別，當作 catch-all。
STAGE2_FALLBACK_CATEGORY = "Vandalism"


def extract_frames(video_path: str, n: int = 8) -> List[Image.Image]:
    """Uniformly sample *n* frames — delegates to the shared pipeline helper."""
    from agents.frame_utils import uniform_keyframes
    return uniform_keyframes(video_path, n)


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


def parse_yes_no(response: str) -> bool:
    """Stage-1 二元解析。回 True 代表偵測到異常。

    取首個非空白 token 比對 yes/no；無法判斷時保守回 False（傾向 Normal），
    避免無意義的 Stage 2 呼叫造成 wasted compute。
    """
    if not response:
        return False
    text = response.strip().lower()
    # 直接匹配開頭 yes/no（容忍標點和換行）
    import re
    m = re.match(r"\s*['\"`]*\s*(yes|no)\b", text)
    if m:
        return m.group(1) == "yes"
    # 後備：整段 token 中 yes 出現次數 vs no
    yes_count = len(re.findall(r"\byes\b", text))
    no_count = len(re.findall(r"\bno\b", text))
    return yes_count > no_count


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


def load_qwen3vl_32b():
    """Qwen3-VL-32B-Instruct (INT4, ~18GB) — Experiment 1 max-spec for Qwen family."""
    import torch
    from transformers import (
        Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct",
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct", trust_remote_code=True,
    )
    return model, processor


def _call_qwen3vl(model, processor, frames: List[Image.Image],
                  prompt: str, max_new_tokens: int = 64) -> str:
    """單次 Qwen3-VL 推論。抽出共用邏輯供 1-stage 與 2-stage 共用。"""
    import torch, re
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.1, do_sample=False)
    response = processor.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
    return response


def classify_qwen3vl(model, processor, frames: List[Image.Image]) -> str:
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CRIME_CATEGORIES))
    return _call_qwen3vl(model, processor, frames, prompt, max_new_tokens=64)


def classify_qwen3vl_no_normal(model, processor, frames: List[Image.Image]) -> str:
    """Same as classify_qwen3vl but uses CLASSIFY_PROMPT_NO_NORMAL — no Normal escape."""
    prompt = CLASSIFY_PROMPT_NO_NORMAL.format(categories=", ".join(ANOMALY_CATEGORIES))
    return _call_qwen3vl(model, processor, frames, prompt, max_new_tokens=64)


def classify_qwen3vl_two_stage(model, processor, frames: List[Image.Image]) -> str:
    """Two-stage classify：先二元偵測，確認異常後再做 13-way 細分類。

    Returns combined response containing both stages — parse_category 仍可
    從最後的 'CATEGORY: <name>' 抽出最終類別。
    """
    # Stage 1
    stage1 = _call_qwen3vl(model, processor, frames, BINARY_PROMPT, max_new_tokens=8)
    if not parse_yes_no(stage1):
        return f"STAGE1: {stage1}\nCATEGORY: Normal"
    # Stage 2
    stage2_prompt = STAGE2_PROMPT.format(categories=", ".join(ANOMALY_CATEGORIES))
    stage2 = _call_qwen3vl(model, processor, frames, stage2_prompt, max_new_tokens=64)
    if parse_category(stage2) == "Normal":
        # Stage 2 未命中任何 13 類 — 保留 Stage 1 binary 訊號，14-way 計為 fallback
        return (f"STAGE1: {stage1}\nSTAGE2: {stage2}\n"
                f"FALLBACK\nCATEGORY: {STAGE2_FALLBACK_CATEGORY}")
    return f"STAGE1: {stage1}\n{stage2}"


# ── InternVL3 ────────────────────────────────────────────

def load_internvl3():
    """InternVL3-8B-hf — uses AutoModelForImageTextToText for .generate() support."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-8B-hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-8B-hf", trust_remote_code=True)
    return model, processor


def load_internvl3_38b():
    """InternVL3-38B-hf (INT4, ~20GB) — Experiment 1 max-spec for InternVL family.

    Fix 2026-04-25: AutoModel was returning the bare InternVLModel class which
    lacks `.generate()` — use AutoModelForImageTextToText to get the
    InternVLForConditionalGeneration wrapper (same pattern as Gemma4).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "OpenGVLab/InternVL3-38B-hf",
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "OpenGVLab/InternVL3-38B-hf", trust_remote_code=True,
    )
    return model, processor


def _call_internvl3(model, processor, frames: List[Image.Image],
                    prompt: str, max_new_tokens: int = 64) -> str:
    import torch
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.1, do_sample=False)
    response = processor.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return response


def classify_internvl3(model, processor, frames: List[Image.Image]) -> str:
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CRIME_CATEGORIES))
    return _call_internvl3(model, processor, frames, prompt, max_new_tokens=64)


def classify_internvl3_no_normal(model, processor, frames: List[Image.Image]) -> str:
    """Same as classify_internvl3 but uses CLASSIFY_PROMPT_NO_NORMAL — no Normal escape."""
    prompt = CLASSIFY_PROMPT_NO_NORMAL.format(categories=", ".join(ANOMALY_CATEGORIES))
    return _call_internvl3(model, processor, frames, prompt, max_new_tokens=64)


def classify_internvl3_two_stage(model, processor, frames: List[Image.Image]) -> str:
    stage1 = _call_internvl3(model, processor, frames, BINARY_PROMPT, max_new_tokens=8)
    if not parse_yes_no(stage1):
        return f"STAGE1: {stage1}\nCATEGORY: Normal"
    stage2_prompt = STAGE2_PROMPT.format(categories=", ".join(ANOMALY_CATEGORIES))
    stage2 = _call_internvl3(model, processor, frames, stage2_prompt, max_new_tokens=64)
    if parse_category(stage2) == "Normal":
        return (f"STAGE1: {stage1}\nSTAGE2: {stage2}\n"
                f"FALLBACK\nCATEGORY: {STAGE2_FALLBACK_CATEGORY}")
    return f"STAGE1: {stage1}\n{stage2}"


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


def load_gemma4_26b_a4b():
    """Gemma-4-26B-A4B-it (MoE 26B/4B active, INT8) — Experiment 1 max-spec for Gemma.

    Switched 4bit NF4 → 8bit (2026-04-25): per-case time was 324s with NF4 due to
    BnB MoE-routing dequantization overhead (each token activates different experts,
    breaking weight cache). INT8 has simpler dequantization and ~26GB fits on 5090
    32GB; expected per-case time 100-150s. Disclosed in outputs/experiment_1/controls.md.
    """
    import torch
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
    )
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-26B-A4B-it",
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
    return model, processor


def _call_gemma4(model, processor, frames: List[Image.Image],
                 prompt: str, max_new_tokens: int = 16) -> str:
    import torch
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.1, do_sample=False)
    response = processor.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return response


def classify_gemma4(model, processor, frames: List[Image.Image]) -> str:
    prompt = CLASSIFY_PROMPT.format(categories=", ".join(CRIME_CATEGORIES))
    # max_new_tokens=16: actual response is ~5 tokens "CATEGORY: Normal"; 16 余裕足
    return _call_gemma4(model, processor, frames, prompt, max_new_tokens=16)


def classify_gemma4_no_normal(model, processor, frames: List[Image.Image]) -> str:
    """Same as classify_gemma4 but uses CLASSIFY_PROMPT_NO_NORMAL — no Normal escape."""
    prompt = CLASSIFY_PROMPT_NO_NORMAL.format(categories=", ".join(ANOMALY_CATEGORIES))
    return _call_gemma4(model, processor, frames, prompt, max_new_tokens=16)


def classify_gemma4_two_stage(model, processor, frames: List[Image.Image]) -> str:
    stage1 = _call_gemma4(model, processor, frames, BINARY_PROMPT, max_new_tokens=8)
    if not parse_yes_no(stage1):
        return f"STAGE1: {stage1}\nCATEGORY: Normal"
    stage2_prompt = STAGE2_PROMPT.format(categories=", ".join(ANOMALY_CATEGORIES))
    stage2 = _call_gemma4(model, processor, frames, stage2_prompt, max_new_tokens=16)
    if parse_category(stage2) == "Normal":
        return (f"STAGE1: {stage1}\nSTAGE2: {stage2}\n"
                f"FALLBACK\nCATEGORY: {STAGE2_FALLBACK_CATEGORY}")
    return f"STAGE1: {stage1}\n{stage2}"


# ── 主程式 ────────────────────────────────────────────────

def run_test(model_name: str, samples: List[Dict]) -> List[Dict]:
    """跑單一 VLM 的分類測試。"""
    import gc, torch

    if model_name == "qwen3vl":
        logger.info("載入 Qwen3-VL-8B-Instruct...")
        model, processor = load_qwen3vl()
        classify_fn = classify_qwen3vl
    elif model_name == "qwen3vl_32b":
        logger.info("載入 Qwen3-VL-32B-Instruct (INT4)...")
        model, processor = load_qwen3vl_32b()
        classify_fn = classify_qwen3vl
    elif model_name == "internvl3":
        logger.info("載入 InternVL3-8B-hf...")
        model, processor = load_internvl3()
        classify_fn = classify_internvl3
    elif model_name == "internvl3_38b":
        logger.info("載入 InternVL3-38B-hf (INT4)...")
        model, processor = load_internvl3_38b()
        classify_fn = classify_internvl3
    elif model_name == "gemma4":
        logger.info("載入 Gemma-4-E4B-it...")
        model, processor = load_gemma4()
        classify_fn = classify_gemma4
    elif model_name == "gemma4_26b_a4b":
        logger.info("載入 Gemma-4-26B-A4B-it (INT4)...")
        model, processor = load_gemma4_26b_a4b()
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
            # OOM 或其他失敗後清 GPU，給下一題 headroom
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        correct = predicted == gt
        # 顯示前 300 字給診斷用（觀察 VLM 實際輸出 vs parser 期望）
        response_preview = response.replace("\n", " | ")[:300]
        logger.info(f"  raw response: {response_preview}")
        results.append({
            "video_id": vid, "ground_truth": gt,
            "predicted": predicted, "correct": correct,
            "response": response[:500],  # 存更多上下文方便事後分析
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
    parser.add_argument("--model",
        choices=["qwen3vl", "qwen3vl_32b",
                 "internvl3", "internvl3_38b",
                 "gemma4", "gemma4_26b_a4b", "all"],
        default="all")
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
