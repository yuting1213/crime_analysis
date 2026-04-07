"""
QLoRA Fine-tune Qwen3-VL-32B for UCF-Crime Classification

用法：
    cd crime_analysis
    python -m scripts.finetune_vlm --epochs 3

VRAM: ~24GB（32B INT4 base 18GB + LoRA gradients ~6GB）
時間: ~3-5 小時（576 影片 × 3 epochs）

產出：
    outputs/vlm_finetune/
    ├── adapter_model/     # LoRA adapter weights
    ├── training_log.json
    └── training_curves.png
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pilot_experiment import (
    load_uca_annotations, video_id_to_category, CRIME_CATEGORIES, VIDEOS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./outputs/vlm_finetune")

# ── 資料準備 ─────────────────────────────────────────────

def prepare_training_data(
    split: str = "Train",
    n_frames: int = 4,
    max_per_category: int = 50,
) -> List[Dict]:
    """
    準備 VLM fine-tune 資料：每支影片取 n_frames 幀 + 類別標籤。
    """
    annotations = load_uca_annotations(split)
    if not annotations:
        return []

    category_counts = {}
    samples = []

    for video_id, ann in annotations.items():
        cat = video_id_to_category(video_id)
        if cat == "Normal" or cat not in CRIME_CATEGORIES:
            continue
        if category_counts.get(cat, 0) >= max_per_category:
            continue

        # 找影片
        video_path = VIDEOS_DIR / cat / f"{video_id}.mp4"
        if not video_path.exists():
            continue

        # 抽幀（使用 UCA 引導：70% 犯罪時段）
        frames_pil = extract_training_frames(
            str(video_path), ann, n_frames,
        )
        if not frames_pil:
            continue

        samples.append({
            "video_id": video_id,
            "category": cat,
            "frames": frames_pil,
        })
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info(f"訓練資料：{len(samples)} 支影片")
    logger.info(f"  類別分布：{category_counts}")
    return samples


def extract_training_frames(
    video_path: str,
    ann: Dict,
    n_frames: int = 4,
) -> List[Image.Image]:
    """從影片抽取訓練幀（UCA 引導）。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        return []

    # UCA 引導：70% 犯罪時段
    timestamps = ann.get("timestamps", [])
    event_frames = set()
    for ts in timestamps:
        sf = int(ts[0] * fps)
        ef = int(ts[1] * fps)
        for i in range(max(0, sf), min(ef + 1, total)):
            event_frames.add(i)

    n_event = int(n_frames * 0.7)
    n_context = n_frames - n_event

    if event_frames:
        elist = sorted(event_frames)
        if len(elist) >= n_event:
            ev_indices = [elist[int(i * len(elist) / n_event)] for i in range(n_event)]
        else:
            ev_indices = elist
    else:
        ev_indices = []

    ctx_indices = [int(i * total / max(n_context, 1)) for i in range(n_context)]
    all_indices = sorted(set(ev_indices + ctx_indices))[:n_frames]

    frames = []
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ── Fine-tune ────────────────────────────────────────────

def finetune(
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 1,
    n_frames: int = 4,
    max_per_category: int = 50,
):
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: 準備資料
    logger.info("Step 1: 準備訓練資料...")
    samples = prepare_training_data(
        split="Train", n_frames=n_frames, max_per_category=max_per_category,
    )
    if not samples:
        logger.error("無訓練資料")
        return

    random.shuffle(samples)

    # Step 2: 載入模型（INT4 量化）
    logger.info("Step 2: 載入 Qwen3-VL-32B INT4...")
    model_name = "Qwen/Qwen3-VL-32B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Step 3: 設定 LoRA
    logger.info("Step 3: 設定 QLoRA...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA 參數：{trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    # Step 4: 訓練
    categories_str = ", ".join(CRIME_CATEGORIES)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    training_log = []
    total_start = time.time()

    logger.info(f"Step 4: 開始訓練 — {epochs} epochs, {len(samples)} samples, lr={lr}")

    for epoch in range(epochs):
        random.shuffle(samples)
        epoch_loss = 0.0
        n_steps = 0

        for i, sample in enumerate(samples):
            cat = sample["category"]
            frames = sample["frames"]

            # 組裝訓練 prompt + 正確答案
            prompt = (
                f"You are a forensic surveillance video analyst.\n"
                f"Look at these frames from a CCTV video and determine what crime is occurring.\n\n"
                f"Choose ONE category from: {categories_str}\n\n"
                f"Reply with ONLY: CATEGORY: <name>"
            )
            answer = f"CATEGORY: {cat}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in frames],
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]

            # Tokenize
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            inputs = processor(
                text=[text],
                images=frames,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            # Forward + loss
            inputs["labels"] = inputs["input_ids"].clone()
            outputs = model(**inputs)
            loss = outputs.loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            if (i + 1) % 50 == 0:
                avg = epoch_loss / n_steps
                logger.info(f"  Epoch {epoch+1} [{i+1}/{len(samples)}] loss={avg:.4f}")

        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - total_start
        log_entry = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "elapsed_min": round(elapsed / 60, 1),
        }
        training_log.append(log_entry)
        logger.info(
            f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
            f"elapsed={elapsed/60:.1f}min"
        )

    # Step 5: 儲存 adapter
    adapter_dir = OUTPUT_DIR / "adapter_model"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    logger.info(f"LoRA adapter → {adapter_dir}")

    # 儲存 log
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # 視覺化
    _plot_training(training_log, OUTPUT_DIR)

    total_elapsed = time.time() - total_start
    logger.info(f"Fine-tune 完成！{total_elapsed/60:.1f} 分鐘")


def _plot_training(log, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [e["epoch"] for e in log]
        losses = [e["loss"] for e in log]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, losses, "b-o", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("QLoRA Fine-tune: Qwen3-VL-32B on UCF-Crime")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-tune Qwen3-VL-32B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_frames", type=int, default=4, help="每支影片的幀數（越少越快）")
    parser.add_argument("--max_per_category", type=int, default=50, help="每類最多幾支影片")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    finetune(
        epochs=args.epochs,
        lr=args.lr,
        n_frames=args.n_frames,
        max_per_category=args.max_per_category,
    )
