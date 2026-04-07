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

    training_log = {
        "step_losses": [],       # 每 step 的 loss
        "epoch_losses": [],      # 每 epoch 平均 loss
        "cat_losses": [],        # 每 epoch 各類別平均 loss
        "val_accuracy": [],      # 每 epoch 結束的 validation 準確率
    }
    total_start = time.time()
    global_step = 0

    # 準備 validation set（13 支，每類 1 個，從 Test split）
    val_samples = prepare_training_data(split="Test", n_frames=n_frames, max_per_category=1)
    logger.info(f"Validation set: {len(val_samples)} 支")

    logger.info(f"Step 4: 開始訓練 — {epochs} epochs, {len(samples)} samples, lr={lr}")

    for epoch in range(epochs):
        random.shuffle(samples)
        epoch_loss = 0.0
        n_steps = 0
        cat_losses: Dict[str, List[float]] = {c: [] for c in CRIME_CATEGORIES}

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

            step_loss = loss.item()
            epoch_loss += step_loss
            n_steps += 1
            global_step += 1
            cat_losses[cat].append(step_loss)
            training_log["step_losses"].append({
                "step": global_step, "loss": round(step_loss, 4), "category": cat,
            })

            if (i + 1) % 20 == 0:
                avg = epoch_loss / n_steps
                elapsed = time.time() - total_start
                logger.info(
                    f"  Epoch {epoch+1} [{i+1}/{len(samples)}] "
                    f"loss={avg:.4f} step_loss={step_loss:.4f} "
                    f"({elapsed/60:.1f}min)"
                )

        # Epoch 統計
        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - total_start

        # Per-category loss
        cat_avg = {c: round(sum(v)/len(v), 4) if v else 0.0 for c, v in cat_losses.items()}
        training_log["cat_losses"].append({"epoch": epoch + 1, **cat_avg})

        # Validation：快速跑 13 支 test 看分類準確率
        val_acc = _validate(model, processor, val_samples, categories_str)
        training_log["val_accuracy"].append({
            "epoch": epoch + 1, "accuracy": val_acc,
        })

        training_log["epoch_losses"].append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "elapsed_min": round(elapsed / 60, 1),
            "val_accuracy": val_acc,
        })
        logger.info(
            f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | "
            f"val_acc={val_acc:.1%} | elapsed={elapsed/60:.1f}min"
        )
        logger.info(f"  Per-cat loss: {cat_avg}")

        # 每 epoch 更新視覺化（可即時查看）
        _plot_training(training_log, OUTPUT_DIR)

        # 每 epoch 儲存 log（斷電也不丟）
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    # Step 5: 儲存 adapter
    adapter_dir = OUTPUT_DIR / "adapter_model"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    logger.info(f"LoRA adapter → {adapter_dir}")

    total_elapsed = time.time() - total_start
    logger.info(f"Fine-tune 完成！{total_elapsed/60:.1f} 分鐘")


def _validate(model, processor, val_samples, categories_str):
    """快速 validation：跑 val_samples 看分類準確率。"""
    import re
    correct = 0
    total = 0

    for sample in val_samples:
        cat = sample["category"]
        frames = sample["frames"]

        prompt = (
            f"You are a forensic surveillance video analyst.\n"
            f"Look at these frames from a CCTV video and determine what crime is occurring.\n\n"
            f"Choose ONE category from: {categories_str}\n\n"
            f"Reply with ONLY: CATEGORY: <name>"
        )

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in frames],
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, temperature=0.1, do_sample=False)
            resp = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            resp = re.sub(r"<think>.*?</think>\s*", "", resp, flags=re.DOTALL).strip()

            # 解析
            m = re.search(r"CATEGORY:\s*(\w+)", resp, re.IGNORECASE)
            if m:
                pred = m.group(1)
                for c in CRIME_CATEGORIES:
                    if c.lower() == pred.lower():
                        if c == cat:
                            correct += 1
                        break
            total += 1
        except Exception:
            total += 1

    return correct / max(total, 1)


def _plot_training(log, output_dir):
    """4-panel training dashboard，每 epoch 更新。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("QLoRA Fine-tune Dashboard: Qwen3-VL-32B on UCF-Crime", fontsize=14, fontweight="bold")

    # 1. Step-level Loss（每個訓練步的 loss）
    ax = axes[0, 0]
    step_data = log.get("step_losses", [])
    if step_data:
        steps = [s["step"] for s in step_data]
        losses = [s["loss"] for s in step_data]
        ax.plot(steps, losses, "b-", alpha=0.3, linewidth=0.5)
        # 滑動平均
        window = min(20, len(losses))
        if window > 1:
            smooth = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax.plot(steps[window-1:], smooth, "r-", linewidth=2, label=f"MA-{window}")
            ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Step Loss (raw + moving avg)")
    ax.grid(True, alpha=0.3)

    # 2. Epoch Loss + Validation Accuracy（雙 Y 軸）
    ax = axes[0, 1]
    epoch_data = log.get("epoch_losses", [])
    if epoch_data:
        epochs = [e["epoch"] for e in epoch_data]
        ep_losses = [e["loss"] for e in epoch_data]
        ax.plot(epochs, ep_losses, "b-o", linewidth=2, label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")

        # 右軸：validation accuracy
        val_data = log.get("val_accuracy", [])
        if val_data:
            ax2 = ax.twinx()
            val_acc = [v["accuracy"] for v in val_data]
            ax2.plot(epochs[:len(val_acc)], val_acc, "g-s", linewidth=2, label="Val Accuracy")
            ax2.set_ylabel("Accuracy", color="green")
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.set_ylim(0, 1)
            ax2.legend(loc="center right")
    ax.set_title("Epoch Loss + Validation Accuracy")
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)

    # 3. Per-Category Loss（每類的 loss 變化）
    ax = axes[1, 0]
    cat_data = log.get("cat_losses", [])
    if cat_data:
        categories = [c for c in cat_data[0].keys() if c != "epoch"]
        epochs = [e["epoch"] for e in cat_data]
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
        for cat, color in zip(sorted(categories), colors):
            values = [e.get(cat, 0) for e in cat_data]
            if any(v > 0 for v in values):
                ax.plot(epochs, values, "-o", color=color, linewidth=1.5, markersize=4, label=cat)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Per-Category Loss")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)

    # 4. Step Loss 分布（histogram）
    ax = axes[1, 1]
    if step_data:
        # 按 epoch 分組的 loss 分布
        epoch_nums = sorted(set(
            1 + s["step"] // max(len(step_data) // max(len(epoch_data), 1), 1)
            for s in step_data
        ))
        n_epochs = len(log.get("epoch_losses", []))
        if n_epochs > 0:
            steps_per_epoch = len(step_data) // n_epochs
            for ep in range(n_epochs):
                start = ep * steps_per_epoch
                end = start + steps_per_epoch
                ep_losses = [s["loss"] for s in step_data[start:end]]
                if ep_losses:
                    ax.hist(ep_losses, bins=30, alpha=0.5, label=f"Epoch {ep+1}")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Count")
        ax.set_title("Loss Distribution per Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Dashboard → {output_dir / 'training_curves.png'}")


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
