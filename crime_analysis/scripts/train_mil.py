"""
MIL Head 訓練腳本

訓練 FusionEncoder + crime_head + escalation_head。
使用 UCF-Crime video-level labels（weakly supervised）。

步驟：
  1. 預提取所有影片的 R3D-18 snippet 特徵 → 快取到 .npy
  2. 每個 epoch 隨機配對 (anomaly, normal) 影片
  3. 損失 = CE_loss (分類) + λ × MIL_ranking_loss (異常偵測)

使用方式：
  cd crime_analysis
  python -m scripts.train_mil --epochs 30 --lr 1e-4 --batch_size 16

產出：
  outputs/mil_weights/
    ├── fusion_encoder.pt
    ├── crime_head.pt
    ├── escalation_head.pt
    └── training_log.json
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# 加入 crime_analysis 到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from agents.action_emotion_agent import (
    FusionEncoder, FUSION_DIM, NUM_CLASSES, NUM_SNIPPETS,
    FRAMES_PER_CLIP, R3D_DIM, UCF_CATEGORIES,
)
from scripts.pilot_experiment import VIDEOS_DIR, UCA_ROOT, CRIME_CATEGORIES

# ── 設定 ─────────────────────────────────────────────────────
FEATURE_CACHE_DIR = Path("./outputs/feature_cache")
WEIGHTS_DIR = Path("./outputs/mil_weights")


# ── 特徵預提取 ───────────────────────────────────────────────

# 特徵提取版本 — 改了取幀策略/模型時遞增此值，會自動使舊快取失效
FEATURE_VERSION = "v2"  # v1=前16幀+4幀ViT, v2=均勻16幀+8幀ViT


def extract_and_cache_features(
    split: str = "Train",
    force: bool = False,
) -> Dict[str, Path]:
    """
    用 R3D-18 + ViT 預提取所有影片的特徵並存為 .npy。
    快取檔名包含版本號，避免舊參數的快取被誤用。
    回傳 {video_id: npy_path} 映射。
    """
    cache_dir = FEATURE_CACHE_DIR / FEATURE_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 收集影片
    from scripts.pilot_experiment import load_uca_annotations, video_id_to_category
    annotations = load_uca_annotations(split)
    if not annotations:
        logger.error("無法載入 UCA 標註")
        return {}

    # 初始化 R3D-18 + ViT
    import torchvision.models.video as video_models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r3d = video_models.r3d_18(weights="DEFAULT")
    r3d.fc = nn.Identity()
    r3d = r3d.to(device).eval()
    logger.info(f"R3D-18 載入完成（{device}）")

    vit_model, vit_proc = None, None
    try:
        from transformers import ViTModel, ViTImageProcessor
        vit_proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
        logger.info("ViT-Base 載入完成")
    except Exception as e:
        logger.warning(f"ViT 載入失敗，將只用 R3D-18：{e}")

    cache_map = {}
    processed = 0
    skipped = 0

    for video_id in annotations:
        cat = video_id_to_category(video_id)
        npy_path = cache_dir / f"{video_id}.npy"

        if npy_path.exists() and not force:
            cache_map[video_id] = npy_path
            skipped += 1
            continue

        # 找影片
        if cat == "Normal":
            video_path = None
            for d in ["Training_Normal_Videos_Anomaly",
                       "Testing_Normal_Videos_Anomaly",
                       "z_Normal_Videos_event"]:
                p = VIDEOS_DIR / d / f"{video_id}.mp4"
                if p.exists():
                    video_path = p
                    break
        else:
            video_path = VIDEOS_DIR / cat / f"{video_id}.mp4"

        if video_path is None or not video_path.exists():
            continue

        # 抽幀 + 提取特徵（R3D-18 512D + ViT 768D + 零填充 106D = 1386D）
        try:
            frames = _load_video_frames(str(video_path))
            r3d_feat = _extract_r3d_feature(r3d, frames, device)
            vit_feat = _extract_vit_feature(vit_model, vit_proc, frames, device) if vit_model else np.zeros(768, dtype=np.float32)
            # 組裝 1386D = R3D(512) + ViT(768) + Pose(99) + Emotion(7)
            fusion_feat = np.zeros(FUSION_DIM, dtype=np.float32)
            fusion_feat[:R3D_DIM] = r3d_feat
            fusion_feat[R3D_DIM:R3D_DIM+768] = vit_feat
            # Pose + Emotion 留零（這兩個需要 MediaPipe/DeepFace）
            np.save(npy_path, fusion_feat)
            cache_map[video_id] = npy_path
            processed += 1

            if processed % 50 == 0:
                logger.info(f"  已提取 {processed} 部影片特徵")
        except Exception as e:
            logger.warning(f"  {video_id} 特徵提取失敗：{e}")

    logger.info(f"特徵提取完成：{processed} 新提取，{skipped} 已快取")
    return cache_map


def _load_video_frames(video_path: str, n_frames: int = 32) -> List[np.ndarray]:
    """載入影片幀。"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _extract_r3d_feature(
    model: nn.Module,
    frames: List[np.ndarray],
    device: str,
) -> np.ndarray:
    """從幀序列提取 R3D-18 512D 特徵。"""
    import cv2

    if len(frames) < 2:
        return np.zeros(R3D_DIM, dtype=np.float32)

    # 均勻取 16 幀（而非只取前 16 幀）
    n = len(frames)
    n_clip = 16
    if n >= n_clip:
        indices = np.linspace(0, n - 1, n_clip, dtype=int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames + [frames[-1]] * (n_clip - n)

    processed = []
    for frame in selected:
        resized = cv2.resize(frame, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.43216, 0.394666, 0.37645])) / \
                     np.array([0.22803, 0.22145, 0.216989])
        processed.append(normalized)

    clip = np.stack(processed)  # (T, H, W, 3)
    clip = clip.transpose(3, 0, 1, 2)  # (3, T, H, W)
    clip_tensor = torch.from_numpy(clip).float().unsqueeze(0).to(device)  # (1, 3, T, H, W)

    with torch.no_grad():
        feat = model(clip_tensor)  # (1, 512)
        feat = F.normalize(feat, p=2, dim=-1)

    return feat.cpu().numpy()[0]


def _extract_vit_feature(
    model: nn.Module,
    processor,
    frames: List[np.ndarray],
    device: str,
) -> np.ndarray:
    """從關鍵幀提取 ViT CLS token 768D 特徵（取平均）。"""
    import cv2
    from PIL import Image as PILImage

    if model is None or len(frames) < 1:
        return np.zeros(768, dtype=np.float32)

    # 均勻取 8 幀（涵蓋更多時序資訊）
    n_vit = 8
    n = len(frames)
    if n >= n_vit:
        indices = np.linspace(0, n - 1, n_vit, dtype=int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames

    feats = []
    for frame in selected:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_feat = outputs.last_hidden_state[:, 0, :]  # (1, 768)
            feats.append(cls_feat.cpu().numpy()[0])

    return np.mean(feats, axis=0).astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────

class UCFCrimeDataset(Dataset):
    """
    UCF-Crime 訓練資料集。
    每個樣本：(feature_512d, category_idx, is_anomaly)
    """

    def __init__(
        self,
        cache_map: Dict[str, Path],
        split: str = "Train",
    ):
        from scripts.pilot_experiment import video_id_to_category

        self.samples = []
        self.anomaly_ids = []
        self.normal_ids = []

        for video_id, npy_path in cache_map.items():
            cat = video_id_to_category(video_id)
            if cat == "Normal":
                cat_idx = -1
                self.normal_ids.append(len(self.samples))
            elif cat in CRIME_CATEGORIES:
                cat_idx = CRIME_CATEGORIES.index(cat)
                self.anomaly_ids.append(len(self.samples))
            else:
                continue

            self.samples.append({
                "video_id": video_id,
                "npy_path": str(npy_path),
                "category": cat,
                "category_idx": cat_idx,
                "is_anomaly": cat != "Normal",
            })

        logger.info(
            f"Dataset: {len(self.samples)} 影片"
            f"（異常 {len(self.anomaly_ids)} / 正常 {len(self.normal_ids)}）"
        )

    def __len__(self):
        return len(self.anomaly_ids)  # 以異常影片數為基準

    def __getitem__(self, idx):
        # 取一個異常影片
        anom_sample = self.samples[self.anomaly_ids[idx]]
        anom_feat = np.load(anom_sample["npy_path"])

        # 隨機配一個正常影片（MIL pair）
        norm_idx = random.choice(self.normal_ids)
        norm_sample = self.samples[norm_idx]
        norm_feat = np.load(norm_sample["npy_path"])

        # 確保維度正確（相容舊的 512D 快取和新的 1386D 快取）
        if len(anom_feat) < FUSION_DIM:
            padded = np.zeros(FUSION_DIM, dtype=np.float32)
            padded[:len(anom_feat)] = anom_feat
            anom_feat = padded
        if len(norm_feat) < FUSION_DIM:
            padded = np.zeros(FUSION_DIM, dtype=np.float32)
            padded[:len(norm_feat)] = norm_feat
            norm_feat = padded

        return {
            "anom_feat": torch.from_numpy(anom_feat),
            "norm_feat": torch.from_numpy(norm_feat),
            "category_idx": anom_sample["category_idx"],
            "video_id": anom_sample["video_id"],
        }


# ── 訓練迴圈 ─────────────────────────────────────────────────

def train(
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 16,
    lambda_mil: float = 0.5,
    split: str = "Train",
):
    """
    訓練 FusionEncoder + crime_head + escalation_head。

    損失 = CE_loss（分類）+ λ × MIL_ranking_loss（異常偵測）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: 預提取特徵
    logger.info("Step 1: 預提取 R3D-18 特徵...")
    cache_map = extract_and_cache_features(split=split)
    if not cache_map:
        logger.error("無特徵可用")
        return

    # Step 2: 建立 Dataset
    dataset = UCFCrimeDataset(cache_map, split=split)
    if len(dataset) == 0:
        logger.error("Dataset 為空")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    # Step 3: 初始化模型
    fusion_encoder = FusionEncoder(
        input_dim=FUSION_DIM, d_model=512, nhead=8, num_layers=4,
    ).to(device)
    crime_head = nn.Sequential(
        nn.Linear(512, NUM_CLASSES),
    ).to(device)  # 不加 Softmax，CE loss 自帶
    escalation_head = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid(),
    ).to(device)

    # Optimizer
    params = (
        list(fusion_encoder.parameters())
        + list(crime_head.parameters())
        + list(escalation_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss_fn = nn.CrossEntropyLoss()

    # Step 4: 訓練
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    training_log = []
    best_loss = float("inf")

    logger.info(f"開始訓練：{epochs} epochs, lr={lr}, batch={batch_size}, λ_MIL={lambda_mil}")

    for epoch in range(epochs):
        fusion_encoder.train()
        crime_head.train()
        escalation_head.train()

        epoch_ce_loss = 0.0
        epoch_mil_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            anom_feat = batch["anom_feat"].to(device)   # (B, FUSION_DIM)
            norm_feat = batch["norm_feat"].to(device)   # (B, FUSION_DIM)
            labels = batch["category_idx"].to(device)   # (B,)

            # Forward: anomaly
            anom_encoded = fusion_encoder(anom_feat)     # (B, 512)
            logits = crime_head(anom_encoded)            # (B, NUM_CLASSES)

            # CE loss（分類）
            ce = ce_loss_fn(logits, labels)

            # MIL ranking loss（異常 vs 正常）
            anom_scores = escalation_head(anom_encoded).squeeze(1)  # (B,)
            norm_encoded = fusion_encoder(norm_feat)
            norm_scores = escalation_head(norm_encoded).squeeze(1)  # (B,)
            mil = _mil_ranking_loss(anom_scores, norm_scores)

            # 總損失
            loss = ce + lambda_mil * mil

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_ce_loss += ce.item()
            epoch_mil_loss += mil.item()
            n_batches += 1

        scheduler.step()

        avg_ce = epoch_ce_loss / max(n_batches, 1)
        avg_mil = epoch_mil_loss / max(n_batches, 1)
        avg_total = avg_ce + lambda_mil * avg_mil

        log_entry = {
            "epoch": epoch + 1,
            "ce_loss": round(avg_ce, 4),
            "mil_loss": round(avg_mil, 4),
            "total_loss": round(avg_total, 4),
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(log_entry)
        logger.info(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"CE={avg_ce:.4f} MIL={avg_mil:.4f} Total={avg_total:.4f} | "
            f"lr={log_entry['lr']:.2e}"
        )

        # 存最佳權重
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(fusion_encoder.state_dict(), WEIGHTS_DIR / "fusion_encoder.pt")
            torch.save(crime_head.state_dict(), WEIGHTS_DIR / "crime_head.pt")
            torch.save(escalation_head.state_dict(), WEIGHTS_DIR / "escalation_head.pt")

    # 存訓練 log
    with open(WEIGHTS_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"訓練完成！最佳 loss={best_loss:.4f}")
    logger.info(f"權重 → {WEIGHTS_DIR}")


def _mil_ranking_loss(
    scores_a: torch.Tensor,
    scores_n: torch.Tensor,
    mu1: float = 8e-5,
    mu2: float = 8e-5,
) -> torch.Tensor:
    """MIL ranking loss (Elmetwally et al. 2025)。"""
    l1 = torch.clamp(1.0 - scores_a.max() + scores_n.max(), min=0.0)
    l2 = torch.clamp(1.0 - scores_a.max() + scores_a.min(), min=0.0)
    smoothness = ((scores_a[1:] - scores_a[:-1]) ** 2).sum()
    sparsity = scores_a.sum()
    return l1 + l2 + mu1 * smoothness + mu2 * sparsity


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIL Head 訓練")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_mil", type=float, default=0.5)
    parser.add_argument("--split", default="Train")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_mil=args.lambda_mil,
        split=args.split,
    )
