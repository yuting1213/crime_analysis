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
  python -m scripts.train_mil --epochs 30 --lr 1e-4 --batch_size 32  # 5090: 可用 32-64

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
FEATURE_VERSION = "v4"  # v1=前16幀+4幀ViT, v2=均勻16幀+8幀ViT, v3=4/4模態, v4=去除R3D L2norm+StandardScaler


def extract_and_cache_features(
    split: str = "Train",
    force: bool = False,
) -> Dict[str, Path]:
    """
    預提取所有影片的 4 模態融合特徵並存為 .npy。
    R3D-18(512D) + ViT(768D) + MediaPipe Pose(99D) + DeepFace Emotion(7D) = 1386D
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

    # 初始化 MediaPipe Pose
    mp_pose = None
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1, min_detection_confidence=0.5,
        )
        logger.info("MediaPipe Pose 初始化完成")
    except Exception as e:
        logger.warning(f"MediaPipe 不可用，Pose 特徵將為零：{e}")

    # 初始化 DeepFace
    deepface_ok = False
    try:
        from deepface import DeepFace as _df
        deepface_ok = True
        logger.info("DeepFace 可用")
    except Exception as e:
        logger.warning(f"DeepFace 不可用，Emotion 特徵將為零：{e}")

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

        # 抽幀 + 提取 4 模態特徵
        try:
            frames = _load_video_frames(str(video_path))
            r3d_feat = _extract_r3d_feature(r3d, frames, device)
            vit_feat = _extract_vit_feature(vit_model, vit_proc, frames, device) if vit_model else np.zeros(768, dtype=np.float32)
            pose_feat = _extract_pose_feature(mp_pose, frames) if mp_pose else np.zeros(99, dtype=np.float32)
            emo_feat = _extract_emotion_feature(frames) if deepface_ok else np.zeros(7, dtype=np.float32)

            # 組裝 1386D = R3D(512) + ViT(768) + Pose(99) + Emotion(7)
            fusion_feat = np.concatenate([r3d_feat, vit_feat, pose_feat, emo_feat])
            assert fusion_feat.shape == (FUSION_DIM,), f"Expected {FUSION_DIM}D, got {fusion_feat.shape}"

            np.save(npy_path, fusion_feat)
            cache_map[video_id] = npy_path
            processed += 1

            if processed % 50 == 0:
                logger.info(f"  已提取 {processed} 部影片特徵（4/4 模態）")
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
    """從幀序列提取 R3D-18 512D 特徵。RTX 5090: BF16 autocast 加速。"""
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

    # RTX 5090: BF16 autocast 加速特徵提取
    _use_amp = (device == "cuda" and cfg.model.torch_dtype == "bfloat16")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp):
        feat = model(clip_tensor)  # (1, 512)
        feat = feat.float()  # 不做 L2 normalize，保留幅度資訊供分類器區分

    return feat.cpu().numpy()[0]


def _extract_vit_feature(
    model: nn.Module,
    processor,
    frames: List[np.ndarray],
    device: str,
) -> np.ndarray:
    """從關鍵幀提取 ViT CLS token 768D 特徵（取平均）。RTX 5090: BF16 autocast 加速。"""
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

    # RTX 5090: BF16 autocast 加速特徵提取
    _use_amp = (device == "cuda" and cfg.model.torch_dtype == "bfloat16")

    feats = []
    for frame in selected:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp):
            outputs = model(**inputs)
            cls_feat = outputs.last_hidden_state[:, 0, :].float()  # (1, 768)
            feats.append(cls_feat.cpu().numpy()[0])

    return np.mean(feats, axis=0).astype(np.float32)


def _extract_pose_feature(
    mp_pose,
    frames: List[np.ndarray],
    n_keyframes: int = 8,
) -> np.ndarray:
    """MediaPipe Pose: 33 keypoints × 3(x,y,z) = 99D，均勻取 keyframes 後平均。"""
    import cv2

    if mp_pose is None or len(frames) < 1:
        return np.zeros(99, dtype=np.float32)

    n = len(frames)
    if n >= n_keyframes:
        indices = np.linspace(0, n - 1, n_keyframes, dtype=int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames

    pose_feats = []
    for frame in selected:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_pose.process(rgb)
            if result.pose_landmarks:
                kpts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                ).flatten()  # (99,)
                pose_feats.append(kpts)
        except Exception:
            pass

    if not pose_feats:
        return np.zeros(99, dtype=np.float32)
    return np.mean(pose_feats, axis=0).astype(np.float32)


def _extract_emotion_feature(
    frames: List[np.ndarray],
    n_keyframes: int = 8,
) -> np.ndarray:
    """DeepFace Emotion: 7D 情緒機率向量，均勻取 keyframes 後平均。"""
    import cv2
    from deepface import DeepFace

    EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    if len(frames) < 1:
        neutral = np.zeros(7, dtype=np.float32)
        neutral[-1] = 1.0
        return neutral

    n = len(frames)
    if n >= n_keyframes:
        indices = np.linspace(0, n - 1, n_keyframes, dtype=int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames

    emo_feats = []
    for frame in selected:
        try:
            result = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False, silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emo_dict = result.get("emotion", {})
            vec = np.array(
                [emo_dict.get(label, 0.0) / 100.0 for label in EMOTION_LABELS],
                dtype=np.float32,
            )
            emo_feats.append(vec)
        except Exception:
            pass

    if not emo_feats:
        neutral = np.zeros(7, dtype=np.float32)
        neutral[-1] = 1.0
        return neutral

    avg = np.mean(emo_feats, axis=0).astype(np.float32)
    total = avg.sum()
    if total > 0:
        avg /= total
    return avg


# ── Dataset ──────────────────────────────────────────────────

class UCFCrimeDataset(Dataset):
    """
    UCF-Crime 訓練資料集。
    每個樣本：(feature_512d, category_idx, is_anomaly)
    """

    def __init__(
        self,
        cache_map: Dict[str, Path],
        feat_mean: Optional[np.ndarray] = None,
        feat_std: Optional[np.ndarray] = None,
        split: str = "Train",
    ):
        from scripts.pilot_experiment import video_id_to_category

        self.feat_mean = feat_mean
        self.feat_std = feat_std
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

        # 確保維度正確
        if len(anom_feat) < FUSION_DIM:
            padded = np.zeros(FUSION_DIM, dtype=np.float32)
            padded[:len(anom_feat)] = anom_feat
            anom_feat = padded
        if len(norm_feat) < FUSION_DIM:
            padded = np.zeros(FUSION_DIM, dtype=np.float32)
            padded[:len(norm_feat)] = norm_feat
            norm_feat = padded

        # StandardScaler: (x - mean) / std
        if self.feat_mean is not None and self.feat_std is not None:
            anom_feat = (anom_feat - self.feat_mean) / self.feat_std
            norm_feat = (norm_feat - self.feat_mean) / self.feat_std

        return {
            "anom_feat": torch.from_numpy(anom_feat),
            "norm_feat": torch.from_numpy(norm_feat),
            "category_idx": anom_sample["category_idx"],
            "video_id": anom_sample["video_id"],
        }


# ── 訓練迴圈 ─────────────────────────────────────────────────

def train(
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    lambda_mil: float = None,
    split: str = "Train",
):
    """
    訓練 FusionEncoder + crime_head + escalation_head。

    損失 = CE_loss（分類）+ λ × MIL_ranking_loss（異常偵測）

    所有超參數從 cfg.training 讀取，CLI 引數可覆蓋。
    RTX 5090 優化：BF16 autocast / cuDNN benchmark / torch.compile / pin_memory
    """
    # 從 config 讀取預設值，CLI 引數可覆蓋
    epochs = epochs or cfg.training.epochs
    lr = lr or cfg.training.learning_rate
    batch_size = batch_size or cfg.training.batch_size
    lambda_mil = lambda_mil or cfg.training.lambda_mil

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── RTX 5090: cuDNN benchmark ──
    if device == "cuda" and cfg.model.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark 已啟用")

    # Step 1: 預提取特徵
    logger.info("Step 1: 預提取 R3D-18 + ViT 特徵...")
    cache_map = extract_and_cache_features(split=split)
    if not cache_map:
        logger.error("無特徵可用")
        return

    # Step 2: StandardScaler — 消除模態間 scale 差異
    # R3D ~10-30, ViT ~20, Pose ~1-4, Emotion ~0.14 → 全部 zero-mean, unit-var
    logger.info("Step 2: 計算 StandardScaler（全特徵）...")
    all_feats = np.stack([np.load(p) for p in cache_map.values()])
    feat_mean = all_feats.mean(axis=0).astype(np.float32)
    feat_std = all_feats.std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-6] = 1.0  # 避免除零（常數維度）

    # 保存 scaler 供推理時使用
    scaler_path = WEIGHTS_DIR / "feature_scaler.npz"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(scaler_path, mean=feat_mean, std=feat_std)
    logger.info(f"  Scaler 儲存 → {scaler_path}")
    logger.info(
        f"  R3D norm: mean={np.linalg.norm(feat_mean[:512]):.2f}, "
        f"ViT norm: mean={np.linalg.norm(feat_mean[512:1280]):.2f}, "
        f"Pose norm: mean={np.linalg.norm(feat_mean[1280:1379]):.2f}"
    )

    # Step 3: 建立 Dataset
    dataset = UCFCrimeDataset(cache_map, split=split, feat_mean=feat_mean, feat_std=feat_std)
    if len(dataset) == 0:
        logger.error("Dataset 為空")
        return

    # RTX 5090: pin_memory 加速 Host→Device 傳輸, num_workers 並行載入
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device == "cuda" and cfg.training.pin_memory),
        drop_last=True,
    )

    # Step 3: 初始化模型（使用 cfg.training.dropout）
    dropout = cfg.training.dropout
    fusion_encoder = FusionEncoder(
        input_dim=FUSION_DIM, d_model=512, nhead=8, num_layers=4,
        dropout=dropout,
    ).to(device)
    crime_head = nn.Sequential(
        nn.Linear(512, NUM_CLASSES),
    ).to(device)  # 不加 Softmax，CE loss 自帶
    escalation_head = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid(),
    ).to(device)

    # RTX 5090: torch.compile 加速訓練（Blackwell inductor 後端）
    if cfg.model.compile_models and device == "cuda":
        try:
            fusion_encoder = torch.compile(fusion_encoder)
            crime_head = torch.compile(crime_head)
            escalation_head = torch.compile(escalation_head)
            logger.info("torch.compile 已啟用（訓練模式）")
        except Exception as e:
            logger.warning(f"torch.compile 失敗，使用 eager mode：{e}")

    # Optimizer
    params = (
        list(fusion_encoder.parameters())
        + list(crime_head.parameters())
        + list(escalation_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.training.weight_decay)
    # Scheduler: CosineAnnealing + Linear Warmup
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)

    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Class weights：反比類別數量，緩解不平衡（Explosion 7 vs RoadAccidents 106）
    cat_counts = {}
    for s in dataset.samples:
        if s["is_anomaly"]:
            idx = s["category_idx"]
            cat_counts[idx] = cat_counts.get(idx, 0) + 1
    if cat_counts:
        weights = torch.zeros(NUM_CLASSES, device=device)
        total_samples = sum(cat_counts.values())
        for idx, count in cat_counts.items():
            weights[idx] = total_samples / (NUM_CLASSES * count)
        logger.info(f"Class weights: {dict(zip(CRIME_CATEGORIES, weights.tolist()))}")
    else:
        weights = None

    ce_loss_fn = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=cfg.training.label_smoothing,
    )

    # RTX 5090: BF16 混合精度訓練
    use_amp = (device == "cuda" and cfg.training.mixed_precision)
    amp_dtype = torch.bfloat16  # Blackwell 原生 BF16（不需要 GradScaler）
    if use_amp:
        logger.info("BF16 混合精度訓練已啟用（RTX 5090 Blackwell 原生）")

    # Step 4: 訓練
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    training_log = []
    best_loss = float("inf")

    logger.info(
        f"開始訓練：{epochs} epochs, lr={lr}, batch={batch_size}, "
        f"λ_MIL={lambda_mil}, wd={cfg.training.weight_decay}, "
        f"dropout={dropout}, label_smooth={cfg.training.label_smoothing}, "
        f"warmup={warmup_steps}/{total_steps} steps"
    )

    for epoch in range(epochs):
        fusion_encoder.train()
        crime_head.train()
        escalation_head.train()

        epoch_ce_loss = 0.0
        epoch_mil_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            anom_feat = batch["anom_feat"].to(device, non_blocking=True)   # (B, FUSION_DIM)
            norm_feat = batch["norm_feat"].to(device, non_blocking=True)   # (B, FUSION_DIM)
            labels = batch["category_idx"].to(device, non_blocking=True)   # (B,)

            # RTX 5090: BF16 autocast（Blackwell 不需要 GradScaler）
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
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

            # BF16 不需要 GradScaler（Blackwell 原生支援 BF16 梯度）
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.training.gradient_clip_norm)
            optimizer.step()
            scheduler.step()  # per-step（配合 warmup + cosine schedule）

            epoch_ce_loss += ce.item()
            epoch_mil_loss += mil.item()
            n_batches += 1

        avg_ce = epoch_ce_loss / max(n_batches, 1)
        avg_mil = epoch_mil_loss / max(n_batches, 1)
        avg_total = avg_ce + lambda_mil * avg_mil
        current_lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch + 1,
            "ce_loss": round(avg_ce, 4),
            "mil_loss": round(avg_mil, 4),
            "total_loss": round(avg_total, 4),
            "lr": current_lr,
        }
        training_log.append(log_entry)
        logger.info(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"CE={avg_ce:.4f} MIL={avg_mil:.4f} Total={avg_total:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # 存最佳權重（torch.compile 包裝後需要取原始 state_dict）
        if avg_total < best_loss:
            best_loss = avg_total
            _save_state_dict(fusion_encoder, WEIGHTS_DIR / "fusion_encoder.pt")
            _save_state_dict(crime_head, WEIGHTS_DIR / "crime_head.pt")
            _save_state_dict(escalation_head, WEIGHTS_DIR / "escalation_head.pt")

    # 存訓練 log
    with open(WEIGHTS_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"訓練完成！最佳 loss={best_loss:.4f}")
    logger.info(f"權重 → {WEIGHTS_DIR}")

    # 訓練視覺化
    _plot_training_curves(training_log, WEIGHTS_DIR)


def _plot_training_curves(training_log: List[Dict], output_dir: Path):
    """訓練過程視覺化：Loss curves + Learning Rate schedule."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安裝，跳過訓練視覺化。pip install matplotlib")
        return

    epochs = [e["epoch"] for e in training_log]
    ce_losses = [e["ce_loss"] for e in training_log]
    mil_losses = [e["mil_loss"] for e in training_log]
    total_losses = [e["total_loss"] for e in training_log]
    lrs = [e["lr"] for e in training_log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MIL Head Training Dashboard", fontsize=14, fontweight="bold")

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, total_losses, "b-", linewidth=1.5, label="Total Loss")
    best_ep = epochs[total_losses.index(min(total_losses))]
    ax.axvline(best_ep, color="r", linestyle="--", alpha=0.5, label=f"Best @ epoch {best_ep}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss (CE + λ·MIL)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. CE vs MIL Loss
    ax = axes[0, 1]
    ax.plot(epochs, ce_losses, "g-", linewidth=1.5, label="CE Loss (classification)")
    ax.plot(epochs, mil_losses, "orange", linewidth=1.5, label="MIL Loss (ranking)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CE Loss vs MIL Ranking Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Learning Rate Schedule
    ax = axes[1, 0]
    ax.plot(epochs, lrs, "m-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (Warmup + Cosine)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax.grid(True, alpha=0.3)

    # 4. Loss Ratio (CE / Total)
    ax = axes[1, 1]
    ce_ratio = [c / max(t, 1e-8) for c, t in zip(ce_losses, total_losses)]
    mil_ratio = [1 - r for r in ce_ratio]
    ax.fill_between(epochs, 0, ce_ratio, alpha=0.4, label="CE portion", color="green")
    ax.fill_between(epochs, ce_ratio, 1, alpha=0.4, label="MIL portion", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ratio")
    ax.set_title("Loss Composition (CE vs MIL)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"訓練曲線圖 → {plot_path}")


def _save_state_dict(model: nn.Module, path: Path):
    """儲存 state_dict，相容 torch.compile 包裝的模型。"""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(m.state_dict(), path)


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
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg.training.epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override cfg.training.learning_rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override cfg.training.batch_size")
    parser.add_argument("--lambda_mil", type=float, default=None, help="Override cfg.training.lambda_mil")
    parser.add_argument("--split", default="Train")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_mil=args.lambda_mil,
        split=args.split,
    )
