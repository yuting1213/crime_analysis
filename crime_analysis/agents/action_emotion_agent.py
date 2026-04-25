"""
ActionEmotionAgent - 行為情緒融合代理
合併原 ActionAgent 與 TimeEmotionAgent，採早期融合架構。

特徵提取（各 backbone 維持不變）：
  I3D-ResNet50（以 R3D-18 替代，計算成本考量）→ 512D   行為特徵（動作識別）
  ViT-Base [CLS] token                          → 768D   視覺語意特徵
  MediaPipe Pose 33 keypoints × 3               →  99D   姿態特徵
  DeepFace 情緒機率                              →   7D   情緒特徵
  ─────────────────────────────────────────────────────
  合計                                          1386D
  → InputProjection（Linear + LayerNorm + ReLU）→ 512D
  → FusionEncoder（4層 TransformerEncoder，8 heads）→ 512D
  → crime_head（Linear→13類）+ escalation_head（Linear→1）

Backbone 說明：
  論文方法論引用：I3D-ResNet50（Elmetwally et al. 2025 原始設計）
  程式碼實作：    R3D-18 替代（torchvision 原生支援，無需額外安裝）

  R3D-18 替代理由（三點）：
  ① 輸出維度相同（512D），不影響後續 1386D 融合計算
  ② 監視器低解析度影像（通常 ≤ 480p）上，兩者特徵表達差異有限
  ③ 本地端搭配 Qwen3-7B 的資源限制下，R3D-18 計算成本更低
  → 若消融實驗顯示差異不顯著，論文可寫：
    「以 R3D-18 作為計算效率考量的替代實作，在本資料集上
     與 I3D-ResNet50 的性能差異在可接受範圍內」

  MIL Ranking Loss 的訓練邏輯（32 snippets × 16 frames）依 I3D 設計不變。

參考文獻：
  Elmetwally et al. (2025) Deep learning based anomaly detection in real-time video
  Kilic & Tuceryan (2024) Crime Detection from Pre-crime Video Analysis
    with Augmented Pose and Emotion Information
  Zou et al. (2025) Unlocking Vision-Language Models for Video Anomaly Detection
    via Fine-Grained Prompting
"""
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base_agent import AgentReport, BaseAgent
from config import cfg

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── RTX 5090 Blackwell 優化 ──────────────────────────────────
_BF16_DTYPE = torch.bfloat16  # Blackwell 原生 BF16 吞吐量 >> FP32

# ── 特徵維度常數 ───────────────────────────────────────────
R3D_DIM     = 512   # R3D-18 輸出維度
VIT_DIM     = 768   # ViT-Base CLS token 維度
POSE_DIM    = 99    # MediaPipe 33 keypoints × 3
EMOTION_DIM = 7     # DeepFace 情緒機率（7 種）
FUSION_DIM  = R3D_DIM + VIT_DIM + POSE_DIM + EMOTION_DIM  # 1386

# ── 時序常數 ───────────────────────────────────────────────
NUM_SNIPPETS    = 32   # 每支影片切分的片段數（Elmetwally et al.）
FRAMES_PER_CLIP = 16   # 每個 snippet 的幀數
CROP_SIZE       = 112  # R3D-18 空間輸入尺寸
RESIZE_H        = 128
RESIZE_W        = 171

# ── UCF-Crime 類別 ─────────────────────────────────────────
UCF_CATEGORIES: List[str] = [
    "Assault", "Robbery", "Stealing", "Shoplifting", "Burglary",
    "Fighting", "Arson", "Explosion", "RoadAccidents", "Vandalism",
    "Abuse", "Shooting", "Arrest",
]
NUM_CLASSES = len(UCF_CATEGORIES)  # 13

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 閾值：憤怒 + 恐懼 > 此值視為情緒升溫
ESCALATION_CALM_THRESHOLD = 0.35

# ── 犯罪群組與嚴重度 ───────────────────────────────────────
CRIME_GROUP: Dict[str, str] = {
    "Fighting":      "暴力 / 人身傷害",
    "Assault":       "暴力 / 人身傷害",
    "Shooting":      "暴力 / 人身傷害",
    "Robbery":       "暴力 / 人身傷害",
    "Abuse":         "暴力 / 人身傷害",
    "Arrest":        "暴力 / 人身傷害",
    "Stealing":      "財產犯罪",
    "Shoplifting":   "財產犯罪",
    "Burglary":      "財產犯罪",
    "Vandalism":     "財產犯罪",
    "Arson":         "公共安全",
    "Explosion":     "公共安全",
    "RoadAccidents": "公共安全",
}

CRIME_SEVERITY: Dict[str, str] = {
    "Shooting":      "HIGH",
    "Robbery":       "HIGH",
    "Assault":       "HIGH",
    "Fighting":      "MEDIUM",
    "Burglary":      "MEDIUM",
    "Arson":         "MEDIUM",
    "Explosion":     "MEDIUM",
    "Abuse":         "MEDIUM",
    "Arrest":        "MEDIUM",
    "Vandalism":     "MEDIUM",
    "Shoplifting":   "LOW",
    "Stealing":      "LOW",
    "RoadAccidents": "LOW",
}

# ── ASK-HINT Q* 問題群組（Zou et al. 2025）────────────────
ASK_HINT_Q_STAR: Dict[str, List[str]] = {
    "violence": [
        "Do you see people confronting, attacking, or restraining each other?",
        "Is there evidence of weapons, force, or law enforcement presence?",
    ],
    "property": [
        "Do you see someone unlawfully taking, concealing, or destroying property?",
        "Do you see forced entry, vandalism, or deliberate fire on property?",
    ],
    "public_safety": [
        "Do you see a sudden blast, smoke, or debris in the scene?",
        "Do you see vehicles colliding, losing control, or hitting people?",
    ],
}

ASK_HINT_GROUP_MAP: Dict[str, str] = {
    "Fighting":      "violence",
    "Assault":       "violence",
    "Shooting":      "violence",
    "Robbery":       "violence",
    "Abuse":         "violence",
    "Arrest":        "violence",
    "Stealing":      "property",
    "Shoplifting":   "property",
    "Burglary":      "property",
    "Vandalism":     "property",
    "Arson":         "public_safety",
    "Explosion":     "public_safety",
    "RoadAccidents": "public_safety",
}

# 因果鏈模板（依犯罪類型）
CAUSAL_CHAIN_TEMPLATES: Dict[str, str] = {
    "Fighting":      "情緒對立 → 肢體接觸 → 互毆衝突",
    "Assault":       "情緒激動 → 主動攻擊 → 傷害行為",
    "Shooting":      "威脅態勢 → 武器顯現 → 開槍射擊",
    "Robbery":       "接近目標 → 武力威脅 → 強奪財物",
    "Abuse":         "權力壓制 → 持續施害 → 受害者無力反抗",
    "Arrest":        "追捕行動 → 強制接觸 → 執法壓制",
    "Stealing":      "環境偵察 → 趁隙竊取 → 隱匿財物",
    "Shoplifting":   "進入場所 → 藏匿商品 → 未結帳離開",
    "Burglary":      "非法入侵 → 搜索財物 → 竊取離去",
    "Vandalism":     "接近目標物 → 蓄意破壞 → 損毀現場",
    "Arson":         "準備引火材料 → 點火燃燒 → 火勢蔓延",
    "Explosion":     "可疑物品放置 → 引爆裝置 → 爆炸衝擊",
    "RoadAccidents": "車輛失控 → 碰撞發生 → 事故現場混亂",
}


# ══════════════════════════════════════════════════════════
#  FusionEncoder：1386D → 512D Transformer 編碼器
# ══════════════════════════════════════════════════════════

class FusionEncoder(nn.Module):
    """
    早期融合編碼器
    InputProjection: Linear(1386, 512) → LayerNorm → ReLU
    4-layer TransformerEncoder (d_model=512, nhead=8, dim_feedforward=2048)
    """

    def __init__(
        self,
        input_dim: int = FUSION_DIM,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        Returns: (B, d_model)
        projected vector treated as sequence-of-1 for the Transformer
        """
        projected = self.input_proj(x)          # (B, d_model)
        seq = projected.unsqueeze(1)            # (B, 1, d_model)  — sequence length = 1
        encoded = self.encoder(seq)             # (B, 1, d_model)
        pooled = self.norm(encoded.squeeze(1))  # (B, d_model)
        return pooled


# ══════════════════════════════════════════════════════════
#  ActionEmotionAgent
# ══════════════════════════════════════════════════════════

class ActionEmotionAgent(BaseAgent):
    """
    行為情緒融合代理
    合併 ActionAgent 與 TimeEmotionAgent，採用早期融合架構
    輸出：crime_head（13類）+ escalation_head（升溫分數）
    """

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(
            name="行為情緒分析專家",
            model_name=model_name or cfg.model.base_model,
        )
        self.device = getattr(cfg.model, "device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("CUDA 不可用，切換至 CPU")

        # 懶載旗標
        self._models_loaded: bool = False
        self._r3d = None
        self._vit_model = None
        self._vit_extractor = None
        self._fusion_encoder: Optional[FusionEncoder] = None
        self._crime_head: Optional[nn.Module] = None
        self._escalation_head: Optional[nn.Module] = None

        # MediaPipe / DeepFace（可選）
        self._mp_pose = None
        self._deepface_available: bool = False
        self._feat_scaler: Optional[dict] = None  # StandardScaler from train_mil

        logger.info(f"ActionEmotionAgent 初始化，裝置：{self.device}")

    # ── 公開介面 ────────────────────────────────────────────

    def analyze(self, frames: List[np.ndarray], video_metadata: Dict[str, Any]) -> AgentReport:
        """
        主分析流程：
        1. 懶載模型
        2. 提取四路特徵並拼接為 1386D 向量
        3. FusionEncoder → 512D
        4. crime_head → 犯罪類別
        5. escalation_head → 升溫分數
        6. 建立因果鏈與報告
        """
        self._load_models()

        # 過濾 None 幀，確保有效幀存在
        valid_frames = [f for f in frames if f is not None and hasattr(f, 'shape')]
        if not valid_frames:
            return self._empty_report("無有效影像幀，無法分析")
        if len(valid_frames) < len(frames):
            logger.info(f"有效幀：{len(valid_frames)}/{len(frames)}")
        frames = valid_frames

        fps = float(video_metadata.get("fps", 25.0))

        # ── UCA 時序標註：優先從犯罪時段抽幀 ────────────
        uca_segments = video_metadata.get("uca_segments", [])
        if uca_segments:
            frames = self._targeted_sample(frames, uca_segments, len(frames))

        # ── 特徵提取 ─────────────────────────────────────
        # R3D / ViT 在 GPU 上推理（不用 autocast，避免輸出 BF16 與 numpy 衝突）
        r3d_feat    = self._extract_r3d_features(frames)    # (512,) float32
        vit_feat    = self._extract_vit_features(frames)    # (768,) float32
        pose_feat   = self._extract_pose_features(frames)   # (99,)  CPU-bound
        emo_feat    = self._extract_emotion_features(frames) # (7,)   CPU-bound

        fused = np.concatenate([r3d_feat, vit_feat, pose_feat, emo_feat])  # (1386,)

        # StandardScaler（與 train_mil.py 訓練時一致）
        if self._feat_scaler is not None:
            fused = (fused - self._feat_scaler["mean"]) / self._feat_scaler["std"]

        # ── 融合編碼（RTX 5090: BF16 autocast 只在 encoder forward 用）─────
        _use_amp = (self.device == "cuda" and cfg.model.torch_dtype == "bfloat16")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=_BF16_DTYPE, enabled=_use_amp):
            x = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 1386)
            fusion_output = self._fusion_encoder(x)  # (1, 512)

        # ── 分類（MIL Head 初篩，VLM 覆核由 Planner Step 2b 負責）──
        crime_type, confidence, mil_top3 = self._classify_crime(fusion_output)
        mil_crime_type, mil_confidence = crime_type, confidence

        # ── snippet 分數（使用 R3D snippets 的異常分數）──
        snippet_scores = self._compute_snippet_anomaly_scores(frames)

        # ── 升溫分析 ─────────────────────────────────────
        escalation_score, escalation_start_frame = self._compute_escalation(
            fusion_output, snippet_scores
        )

        # ── 因果鏈 / 事前事後指標 ─────────────────────────
        causal_chain = self._build_causal_chain(crime_type, escalation_score)
        evidence_frames = self._extract_key_frames(frames)
        pre_crime_indicators  = self._build_pre_crime_indicators(crime_type, emo_feat)
        post_crime_indicators = self._build_post_crime_indicators(crime_type, escalation_score)
        ask_hint_group = self._get_ask_hint_group(crime_type)
        rationale = self._build_rationale(
            crime_type, evidence_frames, escalation_score, causal_chain
        )

        report = AgentReport(
            agent_name=self.name,
            crime_category=crime_type,
            confidence=confidence,
            evidence=[
                {"type": "r3d_feature_norm", "value": float(np.linalg.norm(r3d_feat))},
                {"type": "vit_feature_norm", "value": float(np.linalg.norm(vit_feat))},
                {"type": "dominant_emotion",  "value": EMOTION_LABELS[int(np.argmax(emo_feat))]},
                {"type": "escalation_score",  "value": round(escalation_score, 3)},
                {"type": "causal_chain",       "value": causal_chain},
            ],
            reasoning=rationale,
            frame_references=evidence_frames,
            metadata={
                "crime_type_severity":    CRIME_SEVERITY.get(crime_type, "MEDIUM"),
                "crime_group":            CRIME_GROUP.get(crime_type, ""),
                "rationale":              rationale,
                "escalation_score":       escalation_score,
                "escalation_start_frame": escalation_start_frame,
                "pre_crime_indicators":   pre_crime_indicators,
                "post_crime_indicators":  post_crime_indicators,
                "causal_chain":           causal_chain,
                "ask_hint_group":         ask_hint_group,
                "snippet_scores":         snippet_scores,
                "mil_crime_type":         mil_crime_type,
                "mil_confidence":         mil_confidence,
                "mil_top3":               mil_top3,
                "vlm_used":              crime_type != mil_crime_type,
            },
        )
        self._position = report
        return report

    def refine(self, other_reports: List[AgentReport]) -> AgentReport:
        """
        共識收斂：返回當前立場（細粒度精化由 Planner 統一協調）
        """
        if self._position is None:
            raise RuntimeError("analyze() 必須先於 refine() 執行")
        return self._position

    # ── 懶載模型 ────────────────────────────────────────────

    def _load_models(self) -> None:
        if self._models_loaded:
            return

        logger.info("開始載入 ActionEmotionAgent 模型…")

        # ── RTX 5090: 啟用 cuDNN benchmark 自動調優卷積核 ──
        if cfg.model.cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark 已啟用（RTX 5090 卷積自動調優）")

        # R3D-18
        try:
            import torchvision.models.video as vm
            self._r3d = vm.r3d_18(weights="DEFAULT")
            self._r3d.fc = nn.Identity()  # 移除分類頭，取 512D 特徵
            self._r3d = self._r3d.to(self.device).eval()
            # RTX 5090: torch.compile 加速 R3D-18 推理
            if cfg.model.compile_models:
                try:
                    self._r3d = torch.compile(self._r3d)
                    logger.info("R3D-18 載入完成（torch.compile 已啟用）")
                except Exception as e:
                    logger.warning(f"R3D-18 torch.compile 失敗，使用 eager mode：{e}")
                    logger.info("R3D-18 載入完成")
            else:
                logger.info("R3D-18 載入完成")
        except Exception as e:
            logger.warning(f"R3D-18 載入失敗，將使用零向量：{e}")
            self._r3d = None

        # ViT-Base
        try:
            from transformers import ViTModel
            try:
                from transformers import ViTImageProcessor as _Proc
            except ImportError:
                from transformers import ViTFeatureExtractor as _Proc
            vit_name = "google/vit-base-patch16-224"
            self._vit_extractor = _Proc.from_pretrained(vit_name)
            self._vit_model = ViTModel.from_pretrained(vit_name).to(self.device).eval()
            # RTX 5090: torch.compile 加速 ViT 推理
            if cfg.model.compile_models:
                try:
                    self._vit_model = torch.compile(self._vit_model)
                    logger.info("ViT-Base 載入完成（torch.compile 已啟用）")
                except Exception as e:
                    logger.warning(f"ViT-Base torch.compile 失敗，使用 eager mode：{e}")
                    logger.info("ViT-Base 載入完成")
            else:
                logger.info("ViT-Base 載入完成")
        except Exception as e:
            logger.warning(f"ViT-Base 載入失敗，將使用零向量：{e}")
            self._vit_model = None
            self._vit_extractor = None

        # MediaPipe Pose
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
            logger.info("MediaPipe Pose 初始化完成")
        except Exception as e:
            logger.warning(f"MediaPipe 不可用，姿態特徵將使用零向量：{e}")
            self._mp_pose = None

        # DeepFace
        try:
            import deepface  # noqa: F401
            self._deepface_available = True
            logger.info("DeepFace 可用")
        except Exception as e:
            logger.warning(f"DeepFace 不可用，情緒特徵將使用零向量：{e}")
            self._deepface_available = False

        # FusionEncoder
        self._fusion_encoder = FusionEncoder(
            input_dim=FUSION_DIM,
            d_model=512,
            nhead=8,
            num_layers=4,
        ).to(self.device).eval()

        # crime_head：Linear(512, 13) + Softmax
        self._crime_head = nn.Sequential(
            nn.Linear(512, NUM_CLASSES),
            nn.Softmax(dim=-1),
        ).to(self.device).eval()

        # escalation_head：Linear(512, 1) + Sigmoid
        self._escalation_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        ).to(self.device).eval()

        # 載入訓練好的權重（如果有的話）
        weights_dir = Path("./outputs/mil_weights")
        if weights_dir.exists():
            self._load_trained_weights(weights_dir)

        # RTX 5090: torch.compile 加速 FusionEncoder + heads
        if cfg.model.compile_models:
            try:
                self._fusion_encoder = torch.compile(self._fusion_encoder)
                self._crime_head = torch.compile(self._crime_head)
                self._escalation_head = torch.compile(self._escalation_head)
                logger.info("FusionEncoder + heads torch.compile 已啟用")
            except Exception as e:
                logger.warning(f"FusionEncoder torch.compile 失敗：{e}")

        self._models_loaded = True

        # 特徵可用性報告
        avail = {
            "R3D-18": self._r3d is not None,
            "ViT-Base": self._vit_model is not None,
            "MediaPipe": self._mp_pose is not None,
            "DeepFace": self._deepface_available,
        }
        active = sum(avail.values())
        dim_active = (512 if avail["R3D-18"] else 0) + (768 if avail["ViT-Base"] else 0) + \
                     (99 if avail["MediaPipe"] else 0) + (7 if avail["DeepFace"] else 0)
        if active < 4:
            missing = [k for k, v in avail.items() if not v]
            logger.warning(
                f"特徵降級模式：{active}/4 模態可用（{dim_active}/{FUSION_DIM}D）"
                f"，缺少 {', '.join(missing)}"
            )
        else:
            logger.info("所有模型初始化完成（4/4 模態）")

    def _load_trained_weights(self, weights_dir: Path):
        """載入 train_mil.py 訓練好的權重。"""
        loaded = []
        for name, module in [
            ("fusion_encoder", self._fusion_encoder),
            ("crime_head", self._crime_head),
            ("escalation_head", self._escalation_head),
        ]:
            path = weights_dir / f"{name}.pt"
            if path.exists() and module is not None:
                try:
                    state = torch.load(path, map_location=self.device, weights_only=True)
                    module.load_state_dict(state, strict=False)
                    loaded.append(name)
                except Exception as e:
                    logger.warning(f"權重載入失敗 {name}：{e}")
        if loaded:
            logger.info(f"已載入訓練權重：{', '.join(loaded)}")

        # 載入 StandardScaler（train_mil.py 訓練時產生）
        scaler_path = weights_dir / "feature_scaler.npz"
        if scaler_path.exists():
            data = np.load(scaler_path)
            self._feat_scaler = {"mean": data["mean"], "std": data["std"]}
            logger.info("已載入 feature scaler")

    # ── 特徵提取 ────────────────────────────────────────────

    def _extract_r3d_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        R3D-18 特徵提取：切成 32 snippets × 16 幀，取平均 512D 向量
        """
        if self._r3d is None or len(frames) < 2:
            return np.zeros(R3D_DIM, dtype=np.float32)

        snippets = self._split_into_snippets(frames, NUM_SNIPPETS, FRAMES_PER_CLIP)
        snippet_feats = []

        with torch.no_grad():
            for snippet in snippets:
                clip = self._preprocess_clip(snippet)  # (1, 3, T, H, W)
                if clip is None:
                    continue
                feat = self._r3d(clip.to(self.device))  # (1, 512)
                snippet_feats.append(feat.cpu().numpy()[0])

        if not snippet_feats:
            return np.zeros(R3D_DIM, dtype=np.float32)

        return np.mean(snippet_feats, axis=0).astype(np.float32)

    def _extract_vit_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        ViT-Base CLS token 特徵：從關鍵幀取樣，平均 768D
        """
        if self._vit_model is None or self._vit_extractor is None:
            return np.zeros(VIT_DIM, dtype=np.float32)

        keyframe_indices = self._extract_key_frames(frames)
        feats = []

        with torch.no_grad():
            for idx in keyframe_indices:
                frame_bgr = frames[idx]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                inputs = self._vit_extractor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._vit_model(**inputs)
                cls_feat = outputs.last_hidden_state[:, 0, :]  # (1, 768)
                feats.append(cls_feat.cpu().numpy()[0])

        if not feats:
            return np.zeros(VIT_DIM, dtype=np.float32)

        return np.mean(feats, axis=0).astype(np.float32)

    def _extract_pose_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        MediaPipe Pose：33 keypoints × 3(x,y,z) = 99D，平均所有幀
        若 MediaPipe 不可用返回零向量
        """
        if self._mp_pose is None:
            return np.zeros(POSE_DIM, dtype=np.float32)

        keyframe_indices = self._extract_key_frames(frames)
        pose_feats = []

        for idx in keyframe_indices:
            frame_bgr = frames[idx]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                result = self._mp_pose.process(frame_rgb)
                if result.pose_landmarks:
                    kpts = np.array(
                        [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                        dtype=np.float32,
                    ).flatten()  # (99,)
                    pose_feats.append(kpts)
            except Exception:
                pass

        if not pose_feats:
            return np.zeros(POSE_DIM, dtype=np.float32)

        return np.mean(pose_feats, axis=0).astype(np.float32)

    def _extract_emotion_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        DeepFace 情緒機率：7D，平均所有關鍵幀
        若 DeepFace 不可用返回零向量（neutral=1.0）
        """
        if not self._deepface_available:
            neutral_vec = np.zeros(EMOTION_DIM, dtype=np.float32)
            neutral_vec[-1] = 1.0  # neutral
            return neutral_vec

        from deepface import DeepFace  # noqa: PLC0415

        keyframe_indices = self._extract_key_frames(frames)
        emo_feats = []

        for idx in keyframe_indices:
            frame_bgr = frames[idx]
            try:
                result = DeepFace.analyze(
                    frame_bgr,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
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
            neutral_vec = np.zeros(EMOTION_DIM, dtype=np.float32)
            neutral_vec[-1] = 1.0
            return neutral_vec

        avg = np.mean(emo_feats, axis=0).astype(np.float32)
        total = avg.sum()
        if total > 0:
            avg /= total
        return avg

    # ── 分類與升溫計算 ─────────────────────────────────────
    # VLM 分類已移至 Planner Step 2b（Qwen3-VL 統一模型）

    def _classify_crime(
        self, fusion_output: torch.Tensor
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        crime_head: (1, 512) → softmax → argmax → (crime_type, confidence, top3)

        top3 是 [(cat, prob)] × 3，依機率遞減，供 RAG priming 使用。
        """
        with torch.no_grad():
            probs = self._crime_head(fusion_output)  # (1, 13)
            probs_np = probs.cpu().numpy()[0]

        top_idx = int(np.argmax(probs_np))
        crime_type = UCF_CATEGORIES[top_idx]
        confidence = float(probs_np[top_idx])

        # top-3（含 top-1 本身）供 Pre-classification RAG Priming 使用
        order = np.argsort(-probs_np)[:3]
        top3 = [(UCF_CATEGORIES[int(i)], round(float(probs_np[int(i)]), 4)) for i in order]

        return crime_type, round(confidence, 4), top3

    def _compute_snippet_anomaly_scores(self, frames: List[np.ndarray]) -> List[float]:
        """
        使用 R3D-18 特徵的 L2 norm 作為每個 snippet 的異常分數代理
        """
        if self._r3d is None or len(frames) < 2:
            return [0.5] * NUM_SNIPPETS

        snippets = self._split_into_snippets(frames, NUM_SNIPPETS, FRAMES_PER_CLIP)
        scores = []

        with torch.no_grad():
            for snippet in snippets:
                clip = self._preprocess_clip(snippet)
                if clip is None:
                    scores.append(0.0)
                    continue
                feat = self._r3d(clip.to(self.device))  # (1, 512)
                score = float(feat.norm(p=2).item()) / 50.0  # normalize heuristically
                scores.append(min(score, 1.0))

        return scores

    def _compute_escalation(
        self, fusion_output: torch.Tensor, snippet_scores: List[float]
    ) -> Tuple[float, Optional[int]]:
        """
        escalation_head: (1, 512) → sigmoid → score
        escalation_start_frame: snippet_scores で閾値 > 0.5 の最初の snippet
        """
        with torch.no_grad():
            score_tensor = self._escalation_head(fusion_output)  # (1, 1)
            escalation_score = float(score_tensor.item())

        # 找到 snippet_scores 首次超過 0.5 的 snippet
        escalation_start_frame: Optional[int] = None
        total_frames = NUM_SNIPPETS * FRAMES_PER_CLIP
        frames_per_snippet = max(1, total_frames // len(snippet_scores)) if snippet_scores else FRAMES_PER_CLIP

        for i, s in enumerate(snippet_scores):
            snippet_threshold = 0.5  # 此值为启发式设定，可从 config.inference.escalation_high_threshold 扩展
            if s > snippet_threshold:
                escalation_start_frame = i * frames_per_snippet
                break

        return round(escalation_score, 4), escalation_start_frame

    # ── 輔助方法 ────────────────────────────────────────────

    def _build_causal_chain(self, crime_type: str, escalation_score: float) -> str:
        """根據犯罪類型返回三段式因果鏈描述"""
        base = CAUSAL_CHAIN_TEMPLATES.get(crime_type, "異常行為 → 犯罪實施 → 事後逃逸")
        high_threshold = cfg.inference.escalation_high_threshold  # 0.7
        mid_threshold = cfg.inference.escalation_calm_threshold + 0.35  # ~0.7
        if escalation_score > high_threshold:
            return f"[高度升溫] {base}"
        elif escalation_score > mid_threshold:
            return f"[中度升溫] {base}"
        else:
            return f"[低度升溫] {base}"

    def _get_ask_hint_group(self, crime_type: str) -> str:
        """返回 ASK-HINT Q* 群組名稱"""
        return ASK_HINT_GROUP_MAP.get(crime_type, "violence")

    def _build_rationale(
        self,
        crime_type: str,
        evidence_frames: List[int],
        escalation_score: float,
        causal_chain: str,
    ) -> str:
        severity = CRIME_SEVERITY.get(crime_type, "MEDIUM")
        group = CRIME_GROUP.get(crime_type, "")
        ask_group = self._get_ask_hint_group(crime_type)
        hints = ASK_HINT_Q_STAR.get(ask_group, [])
        hint_text = " ".join(hints)

        frame_ref = f"幀 {evidence_frames[0]}–{evidence_frames[-1]}" if evidence_frames else "N/A"

        return (
            f"【行為情緒分析】偵測到犯罪行為：{crime_type}（{group}）。"
            f"嚴重程度：{severity}。"
            f"情緒升溫分數：{escalation_score:.2f}。"
            f"因果鏈：{causal_chain}。"
            f"關鍵幀範圍：{frame_ref}。"
            f"語義問題提示：{hint_text}"
        )

    def _build_pre_crime_indicators(
        self, crime_type: str, emo_feat: np.ndarray
    ) -> List[str]:
        """建立事前指標清單"""
        indicators = []
        dominant_idx = int(np.argmax(emo_feat))
        dominant_emo = EMOTION_LABELS[dominant_idx]

        if dominant_emo in {"angry", "fear", "disgust"}:
            indicators.append(f"偵測到事前情緒：{dominant_emo}（可能的衝突前兆）")

        group = ASK_HINT_GROUP_MAP.get(crime_type, "violence")
        if group == "violence":
            indicators.append("人員間肢體距離縮短，存在對峙跡象")
        elif group == "property":
            indicators.append("嫌疑人有環境偵察行為，鎖定目標物")
        elif group == "public_safety":
            indicators.append("場景出現可疑物品或車輛異常移動")

        return indicators

    def _build_post_crime_indicators(
        self, crime_type: str, escalation_score: float
    ) -> List[str]:
        """建立事後指標清單"""
        indicators = []
        if escalation_score > 0.6:
            indicators.append("事後現場人員快速散離")
        if crime_type in {"Assault", "Fighting", "Shooting", "Robbery"}:
            indicators.append("受害者出現防禦性姿態或倒地行為")
        if crime_type in {"Stealing", "Shoplifting", "Burglary"}:
            indicators.append("嫌疑人快速離開現場，攜帶隱匿物品")
        if crime_type in {"Arson", "Explosion"}:
            indicators.append("現場出現煙霧或人員逃散")
        if not indicators:
            indicators.append("事後行為正常，需進一步觀察")
        return indicators

    def _targeted_sample(
        self,
        frames: List[np.ndarray],
        uca_segments: List[Dict],
        total_frames: int,
    ) -> List[np.ndarray]:
        """
        利用 UCA 時序標註做 targeted sampling。
        策略：70% 幀從犯罪時段抽取，30% 從全影片均勻抽取（保留背景脈絡）。
        """
        n_target = len(frames)
        n_event = int(n_target * 0.7)
        n_context = n_target - n_event

        # 收集犯罪時段的幀索引
        event_frame_indices = set()
        for seg in uca_segments:
            start_f = seg.get("start_frame", 0)
            end_f = seg.get("end_frame", total_frames)
            for i in range(start_f, min(end_f + 1, len(frames))):
                event_frame_indices.add(i)

        if not event_frame_indices:
            return frames  # 無標註資訊，不改變

        # 從犯罪時段均勻取 n_event 幀
        event_indices = sorted(event_frame_indices)
        if len(event_indices) >= n_event:
            step = max(1, len(event_indices) // n_event)
            sampled_event = [frames[event_indices[i]] for i in range(0, len(event_indices), step)][:n_event]
        else:
            sampled_event = [frames[i] for i in event_indices]

        # 從全影片均勻取 n_context 幀（背景脈絡）
        context_step = max(1, len(frames) // n_context) if n_context > 0 else len(frames)
        sampled_context = frames[::context_step][:n_context]

        result = sampled_event + sampled_context
        # 補足到原始數量
        while len(result) < n_target:
            result.append(result[-1])

        return result[:n_target]

    def _extract_key_frames(
        self, frames: List[np.ndarray], max_keys: int = 8
    ) -> List[int]:
        """均勻取樣關鍵幀索引（ViT/Pose/Emotion 共用）"""
        n = len(frames)
        if n == 0:
            return []
        num_keys = min(max_keys, n)
        indices = [int(round(i * (n - 1) / (num_keys - 1))) for i in range(num_keys)] \
            if num_keys > 1 else [0]
        return sorted(set(indices))

    # ── 內部工具 ────────────────────────────────────────────

    def _split_into_snippets(
        self, frames: List[np.ndarray], num_snippets: int, frames_per_clip: int
    ) -> List[List[np.ndarray]]:
        """均勻切割影片為 num_snippets 個片段，每片段取 frames_per_clip 幀"""
        n = len(frames)
        if n < 2:
            return []

        snippets = []
        snippet_size = max(1, n // num_snippets)

        for i in range(num_snippets):
            start = i * snippet_size
            end = min(start + snippet_size, n)
            segment = frames[start:end]

            if len(segment) < frames_per_clip:
                # 重複最後一幀補足
                segment = segment + [segment[-1]] * (frames_per_clip - len(segment))

            # 均勻取 frames_per_clip 幀
            indices = np.linspace(0, len(segment) - 1, frames_per_clip, dtype=int)
            clip = [segment[j] for j in indices]
            snippets.append(clip)

        return snippets

    def _preprocess_clip(self, clip_frames: List[np.ndarray]) -> Optional[torch.Tensor]:
        """
        將一個 snippet（List[np.ndarray] BGR）預處理為 R3D-18 輸入
        輸出：(1, 3, T, CROP_SIZE, CROP_SIZE)，值域 [0, 1]
        """
        try:
            processed = []
            for frame in clip_frames:
                # Resize
                h, w = frame.shape[:2]
                scale = RESIZE_H / h
                new_w = int(w * scale)
                resized = cv2.resize(frame, (new_w, RESIZE_H))

                # Center crop
                cx = resized.shape[1] // 2
                cy = resized.shape[0] // 2
                x1 = max(0, cx - CROP_SIZE // 2)
                y1 = max(0, cy - CROP_SIZE // 2)
                cropped = resized[y1:y1 + CROP_SIZE, x1:x1 + CROP_SIZE]

                if cropped.shape[:2] != (CROP_SIZE, CROP_SIZE):
                    cropped = cv2.resize(cropped, (CROP_SIZE, CROP_SIZE))

                # BGR → RGB, normalize
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                processed.append(rgb)

            # Stack: (T, H, W, 3) → (3, T, H, W)
            clip_array = np.stack(processed, axis=0)         # (T, H, W, 3)
            clip_tensor = torch.from_numpy(clip_array).permute(3, 0, 1, 2)  # (3, T, H, W)
            return clip_tensor.unsqueeze(0)                  # (1, 3, T, H, W)
        except Exception as e:
            logger.debug(f"clip 預處理失敗：{e}")
            return None

    def _empty_report(self, reason: str) -> AgentReport:
        """無法分析時返回的空報告"""
        report = AgentReport(
            agent_name=self.name,
            crime_category="Unknown",
            confidence=0.0,
            evidence=[{"type": "error", "value": reason}],
            reasoning=f"分析失敗：{reason}",
            frame_references=[],
            metadata={
                "crime_type_severity": "LOW",
                "crime_group": "",
                "rationale": reason,
                "escalation_score": 0.0,
                "escalation_start_frame": None,
                "pre_crime_indicators": [],
                "post_crime_indicators": [],
                "causal_chain": "",
                "ask_hint_group": "violence",
                "snippet_scores": [],
            },
        )
        self._position = report
        return report
