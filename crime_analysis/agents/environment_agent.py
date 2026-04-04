"""
Environment Agent - 環境評估代理
基於：Aberkane & Boudihir (2019) Deep RL for anomaly detection in surveillance videos
架構：DQN + C3D features (以 R3D-18 替代)，32 temporal segments，Prioritized Experience Replay
"""
import logging
import random
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .base_agent import BaseAgent, AgentReport
from config import cfg

logger = logging.getLogger(__name__)

# ── 常數（對應 Aberkane & Boudihir 2019）─────────────────
NUM_SEGMENTS = 32          # 每支影片切分的時間片段數（論文設定）
C3D_DIM = 512              # R3D-18 avgpool 輸出維度（論文用 C3D FC6 = 4096D）
HIDDEN1 = 256              # 第一隱藏層（依比例縮放）
HIDDEN2 = 32               # 第二隱藏層（與論文相同）
GAMMA = 0.95               # 折扣因子（論文設定）
BATCH_SIZE = 32            # mini-batch 大小（論文用 500，推論階段縮小）
REPLAY_CAPACITY = 10_000   # Replay Buffer 容量
ANOMALY_THRESHOLD = 0.5    # DQN 輸出 > 0.5 → 異常片段

# 光線閾值（OpenCV LAB 色域 L 通道，範圍 0–255）
DARK_THRESHOLD = 50        # L < 50 → 過暗
BRIGHT_THRESHOLD = 200     # L > 200 → 過曝


# ── DQN 架構 ─────────────────────────────────────────────

class DQNAnomalyDetector(nn.Module):
    """
    Aberkane & Boudihir (2019) DQN 架構
    原始論文：C3D FC6 (4096D) → FC(512) → FC(32) → FC(1, Sigmoid)
    本實作：  R3D-18 (512D)   → FC(256) → FC(32) → FC(1, Sigmoid)
    """

    def __init__(self, input_dim: int = C3D_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) → (B, 1) 異常分數 ∈ [0, 1]"""
        return self.net(x)


# ── Prioritized Experience Replay ────────────────────────

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay（Aberkane & Boudihir 2019）
    以 |TD-error| 作為 priority；推論期間僅儲存、不訓練
    """

    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

    def push(self, transition: Transition, priority: float = 1.0):
        self.buffer.append(transition)
        self.priorities.append(max(priority, 1e-6))

    def sample(self, batch_size: int) -> List[Transition]:
        probs = np.array(self.priorities, dtype=np.float32)
        probs /= probs.sum()
        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, p=probs, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


# ── VideoSegment 資料結構 ─────────────────────────────────

@dataclass
class VideoSegment:
    segment_idx: int
    start_frame: int
    end_frame: int
    frames: List
    feature: Optional[np.ndarray] = None   # R3D-18 特徵（512D）
    anomaly_score: float = 0.0             # DQN 輸出
    illumination: float = 1.0             # 光線品質 [0,1]
    occlusion: float = 0.0               # 遮擋率 [0,1]
    is_anomalous: bool = False


# ── EnvironmentAgent ─────────────────────────────────────

class EnvironmentAgent(BaseAgent):
    """
    環境評估代理
    - Aberkane & Boudihir (2019): DQN 對每個時間片段評分，異常片段賦高權重
    - 32 temporal segments，Prioritized Experience Replay
    - 附加分析：光線（OpenCV LAB）、遮擋（MOG2）、場景分類（色彩啟發式）
    """

    def __init__(self, model_name: str = None):
        super().__init__(
            name="環境分析專家",
            model_name=model_name or cfg.model.base_model,
        )
        self.device = torch.device(
            cfg.model.device if torch.cuda.is_available() else "cpu"
        )
        self._dqn: Optional[DQNAnomalyDetector] = None
        self._r3d: Optional[nn.Module] = None
        self._replay_buffer = PrioritizedReplayBuffer()

    # ── 模型延遲載入 ─────────────────────────────────────

    def _load_models(self):
        """Lazy-load DQN 與 R3D-18（避免 import 時占用 GPU）"""
        # RTX 5090: cuDNN benchmark 自動調優
        if cfg.model.cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        if self._dqn is None:
            self._dqn = DQNAnomalyDetector(input_dim=C3D_DIM).to(self.device)
            self._dqn.eval()
            if cfg.model.compile_models:
                try:
                    self._dqn = torch.compile(self._dqn)
                except Exception:
                    pass
            logger.info("DQN Anomaly Detector 初始化（隨機權重；請載入訓練好的 checkpoint）")

        if self._r3d is None:
            import torchvision.models.video as vm
            backbone = vm.r3d_18(weights="DEFAULT")
            backbone.fc = nn.Identity()   # 移除分類頭，輸出 (B, 512)
            self._r3d = backbone.to(self.device)
            self._r3d.eval()
            if cfg.model.compile_models:
                try:
                    self._r3d = torch.compile(self._r3d)
                    logger.info("R3D-18 backbone 載入（torch.compile 已啟用）")
                except Exception:
                    logger.info("R3D-18 backbone 載入（作為 C3D 替代）")
            else:
                logger.info("R3D-18 backbone 載入（作為 C3D 替代）")

    # ── 主要分析流程 ─────────────────────────────────────

    def analyze(self, frames: List, video_metadata: Dict) -> AgentReport:
        """
        Aberkane & Boudihir (2019) 完整流程：
        1. 將影像切分為 32 個時間片段
        2. R3D-18 提取每片段時空特徵（C3D 替代）
        3. DQN 對每片段評分（異常/正常）
        4. 光線品質與遮擋評估
        5. 場景分類
        6. 建立環境可信度地圖
        """
        self._load_models()

        segments = self._segment_video(frames)
        segments = self._extract_features(segments)
        segments = self._score_segments_dqn(segments)
        segments = self._assess_segments_environment(segments)
        scene_type = self._classify_scene(frames, video_metadata)

        illumination_score = float(np.mean([s.illumination for s in segments]))
        occlusion_map = [s.occlusion for s in segments]
        frame_credibility = self._compute_frame_credibility(illumination_score, occlusion_map)

        anomalous_segs = [s for s in segments if s.is_anomalous]
        anomaly_ratio = len(anomalous_segs) / len(segments)

        evidence = [
            {
                "type": "dqn_anomaly_detection",
                "anomalous_segments": [s.segment_idx for s in anomalous_segs],
                "anomaly_ratio": round(anomaly_ratio, 3),
                "segment_scores": [round(s.anomaly_score, 3) for s in segments],
                "note": (
                    f"Aberkane & Boudihir (2019): "
                    f"{len(anomalous_segs)}/{NUM_SEGMENTS} 片段標記為異常"
                ),
            },
            {
                "type": "illumination",
                "score": illumination_score,
                "note": self._illumination_note(illumination_score),
            },
            {
                "type": "occlusion",
                "affected_frames": [i for i, v in enumerate(occlusion_map) if v > 0.5],
                "note": "高遮擋率可能降低視覺證據可信度",
            },
            {
                "type": "scene",
                "classification": scene_type,
                "risk_level": self._scene_risk_level(scene_type),
            },
        ]

        env_confidence = float(np.mean(frame_credibility))

        reasoning = (
            f"場景類型：{scene_type}，"
            f"平均光線分數：{illumination_score:.2f}，"
            f"環境可信度：{env_confidence:.2f}。"
            "Let's think step by step. "
            + self._generate_environmental_reasoning(
                illumination_score, occlusion_map, scene_type, anomaly_ratio, segments
            )
        )

        key_frames = self._extract_key_frames_from_segments(segments, frames)

        # ── estimated_env_contribution（供 Reflector Layer 3 直接讀取）────
        # 比較 raw anomaly scores 與 credibility-weighted scores 的差距，
        # 估算環境雜訊對異常偵測的貢獻度（貢獻越高 → 偽相關風險越高）。
        # weighted_avg = credibility × raw_score → 可信環境下的預期分數
        # contribution ≈ raw_avg - weighted_avg → 環境雜訊造成的分數膨脹
        seg_scores = [s.anomaly_score for s in segments]
        raw_avg       = float(np.mean(seg_scores)) if seg_scores else 0.0
        weighted_avg  = float(np.mean([
            sc * cr for sc, cr in zip(seg_scores, frame_credibility)
        ])) if seg_scores else 0.0
        estimated_env_contribution = float(np.clip(raw_avg - weighted_avg, 0.0, 1.0))

        # 高偽相關風險幀：可信度低 + 異常分數高的片段代表幀
        false_corr_risk_frames = [
            kf for kf, sc, cr in zip(
                key_frames,
                seg_scores[:len(key_frames)],
                frame_credibility[:len(key_frames)],
            )
            if cr < 0.4 and sc > ANOMALY_THRESHOLD
        ]

        report = AgentReport(
            agent_name=self.name,
            crime_category="ENVIRONMENTAL_ASSESSMENT",
            confidence=env_confidence,
            evidence=evidence,
            reasoning=reasoning,
            frame_references=key_frames,
            metadata={
                "frame_credibility": frame_credibility,
                "scene_type": scene_type,
                "illumination_score": illumination_score,
                "anomaly_ratio": anomaly_ratio,
                "segment_scores": seg_scores,
                # ↓ Reflector Layer 3 直接讀取的新欄位
                "estimated_env_contribution":  estimated_env_contribution,
                "false_correlation_risk_frames": false_corr_risk_frames,
            },
        )
        self._position = report
        return report

    def refine(self, other_reports: List[AgentReport]) -> AgentReport:
        """
        若 Action Agent 標記了特定關鍵幀，重新索引那些幀的環境可信度。
        """
        if self._position is None:
            raise RuntimeError("analyze() 必須先於 refine() 執行")

        action_frames = []
        for r in other_reports:
            if "行為" in r.agent_name:
                action_frames.extend(r.frame_references)

        if action_frames:
            credibility = self._position.metadata["frame_credibility"]
            self._position.metadata["key_frame_credibility"] = {
                f: credibility[min(f, len(credibility) - 1)]
                for f in action_frames
                if f < len(credibility)
            }

        return self._position

    # ── 影片切段 ─────────────────────────────────────────

    def _segment_video(self, frames: List) -> List[VideoSegment]:
        """將 frames 均分為 NUM_SEGMENTS=32 個時間片段（論文設定）"""
        n = len(frames)
        seg_size = max(1, n // NUM_SEGMENTS)
        segments = []
        for i in range(NUM_SEGMENTS):
            start = i * seg_size
            end = min(start + seg_size, n)
            seg_frames = frames[start:end] or [frames[-1]]
            segments.append(VideoSegment(
                segment_idx=i,
                start_frame=start,
                end_frame=end,
                frames=seg_frames,
            ))
        return segments

    # ── R3D-18 特徵提取（C3D 替代）──────────────────────

    def _extract_features(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """
        用 R3D-18 avgpool 輸出（512D）代替論文中的 C3D FC6（4096D）。
        輸入：(1, 3, T, H, W)，112×112，ImageNet 視訊標準化
        RTX 5090: BF16 autocast 加速推理
        """
        mean = torch.tensor([0.43216, 0.39467, 0.37645],
                            device=self.device).view(1, 3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145, 0.21699],
                            device=self.device).view(1, 3, 1, 1, 1)

        _use_amp = (str(self.device) == "cuda" and cfg.model.torch_dtype == "bfloat16")

        for seg in segments:
            try:
                clip = self._frames_to_tensor(seg.frames).unsqueeze(0).to(self.device)
                clip = (clip - mean) / std
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=_use_amp):
                    feat = self._r3d(clip)          # (1, 512)
                seg.feature = feat.float().squeeze(0).cpu().numpy()
            except Exception as e:
                logger.warning(f"片段 {seg.segment_idx} 特徵提取失敗：{e}")
                seg.feature = np.zeros(C3D_DIM, dtype=np.float32)

        return segments

    def _frames_to_tensor(self, frames: List) -> torch.Tensor:
        """frames（BGR ndarray list）→ (3, T, H, W) float32，範圍 [0,1]"""
        H, W = 112, 112
        clip = []
        for f in frames:
            if isinstance(f, np.ndarray):
                img = cv2.resize(f, (W, H))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                clip.append(img)

        # R3D-18 要求最少 1 幀，填充至 8 幀
        while len(clip) < 8:
            clip.append(clip[-1] if clip else np.zeros((H, W, 3), np.float32))

        arr = np.stack(clip, axis=0).transpose(3, 0, 1, 2)   # (3, T, H, W)
        return torch.from_numpy(arr)

    # ── DQN 異常評分 ─────────────────────────────────────

    def _score_segments_dqn(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """
        論文 MDP：狀態 = C3D 特徵，動作 = {異常, 正常}，獎勵 = 異常分數
        推論階段：前向傳播得分數，> ANOMALY_THRESHOLD 標記為異常
        同步儲存 Transition 供後續線上訓練
        """
        for seg in segments:
            if seg.feature is None:
                seg.anomaly_score = 0.0
                continue

            feat = torch.from_numpy(seg.feature).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                score = float(self._dqn(feat).item())

            seg.anomaly_score = score
            seg.is_anomalous = score > ANOMALY_THRESHOLD

            # 儲存 (s, a, r, s') 至 Replay Buffer
            action = 1 if seg.is_anomalous else 0
            reward = score if seg.is_anomalous else (1.0 - score)
            next_idx = seg.segment_idx + 1
            if next_idx < len(segments) and segments[next_idx].feature is not None:
                t = Transition(
                    state=seg.feature,
                    action=action,
                    reward=reward,
                    next_state=segments[next_idx].feature,
                    done=(next_idx == len(segments) - 1),
                )
                self._replay_buffer.push(t, priority=abs(score - 0.5) + 1e-5)

        return segments

    # ── 光線評估 ─────────────────────────────────────────

    def _assess_illumination(self, frames: List) -> float:
        """
        OpenCV LAB 色域 L 通道平均值 → 正規化光線分數 [0, 1]
        理想範圍 [80, 160]；過暗／過曝均扣分
        """
        if not frames:
            return 0.5

        l_values = []
        for f in frames:
            if not isinstance(f, np.ndarray):
                continue
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
            l_values.append(float(lab[:, :, 0].mean()))

        if not l_values:
            return 0.5

        avg_l = float(np.mean(l_values))
        if avg_l < DARK_THRESHOLD:
            score = avg_l / DARK_THRESHOLD * 0.4
        elif avg_l > BRIGHT_THRESHOLD:
            score = max(0.0, 1.0 - (avg_l - BRIGHT_THRESHOLD) / 55.0 * 0.4)
        else:
            score = 1.0 - abs(avg_l - 120.0) / 120.0 * 0.3

        return float(np.clip(score, 0.0, 1.0))

    def _assess_segments_environment(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """對每個片段分別計算光線與遮擋"""
        for seg in segments:
            seg.illumination = self._assess_illumination(seg.frames)
            seg.occlusion = self._detect_occlusion_segment(seg.frames)
        return segments

    # ── 遮擋偵測 ─────────────────────────────────────────

    def _detect_occlusion(self, frames: List) -> List[float]:
        """逐幀遮擋率（供 refine 介面兼容）"""
        return [self._detect_occlusion_segment([f]) for f in frames]

    def _detect_occlusion_segment(self, frames: List) -> float:
        """
        MOG2 背景消除：前景佔比高 → 可能有物體遮擋
        聚焦中心 80% 區域，忽略邊緣雜訊
        """
        if not frames or len(frames) < 2:
            return 0.1

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=len(frames), varThreshold=50, detectShadows=False
        )
        fg_ratios = []
        for f in frames:
            if not isinstance(f, np.ndarray):
                continue
            mask = mog2.apply(f)
            h, w = mask.shape[:2]
            py, px = h // 10, w // 10
            roi = mask[py:h - py, px:w - px]
            fg_ratios.append(float(np.count_nonzero(roi)) / max(roi.size, 1))

        return float(np.clip(np.mean(fg_ratios), 0.0, 1.0)) if fg_ratios else 0.1

    # ── 場景分類 ─────────────────────────────────────────

    def _classify_scene(self, frames: List, metadata: Dict) -> str:
        """
        啟發式場景分類：
        - 天空（藍色）比例 → outdoor
        - 地面（低飽和度）比例 → parking
        - 亮度標準差 → indoor 程度
        - metadata 檔名關鍵字優先
        """
        if not frames:
            return "unknown"

        mid = frames[len(frames) // 2]
        if not isinstance(mid, np.ndarray):
            return "unknown"

        # metadata 檔名關鍵字
        filename = str(metadata.get("filename", "")).lower()
        for kw, label in [
            (["park", "lot", "carpark"], "outdoor_parking"),
            (["shop", "store", "mall", "market"], "indoor_store"),
            (["office", "corridor", "hall", "lobby"], "indoor_office"),
            (["road", "street", "highway"], "outdoor_street"),
        ]:
            if any(k in filename for k in kw):
                return label

        h, w = mid.shape[:2]
        hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(mid, cv2.COLOR_BGR2LAB)

        # 天空藍色比例（上 1/3）
        sky_region = hsv[:h // 3, :, :]
        blue_mask = cv2.inRange(sky_region,
                                np.array([100, 30, 80]),
                                np.array([140, 255, 255]))
        sky_ratio = float(np.count_nonzero(blue_mask)) / max(sky_region[:, :, 0].size, 1)

        # 地面低飽和度比例（下 1/2）
        lower = hsv[h // 2:, :, :]
        low_sat = cv2.inRange(lower, np.array([0, 0, 50]), np.array([180, 50, 220]))
        asphalt_ratio = float(np.count_nonzero(low_sat)) / max(lower[:, :, 0].size, 1)

        # 亮度標準差
        brightness_std = float(lab[:, :, 0].std())

        if sky_ratio > 0.15:
            return "outdoor_parking" if asphalt_ratio > 0.3 else "outdoor_street"
        if brightness_std < 25:
            return "indoor_office"
        if brightness_std < 45:
            return "indoor_store"
        return "outdoor_street"

    # ── 可信度計算 ────────────────────────────────────────

    def _compute_frame_credibility(
        self, illumination: float, occlusion_map: List[float]
    ) -> List[float]:
        """可信度 = 0.6 × 光線分數 + 0.4 × (1 − 遮擋率)"""
        return [
            float(np.clip(0.6 * illumination + 0.4 * (1.0 - occ), 0.0, 1.0))
            for occ in occlusion_map
        ]

    # ── 輔助方法 ─────────────────────────────────────────

    def _scene_risk_level(self, scene_type: str) -> str:
        return {
            "outdoor_street": "medium",
            "outdoor_parking": "high",
            "indoor_store": "medium",
            "indoor_office": "low",
            "vehicle": "high",
            "unknown": "medium",
        }.get(scene_type, "medium")

    def _illumination_note(self, score: float) -> str:
        if score < 0.3:
            return "嚴重光線不足，視覺證據可信度低"
        if score < 0.6:
            return "光線不佳，部分細節可能失真"
        return "光線條件良好"

    def _generate_environmental_reasoning(
        self,
        illumination: float,
        occlusion_map: List[float],
        scene: str,
        anomaly_ratio: float,
        segments: List[VideoSegment],
    ) -> str:
        avg_occ = float(np.mean(occlusion_map)) if occlusion_map else 0.0
        top3 = sorted(segments, key=lambda s: s.anomaly_score, reverse=True)[:3]
        top_desc = "、".join(
            f"片段{s.segment_idx}(得分{s.anomaly_score:.2f})" for s in top3
        )
        light_note = (
            "光線不足可能導致部分細節遺失；"
            if illumination < 0.5
            else "光線充足，視覺證據可信。"
        )
        occ_note = (
            "高遮擋率可能遮蔽關鍵行為，需搭配其他代理人交叉驗證。"
            if avg_occ > 0.4
            else ""
        )
        return (
            f"場景為 {scene}，平均遮擋率 {avg_occ:.2f}。"
            f"DQN 偵測 {anomaly_ratio * 100:.1f}% 片段具異常特徵（閾值={ANOMALY_THRESHOLD}）。"
            f"異常得分最高片段：{top_desc}。"
            f"環境可信度評估：{light_note}{occ_note}"
        )

    def _extract_key_frames(self, frames: List) -> List[int]:
        return list(range(0, len(frames), max(1, len(frames) // 5)))[:5]

    def _extract_key_frames_from_segments(
        self, segments: List[VideoSegment], frames: List
    ) -> List[int]:
        """前 3 異常片段的起始幀 + 均勻採樣補至 8 幀"""
        top3 = sorted(segments, key=lambda s: s.anomaly_score, reverse=True)[:3]
        key = {s.start_frame for s in top3}
        step = max(1, len(frames) // 5)
        for i in range(0, len(frames), step):
            key.add(i)
            if len(key) >= 8:
                break
        return sorted(key)[:8]
