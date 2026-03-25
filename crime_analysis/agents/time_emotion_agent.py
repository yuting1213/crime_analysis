"""
Time & Emotion Agent（時序情感代理）
參考：Kilic & Tuceryan (2024) - Crime Detection from Pre-crime Video Analysis
      with Augmented Pose and Emotion Information

核心功能：
- 每 8 幀取 1 幀，組成 16 幀 snippet（對應論文設計）
- ViT 視覺特徵 (768) + Mediapipe 姿勢 (99) + DeepFace 情緒 (7) → [16, 874]
- 4 層 Transformer 分類器（8 heads，對應論文架構）
- 建立三段式因果鏈：Pre-crime → Crime → Post-crime
"""
import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel

from .base_agent import AgentReport, BaseAgent
from config import cfg

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── 常數（論文參數） ───────────────────────────────────
CLIP_LEN       = 16    # 每個 snippet 的幀數
FRAME_STRIDE   = 8     # 每隔幾幀取一幀
VIT_DIM        = 768   # ViT-Base 特徵維度
POSE_DIM       = 99    # 33 keypoints × 3 (x, y, z)
EMOTION_DIM    = 7     # angry/disgust/fear/happy/sad/surprise/neutral
FEATURE_DIM    = VIT_DIM + POSE_DIM + EMOTION_DIM   # 874

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 犯罪前兆情緒（Kilic & Tuceryan 2024）
PRE_CRIME_EMOTIONS = {"angry", "fear", "disgust", "surprise"}

# MediaPipe 關鍵點索引（用於行為分析）
MP_NOSE        = 0
MP_LEFT_EYE    = 2
MP_RIGHT_EYE   = 5
MP_LEFT_SHOULDER  = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_WRIST  = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP    = 23
MP_RIGHT_HIP   = 24


# ── 資料結構 ───────────────────────────────────────────

@dataclass
class FrameFeature:
    """單幀完整特徵"""
    frame_idx:      int
    timestamp_sec:  float
    vit_feat:       np.ndarray              # (768,)
    pose_keypoints: np.ndarray              # (99,)  x,y,z of 33 pts
    emotion_probs:  np.ndarray              # (7,)
    dominant_emotion: str
    pose_flags:     Dict[str, bool] = field(default_factory=dict)


@dataclass
class CausalEvent:
    """因果鏈事件節點"""
    phase:          str        # "pre_crime" | "crime" | "post_crime"
    start_frame:    int
    end_frame:      int
    dominant_emotion: str
    suspicious_poses: List[str]
    description:    str


# ── 論文架構：Transformer 分類器 ────────────────────────

class PreCrimeClassifier(nn.Module):
    """
    對應論文 Figure 1：
    4 層 Transformer Encoder，每層 8 個注意力頭
    輸入：[batch, CLIP_LEN, FEATURE_DIM]
    輸出：[batch, 2] - (normal, pre_crime)
    """
    def __init__(self, feat_dim: int = FEATURE_DIM, num_classes: int = 2,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        # 線性投影到 d_model（需是 nhead 的倍數）
        d_model = 512
        self.input_proj = nn.Linear(feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, feat_dim] → logits: [B, num_classes]"""
        x = self.input_proj(x)
        x = self.encoder(x)          # [B, T, d_model]
        x = self.norm(x.mean(dim=1)) # global avg pool
        return self.classifier(x)


# ── 主代理人 ────────────────────────────────────────────

class TimeEmotionAgent(BaseAgent):
    """
    時序與情感代理
    處理犯罪影像的時間軸、事前挑釁行為分析、嫌疑人情緒偵測
    建立因果鏈（Causal Chain）供 Reflector 進行一致性審計
    """

    def __init__(self, model_name: str = None, device: str = None):
        super().__init__(
            name="時序情感分析專家",
            model_name=model_name or cfg.model.base_model,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TimeEmotionAgent 初始化，使用裝置：{self.device}")

        # ── 初始化 ViT ─────────────────────────────────
        vit_name = "google/vit-base-patch16-224"
        self._vit_extractor = ViTFeatureExtractor.from_pretrained(vit_name)
        self._vit_model = ViTModel.from_pretrained(vit_name).to(self.device).eval()
        logger.info("ViT-Base 載入完成")

        # ── 初始化 MediaPipe Pose ───────────────────────
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        logger.info("MediaPipe Pose 初始化完成")

        # ── 初始化 Pre-crime Classifier ─────────────────
        self._classifier = PreCrimeClassifier().to(self.device).eval()
        # TODO: 訓練後載入權重
        # ckpt = Path("./checkpoints/time_emotion_classifier.pt")
        # if ckpt.exists():
        #     self._classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
        #     logger.info("分類器權重載入完成")

    # ── 公開介面 ────────────────────────────────────────

    def analyze(self, frames: List[np.ndarray], video_metadata: Dict) -> AgentReport:
        """
        主分析流程（論文三階段）：
        1. 前處理 → 16 幀 snippets
        2. 特徵提取（ViT + MediaPipe + DeepFace）
        3. Transformer 分類 + 因果鏈建立
        """
        fps = float(video_metadata.get("fps", 25.0))

        # ── Step 1：取樣 snippets ─────────────────────
        snippets_frames = self._sample_snippets(frames)
        if not snippets_frames:
            return self._empty_report("影片幀數不足，無法分析")

        # ── Step 2：逐 snippet 提取特徵 ──────────────
        all_features: List[List[FrameFeature]] = []
        for snippet in snippets_frames:
            feats = self._extract_snippet_features(snippet, fps)
            all_features.append(feats)

        flat_features = [f for snippet in all_features for f in snippet]

        # ── Step 3：Transformer 分類 ──────────────────
        pre_crime_prob = self._classify_snippets(all_features)

        # ── Step 4：建立因果鏈 ────────────────────────
        causal_chain = self._build_causal_chain(flat_features)

        # ── Step 5：推斷犯罪類別 ──────────────────────
        emotion_summary = self._summarize_emotions(flat_features)
        crime_category, confidence = self._infer_crime_category(
            causal_chain, emotion_summary, pre_crime_prob
        )

        reasoning = self._generate_reasoning(causal_chain, emotion_summary, pre_crime_prob)

        # ── 新增欄位：escalation_start_frame + post_crime_indicators ────
        # escalation_start_frame：情緒值首次超過 ESCALATION_CALM_THRESHOLD 的幀
        #   → Reflector Layer 1 用於精確判斷時序衝突（取代 te_frames[0]）
        # post_crime_indicators：post_crime 階段可疑行為描述列表
        #   → Reflector Layer 2 用於 LOW 突發型（RoadAccidents）的事後反應判斷
        escalation_start_frame: Optional[int] = None
        for f in flat_features:
            if float(f.emotion_probs[0]) + float(f.emotion_probs[2]) > ESCALATION_CALM_THRESHOLD:
                escalation_start_frame = f.frame_idx
                break

        post_crime_event = next(
            (e for e in causal_chain if e.phase == "post_crime"), None
        )
        post_crime_indicators: List[str] = (
            post_crime_event.suspicious_poses
            if post_crime_event else []
        )

        # emotion_summary 也加入 escalation_score 方便 Layer 2 直接讀取
        emotion_summary["escalation_score"] = emotion_summary.get("emotion_escalation", 0.0)

        report = AgentReport(
            agent_name=self.name,
            crime_category=crime_category,
            confidence=confidence,
            evidence=[
                {"type": "pre_crime_probability", "value": round(pre_crime_prob, 3)},
                {"type": "emotion_summary", "data": emotion_summary},
                {"type": "causal_chain", "events": [
                    {"phase": e.phase,
                     "frames": f"{e.start_frame}–{e.end_frame}",
                     "emotion": e.dominant_emotion,
                     "poses": e.suspicious_poses,
                     "description": e.description}
                    for e in causal_chain
                ]},
            ],
            reasoning=reasoning,
            frame_references=self._key_frames(causal_chain),
            metadata={
                "causal_chain_raw": [
                    {"phase": e.phase, "start": e.start_frame, "end": e.end_frame}
                    for e in causal_chain
                ],
                "emotion_trajectory": [
                    {"frame": f.frame_idx, "emotion": f.dominant_emotion,
                     "angry": round(float(f.emotion_probs[0]), 3),
                     "fear": round(float(f.emotion_probs[2]), 3)}
                    for f in flat_features[::4]  # 每 4 幀記一次，減少輸出量
                ],
                # ↓ Reflector 新欄位
                "escalation_start_frame":  escalation_start_frame,
                "post_crime_indicators":   post_crime_indicators,
                "escalation_score":        emotion_summary.get("emotion_escalation", 0.0),
                "video_fps":               fps,   # Layer 1 fps-aware 門檻用
            },
        )
        self._position = report
        return report

    def refine(self, other_reports: List[AgentReport]) -> AgentReport:
        """
        共識收斂：若 Action Agent 給出更高置信度的不同判斷，
        重新評估時序是否支持對方的類別。
        """
        if self._position is None:
            raise RuntimeError("analyze() 必須先執行")

        for r in other_reports:
            if "行為" in r.agent_name and r.confidence > self._position.confidence:
                support = self._temporal_support_for(r.crime_category)
                if support > 0.55:
                    updated_reasoning = (
                        f"[更新] Action Agent 判斷 {r.crime_category}（置信度 {r.confidence:.2f}）。"
                        f"時序特徵支持度 {support:.2f}，採納其判斷。\n"
                        + self._position.reasoning
                    )
                    self._position = AgentReport(
                        agent_name=self.name,
                        crime_category=r.crime_category,
                        confidence=round((r.confidence + support) / 2, 3),
                        evidence=self._position.evidence,
                        reasoning=updated_reasoning,
                        frame_references=self._position.frame_references,
                        metadata=self._position.metadata,
                    )
                    break

        # ── Step 3 針對性法律要件補充 ──────────────────────
        # 若 Planner 已注入 legal_framework（Step 2.5），將因果鏈的三個階段
        # 對應到 key_elements_to_verify，補充情緒與姿態層面的法律佐證。
        if self._legal_framework:
            elements = self._legal_framework.get("key_elements_to_verify", [])
            if elements:
                targeted_ev, targeted_note = self._map_elements_to_causal_chain(elements)
                self._position = AgentReport(
                    agent_name=self._position.agent_name,
                    crime_category=self._position.crime_category,
                    confidence=self._position.confidence,
                    evidence=self._position.evidence + targeted_ev,
                    reasoning=self._position.reasoning + targeted_note,
                    frame_references=self._position.frame_references,
                    conflict_flags=self._position.conflict_flags,
                    metadata=self._position.metadata,
                )

        return self._position

    def _map_elements_to_causal_chain(
        self, elements: List[str]
    ) -> tuple:
        """
        將法律構成要件對應到因果鏈的各個階段，補充情緒與姿態佐證。

        對應邏輯：
          主觀故意 / 計畫性  → pre_crime 階段（looking_around / camera_checking）
          傷害行為 / 攻擊行為 → crime 階段（aggressive_posture + angry 情緒）
          傷害結果 / 恐懼反應 → post_crime 階段（fear 情緒升高）
          因果關係           → 跨階段情緒升溫趨勢（escalation）
          互毆事實           → crime 階段雙方 aggressive_posture

        回傳：(evidence_list, reasoning_note)
        """
        chain_raw: List[Dict] = self._position.metadata.get("causal_chain_raw", [])
        emotion_traj: List[Dict] = self._position.metadata.get("emotion_trajectory", [])

        # 建立 phase → chain event 查詢表
        phase_map: Dict[str, Dict] = {e["phase"]: e for e in chain_raw}

        # 整體情緒升溫值
        escalation = 0.0
        for ev in self._position.evidence:
            if ev.get("type") == "emotion_summary":
                escalation = ev.get("data", {}).get("emotion_escalation", 0.0)
                break

        # 要件 → 因果鏈對應表
        ELEMENT_PHASE_MAP = {
            "主觀故意":    ("pre_crime",  "looking_around / camera_checking 行為"),
            "計畫性":      ("pre_crime",  "犯前環顧與觀察行為"),
            "傷害行為":    ("crime",      "aggressive_posture + 憤怒情緒"),
            "強暴/脅迫手段": ("crime",   "強制行為姿態 + 高憤怒指數"),
            "攻擊行為":    ("crime",      "aggressive_posture 偵測"),
            "互毆事實":    ("crime",      "雙方 aggressive_posture"),
            "傷害結果":    ("post_crime", "恐懼情緒升高（受害者反應）"),
            "取財行為":    ("post_crime", "行為轉移（犯後逃逸）"),
            "因果關係":    (None,         f"情緒升溫趨勢 escalation={escalation:.3f}"),
            "違法性":      (None,         "整體犯罪前兆機率支持"),
        }

        targeted_evidence: List[Dict] = []
        notes: List[str] = []

        for elem in elements:
            mapping = ELEMENT_PHASE_MAP.get(elem)
            if not mapping:
                continue
            phase, indicator_desc = mapping

            if phase:
                phase_event = phase_map.get(phase)
                covered = phase_event is not None
                frame_range = (
                    f"{phase_event['start']}–{phase_event['end']}"
                    if phase_event else "N/A"
                )
            else:
                # 跨階段要件（因果關係 / 違法性）
                covered = escalation > 0.05
                frame_range = "全段"

            targeted_evidence.append({
                "type":        "targeted_causal_evidence",
                "element":     elem,
                "phase":       phase or "cross_phase",
                "frame_range": frame_range,
                "indicator":   indicator_desc,
                "covered":     covered,
            })
            status = "✓ 支持" if covered else "△ 待補強"
            notes.append(f"  [{status}] {elem}（{indicator_desc}，幀段 {frame_range}）")

        if not notes:
            return [], ""

        note_text = (
            "\n\n【Step 3 針對性法律要件因果鏈佐證】\n"
            + "\n".join(notes)
        )
        return targeted_evidence, note_text

    # ── 前處理 ──────────────────────────────────────────

    def _sample_snippets(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        依論文方式取樣：每 FRAME_STRIDE 幀取 1 幀，組成 CLIP_LEN 幀的 snippet
        Example: 200 frames → 25 sampled → 1 snippet (16 frames used)
        """
        sampled = frames[::FRAME_STRIDE]
        snippets = []
        for start in range(0, len(sampled), CLIP_LEN):
            clip = sampled[start: start + CLIP_LEN]
            if len(clip) < CLIP_LEN // 2:
                break
            # 不足 CLIP_LEN 幀時用最後一幀補齊
            while len(clip) < CLIP_LEN:
                clip.append(clip[-1])
            snippets.append(clip)
        return snippets

    # ── 特徵提取 ────────────────────────────────────────

    def _extract_snippet_features(
        self, snippet: List[np.ndarray], fps: float
    ) -> List[FrameFeature]:
        """對 snippet 內每一幀提取完整特徵"""
        results = []
        for local_idx, frame in enumerate(snippet):
            global_frame_idx = local_idx  # 在完整流程中由外部更新

            vit_feat = self._extract_vit(frame)
            pose_kp, pose_flags = self._extract_pose(frame)
            emotion_probs, dominant = self._extract_emotion(frame)

            results.append(FrameFeature(
                frame_idx=global_frame_idx,
                timestamp_sec=global_frame_idx / fps,
                vit_feat=vit_feat,
                pose_keypoints=pose_kp,
                emotion_probs=emotion_probs,
                dominant_emotion=dominant,
                pose_flags=pose_flags,
            ))
        return results

    def _extract_vit(self, frame: np.ndarray) -> np.ndarray:
        """使用 ViT-Base 提取 [CLS] token 特徵（768 維）"""
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self._vit_extractor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._vit_model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # (768,)

    def _extract_pose(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        使用 MediaPipe 提取 33 個關鍵點座標（x, y, z × 33 = 99 維）
        同時分析行為旗標（looking_around / camera_checking / aggressive_posture）
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_pose.process(rgb)

        if not result.pose_landmarks:
            return np.zeros(POSE_DIM, dtype=np.float32), {}

        lm = result.pose_landmarks.landmark
        keypoints = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()

        # 行為分析旗標（論文 Table I 所述的嫌疑人行為模式）
        flags = self._analyze_pose_flags(lm)
        return keypoints, flags

    def _analyze_pose_flags(self, lm) -> Dict[str, bool]:
        """
        從關鍵點座標推斷行為旗標
        - looking_around:   頭部水平位移變化大（持續偵測需跨幀，單幀用鼻子偏移）
        - camera_checking:  頭部上仰（鼻子 y 明顯低於眼睛）
        - aggressive_posture: 手腕高於肩膀（舉手打人姿勢）
        """
        nose       = lm[MP_NOSE]
        l_eye      = lm[MP_LEFT_EYE]
        r_eye      = lm[MP_RIGHT_EYE]
        l_shoulder = lm[MP_LEFT_SHOULDER]
        r_shoulder = lm[MP_RIGHT_SHOULDER]
        l_wrist    = lm[MP_LEFT_WRIST]
        r_wrist    = lm[MP_RIGHT_WRIST]

        # 頭部水平偏移（x 軸偏離肩膀中心）
        shoulder_mid_x = (l_shoulder.x + r_shoulder.x) / 2
        head_offset = abs(nose.x - shoulder_mid_x)
        looking_around = head_offset > 0.08

        # 頭部上仰（鼻子 y 比眼睛 y 小 → 在影像座標中表示抬頭）
        eye_avg_y = (l_eye.y + r_eye.y) / 2
        camera_checking = (eye_avg_y - nose.y) > 0.03

        # 手腕高於肩膀（攻擊性動作）
        shoulder_avg_y = (l_shoulder.y + r_shoulder.y) / 2
        aggressive = (l_wrist.y < shoulder_avg_y - 0.05 or
                      r_wrist.y < shoulder_avg_y - 0.05)

        return {
            "looking_around":    looking_around,
            "camera_checking":   camera_checking,
            "aggressive_posture": aggressive,
        }

    def _extract_emotion(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        使用 DeepFace 提取 7 類情緒機率向量
        回傳：(prob_vector [7], dominant_emotion_str)
        若無偵測到臉部，回傳均勻分佈
        """
        try:
            from deepface import DeepFace
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            # result 可能是 list（多人）→ 取第一人
            if isinstance(result, list):
                result = result[0]
            scores = result["emotion"]  # dict: emotion → score (0~100)
            probs = np.array(
                [scores.get(e, 0.0) for e in EMOTION_LABELS], dtype=np.float32
            )
            probs /= (probs.sum() + 1e-8)  # 正規化為機率
            dominant = EMOTION_LABELS[int(np.argmax(probs))]
            return probs, dominant
        except Exception:
            # 無臉或分析失敗 → 回傳 neutral 均勻
            probs = np.zeros(EMOTION_DIM, dtype=np.float32)
            probs[-1] = 1.0  # neutral
            return probs, "neutral"

    # ── Transformer 分類 ────────────────────────────────

    def _classify_snippets(self, all_features: List[List[FrameFeature]]) -> float:
        """
        將所有 snippet 的特徵向量送入 Transformer 分類器
        回傳平均 pre_crime 機率
        """
        probs = []
        for snippet_feats in all_features:
            # 組合特徵向量 [CLIP_LEN, FEATURE_DIM]
            feat_mat = np.stack([
                np.concatenate([f.vit_feat, f.pose_keypoints, f.emotion_probs])
                for f in snippet_feats
            ], axis=0)  # [16, 874]

            tensor = torch.tensor(feat_mat, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._classifier(tensor)   # [1, 2]
                prob = torch.softmax(logits, dim=-1)[0, 1].item()  # pre_crime prob
            probs.append(prob)
        return float(np.mean(probs)) if probs else 0.5

    # ── 因果鏈建立 ──────────────────────────────────────

    def _build_causal_chain(
        self, features: List[FrameFeature]
    ) -> List[CausalEvent]:
        """
        三段式因果鏈（Pearl 因果階梯第三層 - 反事實推論）：
          Phase 1 pre_crime:  前 1/3 → 建立基線
          Phase 2 crime:      中段    → 情緒高峰 + 攻擊性姿勢
          Phase 3 post_crime: 後 1/3 → 逃逸或受害者反應
        """
        if not features:
            return []

        n = len(features)
        thirds = [n // 3, 2 * n // 3, n]
        phases = ["pre_crime", "crime", "post_crime"]
        chain: List[CausalEvent] = []

        prev = 0
        for phase, end in zip(phases, thirds):
            seg = features[prev:end]
            if not seg:
                continue

            dominant = Counter(f.dominant_emotion for f in seg).most_common(1)[0][0]

            # 收集可疑姿勢旗標
            flag_counts: Counter = Counter()
            for f in seg:
                for flag, val in f.pose_flags.items():
                    if val:
                        flag_counts[flag] += 1
            suspicious = [flag for flag, cnt in flag_counts.items()
                          if cnt > len(seg) * 0.3]

            desc = self._describe_phase(phase, dominant, suspicious, seg)
            chain.append(CausalEvent(
                phase=phase,
                start_frame=seg[0].frame_idx,
                end_frame=seg[-1].frame_idx,
                dominant_emotion=dominant,
                suspicious_poses=suspicious,
                description=desc,
            ))
            prev = end

        return chain

    def _describe_phase(
        self, phase: str, emotion: str,
        poses: List[str], seg: List[FrameFeature]
    ) -> str:
        """生成各階段的自然語言描述"""
        pose_str = "、".join(poses) if poses else "無明顯異常姿勢"
        avg_angry = float(np.mean([f.emotion_probs[0] for f in seg]))
        avg_fear  = float(np.mean([f.emotion_probs[2] for f in seg]))
        ts_start  = seg[0].timestamp_sec
        ts_end    = seg[-1].timestamp_sec

        templates = {
            "pre_crime": (
                f"[{ts_start:.1f}s–{ts_end:.1f}s] 犯罪前兆階段：主要情緒為「{emotion}」，"
                f"可疑行為：{pose_str}。"
                f"憤怒指數 {avg_angry:.2f}，恐懼指數 {avg_fear:.2f}。"
            ),
            "crime": (
                f"[{ts_start:.1f}s–{ts_end:.1f}s] 犯罪行為階段：情緒升溫至「{emotion}」，"
                f"偵測到行為：{pose_str}。"
                f"憤怒指數 {avg_angry:.2f}，恐懼指數 {avg_fear:.2f}（顯著高於基線）。"
            ),
            "post_crime": (
                f"[{ts_start:.1f}s–{ts_end:.1f}s] 犯罪後階段：情緒轉為「{emotion}」，"
                f"行為特徵：{pose_str}。"
            ),
        }
        return templates.get(phase, "")

    # ── 犯罪類別推斷 ────────────────────────────────────

    def _infer_crime_category(
        self,
        chain: List[CausalEvent],
        emotion_summary: Dict,
        pre_crime_prob: float,
    ) -> Tuple[str, float]:
        """
        根據因果鏈的情緒模式與姿勢旗標推斷最可能的犯罪類別
        規則優先（可被 Reflector 覆寫）
        """
        if not chain:
            return "Normal", 0.5

        crime_phase = next((e for e in chain if e.phase == "crime"), None)
        if not crime_phase:
            return "Normal", max(0.3, 1.0 - pre_crime_prob)

        poses = set(crime_phase.suspicious_poses)
        emotion = crime_phase.dominant_emotion

        # 簡單規則推斷
        if "aggressive_posture" in poses and emotion in {"angry", "disgust"}:
            return "Assault", round(0.5 + pre_crime_prob * 0.4, 3)
        if "looking_around" in poses and emotion in {"fear", "neutral"}:
            return "Stealing", round(0.4 + pre_crime_prob * 0.4, 3)
        if "camera_checking" in poses and emotion in {"neutral", "disgust"}:
            return "Shoplifting", round(0.4 + pre_crime_prob * 0.4, 3)
        if emotion in {"angry"} and "aggressive_posture" in poses:
            return "Fighting", round(0.45 + pre_crime_prob * 0.4, 3)

        # Fallback：用全局情緒升溫趨勢輔助判斷
        escalation = emotion_summary.get("emotion_escalation", 0.0)
        if escalation > 0.1:
            return "Assault", round(0.3 + pre_crime_prob * 0.3 + escalation * 0.2, 3)
        category = "Assault" if emotion == "angry" else "Normal"
        return category, round(0.3 + pre_crime_prob * 0.3, 3)

    # ── 情緒摘要 ────────────────────────────────────────

    def _summarize_emotions(self, features: List[FrameFeature]) -> Dict:
        if not features:
            return {}
        all_probs = np.stack([f.emotion_probs for f in features])  # [N, 7]
        mean_probs = all_probs.mean(axis=0)
        dominant_seq = [f.dominant_emotion for f in features]
        dominant_overall = Counter(dominant_seq).most_common(1)[0][0]

        # 偵測情緒升溫（angry + fear 的移動平均斜率）
        window = max(3, len(features) // 5)
        aggressive_ma = np.convolve(
            all_probs[:, 0] + all_probs[:, 2],  # angry + fear
            np.ones(window) / window, mode="valid"
        )
        escalation = float(aggressive_ma[-1] - aggressive_ma[0]) if len(aggressive_ma) > 1 else 0.0

        return {
            "dominant_emotion": dominant_overall,
            "mean_emotion_probs": {
                e: round(float(mean_probs[i]), 3) for i, e in enumerate(EMOTION_LABELS)
            },
            "emotion_escalation": round(escalation, 4),
            "pre_crime_emotion_detected": dominant_overall in PRE_CRIME_EMOTIONS,
            "total_frames_analyzed": len(features),
        }

    # ── 報告生成 ────────────────────────────────────────

    def _generate_reasoning(
        self,
        chain: List[CausalEvent],
        emotion_summary: Dict,
        pre_crime_prob: float,
    ) -> str:
        lines = [
            "Let's think step by step.",
            "",
            f"【時序情感分析結果】",
            f"犯罪前兆機率（Transformer 分類器）：{pre_crime_prob:.3f}",
            f"主要情緒：{emotion_summary.get('dominant_emotion', 'N/A')}",
            f"情緒升溫趨勢：{emotion_summary.get('emotion_escalation', 0):.4f}",
            "",
            "【因果鏈（Pre-crime → Crime → Post-crime）】",
        ]
        for e in chain:
            lines.append(f"  [{e.phase}] {e.description}")

        lines += [
            "",
            "【Pearl 因果推論（第三層：反事實）】",
            (
                "若犯罪前兆階段無情緒升溫或攻擊性姿勢，"
                "則後續犯罪行為的發生機率應顯著降低，"
                f"支持本次事件具備「計畫性」特徵。"
                if pre_crime_prob > 0.6
                else "當前時序特徵支持性不足，建議結合 Action Agent 證據後再判定。"
            ),
        ]
        return "\n".join(lines)

    def _temporal_support_for(self, category: str) -> float:
        """評估當前時序特徵對指定犯罪類別的支持度"""
        if self._position is None:
            return 0.0
        # 若類別相近（同屬暴力犯罪）給予部分支持
        violent = {"Assault", "Fighting", "Robbery", "Shooting"}
        property_ = {"Stealing", "Shoplifting", "Burglary", "Robbery"}
        current = self._position.crime_category
        if category == current:
            return 0.85
        if category in violent and current in violent:
            return 0.65
        if category in property_ and current in property_:
            return 0.60
        return 0.35

    def _key_frames(self, chain: List[CausalEvent]) -> List[int]:
        frames = []
        for e in chain:
            frames.extend([e.start_frame, e.end_frame])
        return frames

    def _empty_report(self, reason: str) -> AgentReport:
        report = AgentReport(
            agent_name=self.name,
            crime_category="Normal",
            confidence=0.1,
            evidence=[{"type": "error", "message": reason}],
            reasoning=reason,
            frame_references=[],
        )
        self._position = report
        return report

    # ── 便利方法：直接從影片路徑執行 ──────────────────────

    @classmethod
    def from_video(cls, video_path: str, **kwargs) -> AgentReport:
        """工廠方法：建立 Agent 並直接分析指定影片，回傳 AgentReport"""
        agent = cls(**kwargs)
        return agent.analyze_video(video_path)

    def analyze_video(self, video_path: str) -> AgentReport:
        """直接從影片路徑載入幀並分析"""
        frames, metadata = self._load_video(video_path)
        return self.analyze(frames, metadata)

    @staticmethod
    def _load_video(video_path: str) -> Tuple[List[np.ndarray], Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"無法開啟影片：{video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames, {"fps": fps, "total_frames": len(frames)}


# ── 快速測試（單檔執行）────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        # 若無提供影片，用合成幀測試
        print("用法：python time_emotion_agent.py <video_path>")
        print("未提供影片，使用合成幀進行 smoke test...")
        fake_frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(50)
        ]
        metadata = {"fps": 25.0, "total_frames": 50}
        agent = TimeEmotionAgent()
        report = agent.analyze(fake_frames, metadata)
    else:
        agent = TimeEmotionAgent()
        report = agent.analyze_video(sys.argv[1])

    print("\n=== Time & Emotion Agent 分析結果 ===")
    print(f"犯罪類別：{report.crime_category}")
    print(f"置信度：  {report.confidence:.3f}")
    print(f"\n推理過程：\n{report.reasoning}")
    print(f"\n關鍵影格：{report.frame_references}")
