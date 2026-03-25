"""
Reflector Agent - 內部一致性審計代理

各層分工（更新後）：
  Layer 1 - Temporal Consistency [Constraint Propagation]
            規則引擎：fps-aware 時序自洽性，幀取法明確化
  Layer 2 - Antecedent Analysis  [Constraint Propagation]
            規則引擎：因果前兆驗證，含 LOW severity 分類（預謀型/突發型）
  Layer 3 - Spurious Correlation [CASAM]
            保留注意力稀疏化機制，直接讀取 Environment Agent 輸出的
            estimated_env_contribution，不再重新推論信心降幅

Rcons 計算：Dempster-Shafer 衝突係數 K（取代原乘法懲罰公式）
  每個 Agent 輸出信念質量函數 m(crime / no_crime / uncertain)
  K = Σ m_i(A)·m_j(B) for A∩B=∅ （pairwise 平均）
  Rcons_base = 1.0 − K
  Rcons_final = clip(Rcons_base + bonus, 0.0, 1.3)

衝突分類（不變）：
  HARD → 回傳 Planner 完整衝突報告
  SOFT → 直接傳遞目標代理人具體修正指令
  NONE → 通過，附一致性分數
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentReport
from config import cfg

logger = logging.getLogger(__name__)

# ── 常數 ─────────────────────────────────────────────────

# Layer 1：時序門檻（fps-aware）
# 用秒數定義，運行時乘以 video_fps 換算為幀數
# → UCF-Crime 30fps  ≈ 30 frames
# → XD-Violence 25fps ≈ 25 frames
TEMPORAL_GAP_SECONDS   = 1.0          # 攻擊幀早於情緒升溫幀的上限（秒）
DEFAULT_VIDEO_FPS      = 30.0         # 無法從報告取得 fps 時的預設值

# Layer 1：類別衝突判定
CONFIDENCE_GAP_HARD    = 0.4          # 衝突代理人間信心差 > 此值 → HARD
CONFIDENCE_SOFT_MAX    = 0.6          # 信心 < 此值 + 無直接矛盾 → SOFT

# Layer 2：信心門檻（待 pilot experiment 確認）
# 做法：跑 20-30 個 UCF-Crime 標準案例，
#        取 Action Agent 正確分類時信心的第 75 百分位數作為 HARD 門檻。
CONFIDENCE_HARD_THRESHOLD = 0.75      # HARD 觸發最低信心（pilot 建議值）
CONFIDENCE_SOFT_THRESHOLD = 0.70      # SOFT 觸發最低信心（pilot 建議值）

# Layer 2：犯罪嚴重度分層
# HIGH   → 情緒平靜 HARD、前兆=0 HARD
# MEDIUM → 前兆 < 2 SOFT
# LOW 預謀型 → 前兆=0 SOFT（應有觀察期）
# LOW 突發型 → 事後反應=0 SOFT（不要求前兆，但應有事後反應）
HIGH_SEVERITY_CRIMES      = {"Shooting", "Robbery", "Assault"}
MEDIUM_SEVERITY_CRIMES    = {"Fighting", "Burglary", "Arson", "Explosion",
                              "Abuse", "Arrest", "Vandalism"}
LOW_INTENTIONAL_CRIMES    = {"Shoplifting", "Stealing"}   # 預謀型：需有觀察期前置
LOW_ACCIDENTAL_CRIMES     = {"RoadAccidents"}             # 突發型：不需前置，需事後反應

# 情緒升溫門檻說明：
# 參考 Kilic & Tuceryan (2024) §4.2：以 UCF-Crime 正常影片情緒特徵分布的
# 第 25 百分位數（≈ 0.15）作為「情緒顯著升溫」的最低偵測門檻。
# 低於此值視為情緒全程平靜（near-zero escalation）。
ESCALATION_CALM_THRESHOLD = 0.15      # 來源：Kilic & Tuceryan 2024，正常分布 P25


# ── 資料結構 ──────────────────────────────────────────────

@dataclass
class ConflictRecord:
    agent_a: str
    agent_b: str
    conflict_type: str     # category_mismatch / temporal_inconsistency /
                           # causal_conflict / spurious_correlation / missing_antecedent
    description: str
    severity: float        # [0, 1]
    frames_a: List[int] = field(default_factory=list)
    frames_b: List[int] = field(default_factory=list)


@dataclass
class ReflectorOutput:
    """Reflector 的完整審計輸出"""
    # 衝突分類
    conflict_type: str             # "HARD" | "SOFT" | "NONE"
    conflict_layer: str            # "TEMPORAL" | "CAUSAL" | "SPURIOUS" | "NONE"
    conflicts: List[ConflictRecord]

    # 共識
    consensus_category: str
    is_convergent: bool

    # 一致性分數（直接進入 Rcons 獎勵）
    rcons_score: float

    # HARD 衝突專用欄位（回傳 Planner）
    conflict_pairs: List[Dict] = field(default_factory=list)
    recommended_focus_frames: List[int] = field(default_factory=list)

    # SOFT 衝突專用欄位（直接傳給目標代理人）
    target_agent: str = ""
    soft_instruction: str = ""
    soft_focus_frames: List[int] = field(default_factory=list)

    # 追蹤
    retry_count: int = 0
    audit_log: str = ""


# ── ReflectorAgent ────────────────────────────────────────

class ReflectorAgent:
    """
    Reflector 不分析影像，只審計其他代理人的輸出。

    一致性分數公式：
      consistency_score = 1.0
        + 0.1  if 所有代理人引用幀有重疊
        + 0.1  if 因果鏈覆蓋 pre/during/post 三段
        - 1.0  for HARD_CONFLICT
        - 0.5  for SOFT_CONFLICT
        - 0.3  for SPURIOUS_CORRELATION
        - 0.2  per 無佐證幀引用
      clipped to [0.0, 1.0]
    """

    def __init__(self):
        self.name = "Reflector"
        self._debate_log: List[Dict] = []
        self._retry_counts: Dict[str, int] = {}   # agent_name → retry count

    def audit(
        self,
        reports: List[AgentReport],
        retry_count: int = 0,
    ) -> ReflectorOutput:
        """
        主要審計入口：
        1. 分離環境報告 / 分類報告
        2. 三層分析（L1+L2 Constraint Propagation，L3 CASAM）
        3. 衝突分類（HARD / SOFT / NONE）
        4. 計算 Rcons（Dempster-Shafer 衝突係數 K + 加法獎勵）
        5. 計算共識類別
        """
        cat_reports = [
            r for r in reports
            if r.crime_category != "ENVIRONMENTAL_ASSESSMENT"
        ]
        env_reports = [
            r for r in reports
            if r.crime_category == "ENVIRONMENTAL_ASSESSMENT"
        ]

        # ── CASAM 三層 ─────────────────────────────────
        l1_conflicts, hard_l1 = self._layer1_temporal(cat_reports)
        l2_conflicts, hard_l2 = self._layer2_causal(cat_reports)
        l3_conflicts           = self._layer3_spurious(cat_reports, env_reports)

        all_conflicts = l1_conflicts + l2_conflicts + l3_conflicts

        # ── 衝突分類 ───────────────────────────────────
        conflict_type, conflict_layer, target_agent, soft_instr, soft_frames = \
            self._classify_conflict(
                l1_conflicts, hard_l1,
                l2_conflicts, hard_l2,
                l3_conflicts,
                cat_reports, retry_count,
            )

        # ── 一致性分數 ─────────────────────────────────
        rcons = self._compute_consistency_score(
            cat_reports, all_conflicts, conflict_type
        )

        # ── 共識 ───────────────────────────────────────
        consensus, is_convergent = self._compute_consensus(cat_reports)

        # ── HARD 衝突詳情 ──────────────────────────────
        conflict_pairs, focus_frames = [], []
        if conflict_type == "HARD":
            conflict_pairs, focus_frames = self._build_hard_conflict_payload(
                l1_conflicts + l2_conflicts, cat_reports
            )

        audit_log = self._generate_audit_log(
            reports, all_conflicts, consensus, rcons, conflict_type
        )

        output = ReflectorOutput(
            conflict_type=conflict_type,
            conflict_layer=conflict_layer,
            conflicts=all_conflicts,
            consensus_category=consensus,
            is_convergent=is_convergent,
            rcons_score=rcons,
            conflict_pairs=conflict_pairs,
            recommended_focus_frames=focus_frames,
            target_agent=target_agent,
            soft_instruction=soft_instr,
            soft_focus_frames=soft_frames,
            retry_count=retry_count,
            audit_log=audit_log,
        )

        self._debate_log.append({
            "retry_count": retry_count,
            "conflict_type": conflict_type,
            "consensus": consensus,
            "rcons": rcons,
            "num_conflicts": len(all_conflicts),
            "is_convergent": is_convergent,
        })

        logger.info(
            f"[Reflector] {conflict_type} | rcons={rcons:.3f} | "
            f"consensus={consensus} | convergent={is_convergent}"
        )
        return output

    # ── Layer 1: Temporal Consistency [Constraint Propagation] ──

    def _layer1_temporal(
        self, reports: List[AgentReport]
    ) -> Tuple[List[ConflictRecord], bool]:
        """
        Constraint Propagation 規則引擎（取代全 CASAM）。

        幀取法（明確化）：
          攻擊幀 → Action Agent frame_references 的最小值
                   （= 第一個異常 snippet 的代表幀）
          升溫幀 → TimeEmotion metadata["escalation_start_frame"]（新增欄位）
                   fallback：掃描 emotion_trajectory，找第一個 angry+fear > 0.15 的幀
                   fallback：te_report.frame_references 的最小值

        Gap 門檻（fps-aware）：
          threshold_frames = int(TEMPORAL_GAP_SECONDS × video_fps)
          fps 優先從 te_report.metadata["video_fps"] 讀取，fallback = DEFAULT_VIDEO_FPS

        HARD 條件（ANY）：
          ① 攻擊幀 顯著早於 情緒升溫幀（gap > threshold_frames）
          ② 行為與情緒代理人引用幀無重疊 AND 時間差 > threshold_frames
          ③ 不同犯罪類別 AND 信心差 > CONFIDENCE_GAP_HARD
        """
        conflicts = []
        is_hard = False

        # ── 合併後：從單一 ActionEmotion Agent 讀取兩個值 ──────
        # 合併前：兩個獨立 Agent 互相驗證
        # 合併後：同一 Agent 內部的時序一致性（attack_frame vs escalation_start_frame）
        # 邏輯不變：驗證「模型預測的攻擊時間點」和「情緒升溫時間點」是否邏輯一致

        ae_report = next(
            (r for r in reports
             if "行為情緒" in r.agent_name or "ActionEmotion" in r.agent_name
             or ("行為" in r.agent_name and "情緒" in r.agent_name)),
            None,
        )

        if ae_report:
            fps = float(ae_report.metadata.get("video_fps", DEFAULT_VIDEO_FPS))
            threshold_frames = int(TEMPORAL_GAP_SECONDS * fps)

            act_frames = sorted(ae_report.frame_references)

            # 攻擊幀：ActionEmotion Agent evidence_frames 第一個
            attack_frame = act_frames[0] if act_frames else None

            # 升溫幀：同一 Agent 內部的 escalation_start_frame
            escalation_frame = ae_report.metadata.get("escalation_start_frame")

            if attack_frame is not None and escalation_frame is not None:
                gap = escalation_frame - attack_frame

                # HARD ①：攻擊幀顯著早於升溫幀（不合理：應先升溫再攻擊）
                if attack_frame < escalation_frame and gap > threshold_frames:
                    is_hard = True
                    conflicts.append(ConflictRecord(
                        agent_a=ae_report.agent_name,
                        agent_b=ae_report.agent_name,
                        conflict_type="temporal_inconsistency",
                        description=(
                            f"攻擊幀 {attack_frame} 比情緒升溫幀 {escalation_frame} "
                            f"早 {gap} 幀（閾值 {threshold_frames}，fps={fps:.0f}），"
                            "攻擊先於情緒升溫，內部時序不一致（L1 HARD）"
                        ),
                        severity=0.9,
                        frames_a=act_frames[:3],
                        frames_b=[escalation_frame],
                    ))

                # HARD ②：gap 過大（攻擊與情緒升溫相差太遠）
                if abs(gap) > threshold_frames * 3:
                    is_hard = True
                    conflicts.append(ConflictRecord(
                        agent_a=ae_report.agent_name,
                        agent_b=ae_report.agent_name,
                        conflict_type="temporal_inconsistency",
                        description=(
                            f"攻擊幀與情緒升溫幀相差 {abs(gap)} 幀"
                            f"（3× 閾值 {threshold_frames * 3}，fps={fps:.0f}），"
                            "行為與情緒時序嚴重不吻合"
                        ),
                        severity=0.7,
                        frames_a=act_frames[:3],
                        frames_b=[escalation_frame],
                    ))

        # HARD ③：類別直接矛盾（不含 Normal）
        non_normal = [r for r in reports if r.crime_category != "Normal"]
        if len(non_normal) >= 2:
            for i in range(len(non_normal)):
                for j in range(i + 1, len(non_normal)):
                    ri, rj = non_normal[i], non_normal[j]
                    if (ri.crime_category != rj.crime_category
                            and abs(ri.confidence - rj.confidence) > CONFIDENCE_GAP_HARD):
                        is_hard = True
                        conflicts.append(ConflictRecord(
                            agent_a=ri.agent_name,
                            agent_b=rj.agent_name,
                            conflict_type="category_mismatch",
                            description=(
                                f"{ri.agent_name} 判定 {ri.crime_category}（{ri.confidence:.2f}）"
                                f" vs {rj.agent_name} 判定 {rj.crime_category}（{rj.confidence:.2f}）"
                                f"，信心差 {abs(ri.confidence - rj.confidence):.2f} > {CONFIDENCE_GAP_HARD}"
                            ),
                            severity=0.85,
                            frames_a=ri.frame_references[:3],
                            frames_b=rj.frame_references[:3],
                        ))

        return conflicts, is_hard

    # ── Layer 2: Antecedent Analysis [Constraint Propagation] ──

    def _layer2_causal(
        self, reports: List[AgentReport]
    ) -> Tuple[List[ConflictRecord], bool]:
        """
        Constraint Propagation 規則引擎（取代全 CASAM）。

        完整觸發條件對照表：
        ┌──────────┬──────────────────────────────────────────┬────────┐
        │ Severity │ 條件                                     │ 結果   │
        ├──────────┼──────────────────────────────────────────┼────────┤
        │ HIGH     │ 情緒升溫 < 0.15 + 信心 > 0.75           │ HARD   │
        │ HIGH     │ 前置事件 = 0（信心 > 0.75）             │ HARD   │
        │ MEDIUM   │ 前置事件 < 2（信心 > 0.70）             │ SOFT   │
        │ LOW 預謀 │ 前置事件 = 0 + 信心 > 0.70             │ SOFT   │
        │ LOW 突發 │ 事後反應 = 0 + 信心 > 0.70             │ SOFT   │
        └──────────┴──────────────────────────────────────────┴────────┘

        LOW 預謀型（Shoplifting/Stealing）：應有觀察期前置，完全缺失才觸發。
        LOW 突發型（RoadAccidents）：突發事故不要求前置，但應有事後反應。
        """
        conflicts = []
        is_hard = False

        # 合併後：從單一 ActionEmotion Agent 讀取兩個值
        ae_report = next(
            (r for r in reports
             if "行為情緒" in r.agent_name or "ActionEmotion" in r.agent_name
             or ("行為" in r.agent_name and "情緒" in r.agent_name)),
            None,
        )

        # ── HARD ①：情緒平靜 + HIGH_SEVERITY ─────────────
        if ae_report:
            ae_meta    = ae_report.metadata or {}
            escalation = ae_meta.get("escalation_score")
            if (escalation is not None
                    and escalation < ESCALATION_CALM_THRESHOLD
                    and ae_report.crime_category in HIGH_SEVERITY_CRIMES
                    and ae_report.confidence > CONFIDENCE_HARD_THRESHOLD):
                is_hard = True
                conflicts.append(ConflictRecord(
                    agent_a=ae_report.agent_name,
                    agent_b="CausalRule",
                    conflict_type="causal_conflict",
                    description=(
                        f"情緒升溫值 {escalation:.3f} < {ESCALATION_CALM_THRESHOLD}"
                        f"（Kilic & Tuceryan 2024 P25 門檻），"
                        f"但 ActionEmotion 判定 {ae_report.crime_category}"
                        f"（信心 {ae_report.confidence:.2f} > {CONFIDENCE_HARD_THRESHOLD}），"
                        "缺乏情緒觸發前兆 → CP HARD"
                    ),
                    severity=0.9,
                    frames_a=ae_report.frame_references[:3],
                ))

        # ── HARD ②：HIGH_SEVERITY + 完全無前兆 ───────────
        for report in reports:
            if report.crime_category not in HIGH_SEVERITY_CRIMES:
                continue
            n_antecedents = self._count_antecedents(report)
            if n_antecedents == 0 and report.confidence > CONFIDENCE_HARD_THRESHOLD:
                is_hard = True
                conflicts.append(ConflictRecord(
                    agent_a=report.agent_name,
                    agent_b="AntecedentRule",
                    conflict_type="missing_antecedent",
                    description=(
                        f"HIGH_SEVERITY：{report.agent_name} 判定"
                        f" {report.crime_category}（信心 {report.confidence:.2f} > {CONFIDENCE_HARD_THRESHOLD}），"
                        "前置事件 = 0，高嚴重罪毫無前兆 → CP HARD"
                    ),
                    severity=0.85,
                    frames_a=report.frame_references[:3],
                ))

        # ── SOFT：MEDIUM_SEVERITY + 前兆不足（< 2）────────
        for report in reports:
            if report.crime_category not in MEDIUM_SEVERITY_CRIMES:
                continue
            n_antecedents = self._count_antecedents(report)
            if n_antecedents < 2 and report.confidence > CONFIDENCE_SOFT_THRESHOLD:
                conflicts.append(ConflictRecord(
                    agent_a=report.agent_name,
                    agent_b="AntecedentRule",
                    conflict_type="missing_antecedent",
                    description=(
                        f"MEDIUM_SEVERITY：{report.agent_name} 判定"
                        f" {report.crime_category}（信心 {report.confidence:.2f} > {CONFIDENCE_SOFT_THRESHOLD}），"
                        f"前置事件 {n_antecedents} < 2 → CP SOFT"
                    ),
                    severity=0.40,
                    frames_a=report.frame_references[:3],
                ))

        # ── SOFT：LOW 預謀型（Shoplifting/Stealing）+ 前置=0 ──
        for report in reports:
            if report.crime_category not in LOW_INTENTIONAL_CRIMES:
                continue
            n_antecedents = self._count_antecedents(report)
            if n_antecedents == 0 and report.confidence > CONFIDENCE_SOFT_THRESHOLD:
                conflicts.append(ConflictRecord(
                    agent_a=report.agent_name,
                    agent_b="AntecedentRule",
                    conflict_type="missing_antecedent",
                    description=(
                        f"LOW 預謀型：{report.agent_name} 判定"
                        f" {report.crime_category}（信心 {report.confidence:.2f}），"
                        "預謀竊盜應有觀察期前置，前置事件 = 0 → CP SOFT"
                    ),
                    severity=0.35,
                    frames_a=report.frame_references[:3],
                ))

        # ── SOFT：LOW 突發型（RoadAccidents）+ 事後反應=0 ──
        for report in reports:
            if report.crime_category not in LOW_ACCIDENTAL_CRIMES:
                continue
            n_post = self._count_post_crime_indicators(report, te_report)
            if n_post == 0 and report.confidence > CONFIDENCE_SOFT_THRESHOLD:
                conflicts.append(ConflictRecord(
                    agent_a=report.agent_name,
                    agent_b="AntecedentRule",
                    conflict_type="missing_antecedent",
                    description=(
                        f"LOW 突發型：{report.agent_name} 判定"
                        f" {report.crime_category}（信心 {report.confidence:.2f}），"
                        "突發事故不要求前兆，但事後反應 = 0 → CP SOFT"
                    ),
                    severity=0.35,
                    frames_a=report.frame_references[:3],
                ))

        return conflicts, is_hard

    @staticmethod
    def _count_antecedents(report: AgentReport) -> int:
        """計算報告中前兆類型 evidence 的數量。"""
        PRE_CRIME_TYPES = {
            "pre_crime_segments", "causal_chain",
            "pre_crime_indicators", "targeted_causal_evidence",
        }
        return sum(
            1 for e in report.evidence
            if e.get("type") in PRE_CRIME_TYPES
            or (e.get("type") == "targeted_causal_evidence"
                and e.get("phase") == "pre_crime")
        )

    @staticmethod
    def _count_post_crime_indicators(
        report: AgentReport,
        te_report: Optional[AgentReport],
    ) -> int:
        """
        計算事後反應指標數量（供 LOW 突發型判斷）。
        優先讀 TimeEmotion 新增的 post_crime_indicators 欄位，
        fallback 到因果鏈的 post_crime 階段。
        """
        # TimeEmotion 新欄位（Section IV 更新後）
        if te_report:
            post_list = te_report.metadata.get("post_crime_indicators", None)
            if post_list is not None:
                return len(post_list)
            # fallback：掃描 causal_chain_raw 的 post_crime 階段
            chain = te_report.metadata.get("causal_chain_raw", [])
            return sum(1 for e in chain if e.get("phase") == "post_crime")
        # 若無 TimeEmotion，嘗試在 report 自身找 post_crime evidence
        return sum(
            1 for e in report.evidence
            if (e.get("type") == "targeted_causal_evidence"
                and e.get("phase") == "post_crime")
        )

    # ── Layer 3: Spurious Correlation [CASAM] ────────────

    def _layer3_spurious(
        self,
        cat_reports: List[AgentReport],
        env_reports: List[AgentReport],
    ) -> List[ConflictRecord]:
        """
        保留 CASAM 注意力稀疏化機制，此層真正擅長的偽相關排除面向。
        永遠 SOFT（不升 HARD），不影響主要衝突分類。

        更新：直接讀取 Environment Agent 輸出的 estimated_env_contribution
              （with/without 環境特徵的異常分數差異，由 EnvironmentAgent 計算）
              → 降低 Rcost，避免 Reflector 重新推論信心降幅

        fallback：若 estimated_env_contribution 欄位不存在，
                  沿用原估算公式 (report.confidence - env_conf) × 0.5
        """
        conflicts = []
        if not env_reports:
            return conflicts

        env_report = env_reports[0]
        env_conf   = env_report.confidence

        for report in cat_reports:
            if env_conf >= 0.4:
                continue    # 環境品質可接受，無需偵測偽相關

            if report.confidence <= 0.75:
                continue    # 分類信心不足，不觸發

            # 優先讀 EnvironmentAgent 直接計算的貢獻度（新欄位）
            env_contribution = env_report.metadata.get("estimated_env_contribution")
            if env_contribution is not None:
                estimated_drop = float(env_contribution)
                source = "Environment Agent 直接計算"
            else:
                # fallback：原估算公式（保留向後相容）
                estimated_drop = (report.confidence - env_conf) * 0.5
                source = "估算（fallback）"

            if estimated_drop > 0.3:
                conflicts.append(ConflictRecord(
                    agent_a=report.agent_name,
                    agent_b=env_report.agent_name,
                    conflict_type="spurious_correlation",
                    description=(
                        f"環境可信度 {env_conf:.2f} < 0.4，"
                        f"{report.agent_name} 信心 {report.confidence:.2f}，"
                        f"排除環境脈絡後預估降幅 {estimated_drop:.2f} > 0.3"
                        f"（{source}）→ CASAM SOFT"
                    ),
                    severity=0.4,
                ))

        return conflicts

    # ── 衝突分類 ─────────────────────────────────────────

    def _classify_conflict(
        self,
        l1_conflicts: List[ConflictRecord], hard_l1: bool,
        l2_conflicts: List[ConflictRecord], hard_l2: bool,
        l3_conflicts: List[ConflictRecord],
        reports: List[AgentReport],
        retry_count: int,
    ) -> Tuple[str, str, str, str, List[int]]:
        """
        回傳 (conflict_type, conflict_layer, target_agent, soft_instruction, soft_frames)
        """
        # retry >= 2 → 強制升級 HARD
        if retry_count >= 2 and (l1_conflicts or l2_conflicts or l3_conflicts):
            layer = "TEMPORAL" if l1_conflicts else ("CAUSAL" if l2_conflicts else "SPURIOUS")
            return "HARD", layer, "", "", []

        # HARD conditions
        if hard_l1:
            return "HARD", "TEMPORAL", "", "", []
        if hard_l2:
            return "HARD", "CAUSAL", "", "", []

        # SOFT conditions
        all_soft = l2_conflicts + l3_conflicts

        # 低信心但同方向 → SOFT
        for r1, r2 in self._pairs(reports):
            if (r1.crime_category == r2.crime_category
                    and r1.confidence < CONFIDENCE_SOFT_MAX
                    and r2.confidence < CONFIDENCE_SOFT_MAX):
                weaker = r1 if r1.confidence < r2.confidence else r2
                focus = weaker.frame_references[:5]
                instr = (
                    f"Your confidence is {weaker.confidence:.2f} on "
                    f"{weaker.crime_category} detection. "
                    f"Provide binary yes/no confirmation for frames "
                    f"{focus} specifically."
                )
                return "SOFT", "NONE", weaker.agent_name, instr, focus

        if l3_conflicts:
            target = l3_conflicts[0].agent_a
            frames = l3_conflicts[0].frames_a or []
            instr = (
                f"Reanalyze frames {frames}. "
                "Exclude shadow regions and lighting artifacts from your analysis. "
                "Focus only on human motion patterns."
            )
            return "SOFT", "SPURIOUS", target, instr, frames

        if all_soft:
            c = all_soft[0]
            focus = c.frames_a[:5] or []
            instr = (
                f"Your analysis of frames {focus} requires clarification: "
                f"{c.description}. "
                "Reanalyze only these specific frames."
            )
            return "SOFT", "CAUSAL", c.agent_a, instr, focus

        return "NONE", "NONE", "", "", []

    # ── HARD 衝突 Payload ─────────────────────────────────

    def _build_hard_conflict_payload(
        self, conflicts: List[ConflictRecord], reports: List[AgentReport]
    ) -> Tuple[List[Dict], List[int]]:
        pairs = []
        focus_set = set()
        for c in conflicts:
            pairs.append({
                "agent_a": c.agent_a,
                "claim_a": self._get_claim(c.agent_a, reports),
                "frames_a": c.frames_a,
                "agent_b": c.agent_b,
                "claim_b": self._get_claim(c.agent_b, reports),
                "frames_b": c.frames_b,
                "conflict_description": c.description,
            })
            focus_set.update(c.frames_a)
            focus_set.update(c.frames_b)
        return pairs, sorted(focus_set)[:10]

    def _get_claim(self, agent_name: str, reports: List[AgentReport]) -> str:
        for r in reports:
            if r.agent_name == agent_name:
                return f"{r.crime_category}（信心 {r.confidence:.2f}）"
        return agent_name

    # ── Rcons：Dempster-Shafer 衝突係數 ──────────────────

    def _compute_consistency_score(
        self,
        reports: List[AgentReport],
        conflicts: List[ConflictRecord],
        conflict_type: str,
    ) -> float:
        """
        Dempster-Shafer 衝突係數 K（取代原乘法懲罰公式）。

        信念質量函數（簡化版，假設 frame_references 數量代表證據強度）：
          uncertainty = 0.30 if frames == 0
                      = 0.15 if 1 ≤ frames ≤ 2
                      = 0.05 if frames ≥ 3
          m(crime)    = confidence × (1 − uncertainty)
          m(uncertain)= uncertainty
          m(no_crime) = max(0, 1 − m(crime) − m(uncertain))

        Pairwise 衝突係數 K_ij：
          若同犯罪類別：K_ij = m_i(crime)·m_j(no_crime) + m_i(no_crime)·m_j(crime)
          若不同類別：   K_ij += m_i(crime)·m_j(crime)   ← 互斥假設額外衝突

        K = mean(K_ij) over all pairs
        Rcons_base = 1.0 − K

        合併後架構（ActionEmotion + Environment）：
          合併前：K 量化 Action vs TimeEmotion 兩個獨立模態的信念衝突
          合併後：K 量化 ActionEmotion vs Environment 的信念衝突
                  → 環境條件支不支撐行為分析的結論
          無 Environment Agent 時：
                  使用預設信念函數 m_env（假設環境完全支持 AE 的判斷）
                  → K ≈ 0，Rcons_base ≈ 1.0（由加分項決定最終分數）

        加分項（設計規則，與 D-S 互補）：
          +0.2  ActionEmotion 幀引用與 Environment 幀引用有非空交集
          +0.2  因果鏈覆蓋 pre/during/post 三段
          +0.1  Rlegal > 0.8（由 Planner 計算後注入 ae_report.metadata）

        Rcons_final = clip(Rcons_base + bonus, 0.0, 1.3)
        """
        cat_reports = [r for r in reports if r.crime_category != "ENVIRONMENTAL_ASSESSMENT"]
        env_reports = [r for r in reports if r.crime_category == "ENVIRONMENTAL_ASSESSMENT"]

        # ── 信念質量函數 ──────────────────────────────────
        def _mass(report: AgentReport, is_env: bool = False) -> Dict[str, float]:
            n_frames = len(report.frame_references)
            if n_frames == 0:
                u = 0.30
            elif n_frames <= 2:
                u = 0.15
            else:
                u = 0.05
            if is_env:
                # Environment Agent：高可信度 → 支持 AE 判斷；低可信度 → 不確定增加
                env_conf = report.confidence
                m_crime  = env_conf * (1.0 - u)           # 環境支持犯罪判斷的程度
                m_unc    = u + (1.0 - env_conf) * 0.3     # 可信度低 → 不確定增加
                m_no     = max(0.0, 1.0 - m_crime - m_unc)
            else:
                m_crime = report.confidence * (1.0 - u)
                m_unc   = u
                m_no    = max(0.0, 1.0 - m_crime - m_unc)
            return {
                "crime":    m_crime,
                "no_crime": m_no,
                "uncertain": m_unc,
                "category": report.crime_category,
            }

        # ── Pairwise 衝突係數 K_ij ──────────────────────
        # 合併後：ActionEmotion × Environment（而非 Action × TimeEmotion）
        k_values: List[float] = []

        if cat_reports and env_reports:
            # 有 Environment Agent：計算 AE × Env 的 K
            for ae_r in cat_reports:
                for env_r in env_reports:
                    mi = _mass(ae_r, is_env=False)
                    mj = _mass(env_r, is_env=True)
                    # 基礎衝突：crime × no_crime
                    k_ij = mi["crime"] * mj["no_crime"] + mi["no_crime"] * mj["crime"]
                    k_values.append(min(k_ij, 1.0))
        elif cat_reports:
            # 無 Environment Agent：使用預設信念函數（環境完全支持 AE 判斷）
            for ae_r in cat_reports:
                mi = _mass(ae_r, is_env=False)
                # m_env_default：假設環境可信度 = ae_conf（無獨立資訊，不增加衝突）
                m_env_default = {
                    "crime":    ae_r.confidence * 0.95,
                    "no_crime": 0.03,
                    "uncertain": 0.02 + (1.0 - ae_r.confidence) * 0.1,
                    "category": ae_r.crime_category,
                }
                k_ij = mi["crime"] * m_env_default["no_crime"] + mi["no_crime"] * m_env_default["crime"]
                k_values.append(min(k_ij, 1.0))

        K = float(sum(k_values) / len(k_values)) if k_values else 0.0
        rcons_base = 1.0 - K

        # ── 加分項（保留設計規則，與 D-S 互補）──────────
        bonus = 0.0

        # +0.2 AE 幀引用與 Env 幀引用有非空交集
        all_frame_sets = [set(r.frame_references) for r in reports if r.frame_references]
        if len(all_frame_sets) >= 2:
            intersection = all_frame_sets[0]
            for fs in all_frame_sets[1:]:
                intersection &= fs
            if intersection:
                bonus += 0.2

        # +0.2 因果鏈覆蓋三段（ActionEmotion 的 metadata）
        has_causal_chain = any(
            bool(r.metadata.get("causal_chain"))
            and bool(r.metadata.get("pre_crime_indicators"))
            and bool(r.metadata.get("post_crime_indicators"))
            for r in cat_reports
        )
        if has_causal_chain:
            bonus += 0.2

        # +0.1 Rlegal > 0.8（由 Planner 在 Step 3c 計算後注入 ae_report.metadata）
        for r in cat_reports:
            if r.metadata.get("rlegal", 0.0) > 0.8:
                bonus += 0.1
                break

        score = rcons_base + bonus
        logger.debug(
            f"[Reflector] D-S: K={K:.3f} → Rcons_base={rcons_base:.3f} "
            f"+ bonus={bonus:.2f} = {score:.3f}"
        )
        return float(max(0.0, min(1.3, score)))

    # ── 共識計算 ─────────────────────────────────────────

    def _compute_consensus(
        self, reports: List[AgentReport]
    ) -> Tuple[str, bool]:
        if not reports:
            return "Normal", False
        votes: Dict[str, float] = {}
        for r in reports:
            votes[r.crime_category] = votes.get(r.crime_category, 0.0) + r.confidence
        consensus = max(votes, key=lambda k: votes[k])
        total = sum(votes.values())
        ratio = votes[consensus] / total if total > 0 else 0.0
        return consensus, ratio >= cfg.debate.consensus_threshold

    # ── 輔助 ─────────────────────────────────────────────

    def _pairs(self, reports: List[AgentReport]):
        for i in range(len(reports)):
            for j in range(i + 1, len(reports)):
                yield reports[i], reports[j]

    def _generate_audit_log(
        self,
        reports: List[AgentReport],
        conflicts: List[ConflictRecord],
        consensus: str,
        rcons: float,
        conflict_type: str,
    ) -> str:
        lines = [
            "=== Reflector 審計報告（L1+L2: CP，L3: CASAM，Rcons: D-S）===",
            f"代理人：{', '.join(r.agent_name for r in reports)}",
            f"衝突類型：{conflict_type} | 共識：{consensus} | Rcons（D-S）：{rcons:.3f}",
            f"衝突數量：{len(conflicts)}",
        ]
        for c in conflicts:
            lines.append(
                f"  [{c.conflict_type.upper()}] {c.agent_a} ↔ {c.agent_b} "
                f"severity={c.severity:.2f}: {c.description}"
            )
        return "\n".join(lines)

    def get_debate_log(self) -> List[Dict]:
        return self._debate_log

    def reset(self):
        self._debate_log = []
        self._retry_counts = {}
