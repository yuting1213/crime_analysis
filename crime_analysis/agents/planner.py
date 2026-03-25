"""
Planner Agent - 中央規劃與協調代理（規則式）

執行流程（三步驟）：
  Step 1 (條件式): Environment Check（video_quality < 0.6 或 occlusion_detected）
  Step 2 (循序式): ActionEmotion Agent（行為 + 情緒合併分析）
  Step 3 (整合)  : RAG 查詢 + Rlegal 計算 + 報告生成（Planner 直接負責）

衝突解決（Reflector CASAM）：
  SOFT → 直接傳回修正指令給 ActionEmotion Agent refine
  HARD → Planner 重新指派 ActionEmotion Agent re-analyze
  NONE → 直接整合最終報告

訓練：DPO 對齊報告品質（取代 GRPO online RL）
評估：R = w1·Racc + w2·Rcons + w3·Rlegal − w4·Rcost（評估指標，非訓練信號）
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .base_agent import AgentReport
from .reflector import ReflectorAgent, ReflectorOutput
from config import cfg

logger = logging.getLogger(__name__)

# ── 評估權重（依犯罪類型，供 evaluate() 使用）────────────────
EVAL_WEIGHTS = {
    "violent":      {"w1": 0.45, "w2": 0.20, "w3": 0.25, "w4": 0.10},
    "property":     {"w1": 0.25, "w2": 0.20, "w3": 0.45, "w4": 0.10},
    "public_safety":{"w1": 0.30, "w2": 0.25, "w3": 0.35, "w4": 0.10},
    "default":      {"w1": 0.35, "w2": 0.20, "w3": 0.35, "w4": 0.10},
}
VIOLENT_CRIMES       = {"Fighting", "Assault", "Shooting"}
PROPERTY_CRIMES      = {"Robbery", "Burglary", "Stealing", "Shoplifting"}
PUBLIC_SAFETY_CRIMES = {"Arson", "Explosion", "Vandalism"}
HIGH_SEVERITY_CRIMES = {"Shooting", "Robbery", "Assault"}


def compute_rcost(
    total_turns: int,
    threshold_low: int = 3,
    threshold_high: int = 8,
) -> float:
    """
    Rcost 分段懲罰（評估用）：
      turns ≤ low         → 0
      low < turns ≤ high  → 線性 [0, 1]
      turns > high        → 1.0
    """
    if total_turns <= threshold_low:
        return 0.0
    span = threshold_high - threshold_low
    if span <= 0:
        return 1.0
    elif total_turns <= threshold_high:
        return (total_turns - threshold_low) / span
    return 1.0


class PlannerAgent:
    """
    規則式 Planner（不繼承 BaseAgent，角色是協調者）

    agents_dict 格式：
    {
        "environment":    EnvironmentAgent（可選）,
        "action_emotion": ActionEmotionAgent,
    }
    rag_module: RAGModule 實例（共享 RAG 知識庫）
    """

    def __init__(
        self,
        agents_dict: Dict[str, Any],
        reflector: ReflectorAgent,
        rag_module=None,
    ):
        self.agents = agents_dict
        self.reflector = reflector
        self.rag = rag_module          # RAGModule（由 pipeline 注入）
        self._total_turns: int = 0
        self._rcost_threshold_high: int = 8

    # ── 主要入口 ─────────────────────────────────────────

    def run(self, frames: List, video_metadata: Dict) -> Dict[str, Any]:
        """
        規則式三步驟流程：
        Step 1 → Step 2 → Reflector + 衝突解決 → 最終報告
        """
        self._total_turns = 0
        self.reflector.reset()

        case_id = video_metadata.get("case_id", "unknown")
        reports: Dict[str, AgentReport] = {}

        # ── Step 1：Environment Check（條件式）──────────
        env_confidence = 1.0
        reports, env_confidence = self._step1_environment(frames, video_metadata, reports)

        low_reliability = env_confidence < 0.4
        if low_reliability:
            logger.warning(f"[Planner] env_confidence={env_confidence:.2f} → LOW_RELIABILITY")

        # ── Step 2：ActionEmotion 分析 ───────────────────
        human_interaction = video_metadata.get("human_interaction", True)
        if human_interaction:
            reports = self._step2_action_emotion(frames, video_metadata, reports)
        else:
            logger.info("[Planner] human_interaction=False → 跳過 ActionEmotion")

        ae_report = reports.get("action_emotion")
        ae_confidence = ae_report.confidence if ae_report else 0.0

        # 信心不足：嘗試 refine
        if ae_report and 0.4 <= ae_confidence < 0.6:
            logger.info(f"[Planner] conf={ae_confidence:.2f} < 0.6，嘗試 ActionEmotion refine")
            ae_agent = self.agents.get("action_emotion")
            if ae_agent:
                ae_report = ae_agent.refine(list(reports.values()))
                reports["action_emotion"] = ae_report
                ae_confidence = ae_report.confidence
                self._total_turns += 1

        if ae_confidence < 0.4:
            logger.info(f"[Planner] conf={ae_confidence:.2f} < 0.4 → 跳過 RAG 查詢")

        # ── Step 3：法律整合（3a RAG → 3b 報告生成 → 3c Rlegal）──
        # 職責邊界：RAGModule 是工具；Planner（Qwen3-7B）是推理引擎；兩者不混在一起。
        crime_type = ae_report.crime_category if ae_report else "Normal"
        rag_results: Dict = {"laws": [], "judgments": []}
        rlegal = 0.0

        # Step 3a：RAG 查詢取候選法條
        if ae_confidence >= 0.4 and self.rag and crime_type != "Normal":
            rationale = ae_report.metadata.get("rationale", "") if ae_report else ""
            query_text = (
                self.rag.generate_hypothetical_doc(rationale, crime_type)
                if hasattr(self.rag, "generate_hypothetical_doc") else rationale
            )
            rag_results = self.rag.query(query_text, query_type="semantic")
            self._total_turns += 1
            logger.info(f"[Planner] Step 3a: RAG 查詢 → {len(rag_results.get('laws', []))} 條法條")

        # Step 3b：Planner（Qwen3-7B）整合所有資訊生成報告文字
        # TODO: 此處呼叫 Qwen3-7B，傳入 ae_report.rationale + rag_results + conflict_type
        #       生成完整鑑定報告文字並儲存在 generated_report_text
        #       目前以 ae_report.rationale 作為 proxy（DPO 訓練後替換）
        generated_report_text = ae_report.metadata.get("rationale", "") if ae_report else ""

        # Step 3c：compute_rlegal 在報告文字生成後才呼叫
        # 重要：Rlegal 比對的是「Step 3b 生成的最終報告文字」，不是 AE 的 rationale
        # ── 注意：rlegal 是 Planner 計算後注入 ae_report.metadata 的 ──
        #    不是 ActionEmotionAgent 自行輸出的欄位。
        #    ae_report.metadata["rlegal"] 只有在 Planner Step 3c 執行後才存在。
        if self.rag and crime_type != "Normal":
            rlegal = self.rag.compute_rlegal(crime_type, generated_report_text)
            if ae_report:
                # Planner 注入（非 ActionEmotionAgent 輸出）
                # 供 Reflector _compute_consistency_score 的加分項 +0.1 讀取
                ae_report.metadata["rlegal"] = rlegal
            logger.info(f"[Planner] Step 3c: Rlegal={rlegal:.3f} ({crime_type})")

        # ── Reflector CASAM + 衝突解決 ──────────────────
        max_reassign = 2 if crime_type in HIGH_SEVERITY_CRIMES else 1
        rcost_high = 10 if crime_type in HIGH_SEVERITY_CRIMES else 8
        self._rcost_threshold_high = rcost_high

        all_reports = list(reports.values())
        final_audit = self._resolve_conflicts(
            frames, video_metadata, all_reports,
            max_reassignment=max_reassign,
        )

        # ── 最終報告整合 ─────────────────────────────────
        return self._synthesize_final_report(
            list(reports.values()), final_audit,
            case_id, rlegal, rag_results, low_reliability,
        )

    # ── Step 1 ───────────────────────────────────────────

    def _step1_environment(
        self,
        frames: List,
        video_metadata: Dict,
        reports: Dict[str, AgentReport],
    ) -> tuple:
        env_agent = self.agents.get("environment")
        quality = video_metadata.get("video_quality", 1.0)
        occlusion = video_metadata.get("occlusion_detected", False)

        if (quality < 0.6 or occlusion) and env_agent:
            logger.info(f"[Planner] Step 1: Environment Check (q={quality:.2f}, occ={occlusion})")
            env_report = env_agent.analyze(frames, video_metadata)
            reports["environment"] = env_report
            self._total_turns += 1
            return reports, env_report.confidence

        return reports, 1.0

    # ── Step 2 ───────────────────────────────────────────

    def _step2_action_emotion(
        self,
        frames: List,
        video_metadata: Dict,
        reports: Dict[str, AgentReport],
    ) -> Dict[str, AgentReport]:
        ae_agent = self.agents.get("action_emotion")
        if not ae_agent:
            logger.warning("[Planner] ActionEmotion Agent 未設定")
            return reports

        logger.info("[Planner] Step 2: ActionEmotion 分析")
        try:
            ae_report = ae_agent.analyze(frames, video_metadata)
            reports["action_emotion"] = ae_report
            self._total_turns += 1
            logger.info(
                f"  [action_emotion] {ae_report.crime_category} "
                f"conf={ae_report.confidence:.2f} "
                f"escalation={ae_report.metadata.get('escalation_score', 0.0):.2f}"
            )
        except Exception as e:
            logger.error(f"[Planner] ActionEmotion 分析失敗：{e}")

        return reports

    # ── 衝突解決 ─────────────────────────────────────────

    def _resolve_conflicts(
        self,
        frames: List,
        video_metadata: Dict,
        all_reports: List[AgentReport],
        max_reassignment: int = 1,
    ) -> ReflectorOutput:
        retry_count = 0

        while True:
            audit = self.reflector.audit(all_reports, retry_count=retry_count)

            if audit.conflict_type == "NONE":
                logger.info("[Planner] Reflector: NONE，進入最終整合")
                return audit

            elif audit.conflict_type == "SOFT":
                logger.info(f"[Planner] SOFT → {audit.target_agent} refine（retry={retry_count}）")
                target = self._find_agent(audit.target_agent)
                if target and retry_count < 2:
                    others = [r for r in all_reports if r.agent_name != audit.target_agent]
                    updated = target.refine(others)
                    all_reports = [updated if r.agent_name == audit.target_agent else r
                                   for r in all_reports]
                    self._total_turns += 1
                    retry_count += 1
                else:
                    return audit

            elif audit.conflict_type == "HARD":
                logger.warning(f"[Planner] HARD ({audit.conflict_layer}) → re-analyze（retry={retry_count}/{max_reassignment}）")
                if retry_count >= max_reassignment:
                    logger.warning("[Planner] 達到最大重新指派次數，標記 INSUFFICIENT_EVIDENCE")
                    return audit

                ae_agent = self.agents.get("action_emotion")
                if ae_agent and hasattr(ae_agent, "analyze"):
                    conflict_meta = dict(video_metadata)
                    conflict_meta.update({
                        "reassignment_reason": "HARD_CONFLICT",
                        "conflict_context": audit.conflict_layer,
                        "focus_frames": audit.recommended_focus_frames,
                    })
                    try:
                        updated = ae_agent.analyze(frames, conflict_meta)
                        all_reports = [updated if "action_emotion" in r.agent_name.lower()
                                       or "行為情緒" in r.agent_name else r
                                       for r in all_reports]
                        self._total_turns += 2
                    except Exception as e:
                        logger.error(f"[Planner] re-analyze 失敗：{e}")

                retry_count += 1

    # ── 最終報告整合 ─────────────────────────────────────

    def _synthesize_final_report(
        self,
        reports: List[AgentReport],
        final_audit: ReflectorOutput,
        case_id: str,
        rlegal: float,
        rag_results: Dict,
        low_reliability: bool,
    ) -> Dict[str, Any]:
        crime_type = final_audit.consensus_category
        weights = self._get_eval_weights(crime_type)

        ae_report = next(
            (r for r in reports
             if "行為情緒" in r.agent_name or "ActionEmotion" in r.agent_name),
            None,
        )
        env_report = next(
            (r for r in reports if r.crime_category == "ENVIRONMENTAL_ASSESSMENT"),
            None,
        )

        # 環境可信度低 → 加重 Rcons 權重
        if env_report and env_report.confidence < 0.4:
            weights = dict(weights)
            weights["w2"] = min(weights["w2"] + 0.05, 0.30)
            weights["w4"] = max(weights["w4"] - 0.05, 0.05)

        rcost = compute_rcost(
            self._total_turns,
            threshold_low=3,
            threshold_high=self._rcost_threshold_high,
        )
        rcons = final_audit.rcons_score

        # 法條引用
        applicable_articles = (
            [l.get("article_id", "") for l in rag_results.get("laws", [])[:5]]
            if rag_results else []
        )

        # 不確定性說明
        uncertainty_notes = self._build_uncertainty_notes(
            reports, final_audit, low_reliability
        )

        # 因果鏈 / 時間軸
        causal_chain = ae_report.metadata.get("causal_chain", "") if ae_report else ""
        escalation_score = ae_report.metadata.get("escalation_score", 0.0) if ae_report else 0.0
        rationale = ae_report.metadata.get("rationale", "") if ae_report else ""

        # Rlegal 元素（從 RAGModule 查表）
        elements_covered = []
        if self.rag:
            elements_covered = self.rag.get_legal_elements(crime_type)

        final_report = {
            "case_id": case_id,
            "fact_finding": {
                "description": rationale or (ae_report.reasoning if ae_report else "分析不完整"),
                "supporting_frames": self._collect_key_frames(reports),
                "confidence": ae_report.confidence if ae_report else 0.0,
            },
            "behavior_analysis": {
                "causal_chain": causal_chain,
                "escalation_score": escalation_score,
                "pre_crime_indicators": ae_report.metadata.get("pre_crime_indicators", []) if ae_report else [],
                "post_crime_indicators": ae_report.metadata.get("post_crime_indicators", []) if ae_report else [],
                "crime_group": ae_report.metadata.get("crime_group", "") if ae_report else "",
                "severity": ae_report.metadata.get("crime_type_severity", "") if ae_report else "",
            },
            "legal_classification": {
                "applicable_articles": applicable_articles,
                "elements_covered": elements_covered,
                "coverage_rate": rlegal,
                "rag_laws": rag_results.get("laws", [])[:3],
            },
            "uncertainty_notes": uncertainty_notes,
            # 評估指標（非訓練信號）
            "rcons":  rcons,
            "rlegal": rlegal,
            "rcost":  rcost,
            "eval_weights": weights,
            "total_turns": self._total_turns,
            "final_category": crime_type,
            "is_convergent": final_audit.is_convergent,
            "conflict_type": final_audit.conflict_type,
            # 完整代理人輸出
            "agent_reports": [r.to_dict() for r in reports],
            "audit_log": final_audit.audit_log,
            "debate_log": self.reflector.get_debate_log(),
        }

        logger.info(
            f"[Planner] 最終報告 | case={case_id} | category={crime_type} | "
            f"rcons={rcons:.3f} | rlegal={rlegal:.3f} | rcost={rcost:.3f} | turns={self._total_turns}"
        )
        return final_report

    # ── 評估函數（取代 GRPO 訓練信號）──────────────────────

    def evaluate(self, result: Dict, ground_truth: str) -> Dict[str, float]:
        """
        四指標評估框架（測試階段使用，非訓練信號）：
        R = w1·Racc + w2·Rcons + w3·Rlegal − w4·Rcost
        """
        racc  = 1.0 if result.get("final_category") == ground_truth else 0.0
        rcons  = result.get("rcons", 0.0)
        rlegal = result.get("rlegal", 0.0)
        rcost  = result.get("rcost", 0.0)
        w = result.get("eval_weights", EVAL_WEIGHTS["default"])

        total = (
            w["w1"] * racc
            + w["w2"] * rcons
            + w["w3"] * rlegal
            - w["w4"] * rcost
        )
        return {
            "R": total,
            "Racc": racc,
            "Rcons": rcons,
            "Rlegal": rlegal,
            "Rcost": rcost,
        }

    # ── 輔助方法 ─────────────────────────────────────────

    def _find_agent(self, agent_name: str):
        name_lower = agent_name.lower()
        for role, agent in self.agents.items():
            if (role in name_lower or name_lower in role
                    or agent_name == getattr(agent, "name", "")):
                return agent
        return None

    def _get_eval_weights(self, crime_type: str) -> Dict[str, float]:
        if crime_type in VIOLENT_CRIMES:
            return EVAL_WEIGHTS["violent"]
        if crime_type in PROPERTY_CRIMES:
            return EVAL_WEIGHTS["property"]
        if crime_type in PUBLIC_SAFETY_CRIMES:
            return EVAL_WEIGHTS["public_safety"]
        return EVAL_WEIGHTS["default"]

    def _collect_key_frames(self, reports: List[AgentReport]) -> List[int]:
        frames: set = set()
        for r in reports:
            frames.update(r.frame_references)
        return sorted(frames)

    def _build_uncertainty_notes(
        self,
        reports: List[AgentReport],
        audit: ReflectorOutput,
        low_reliability: bool,
    ) -> Dict[str, List[str]]:
        low_conf, conflicting, insufficient = [], [], []

        for r in reports:
            if r.confidence < 0.6 and r.crime_category != "ENVIRONMENTAL_ASSESSMENT":
                low_conf.append(f"{r.agent_name}: conf={r.confidence:.2f}")
            for flag in r.conflict_flags:
                conflicting.append(f"{r.agent_name}: {flag}")

        if low_reliability:
            insufficient.append("環境可信度過低，視覺證據品質不足")

        if audit.conflict_type == "HARD" and not audit.is_convergent:
            insufficient.append(
                f"HARD 衝突無法解決（{audit.conflict_layer}），結論可信度受影響"
            )

        return {
            "low_confidence_items": low_conf,
            "conflicting_evidence": conflicting,
            "insufficient_evidence": insufficient,
        }

    # ── 向後相容 ─────────────────────────────────────────

    @classmethod
    def from_list(
        cls,
        local_agents: List,
        reflector: ReflectorAgent,
        rag_module=None,
    ) -> "PlannerAgent":
        role_map: Dict[str, Any] = {}
        for agent in local_agents:
            name = getattr(agent, "name", "")
            if "環境" in name or "environment" in name.lower():
                role_map["environment"] = agent
            elif "行為情緒" in name or "actionemotion" in name.lower():
                role_map["action_emotion"] = agent
        return cls(role_map, reflector, rag_module)
