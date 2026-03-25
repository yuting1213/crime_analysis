"""
Planner Agent - 中央規劃與協調代理
基於：設計規格 + GRPO 強化學習

執行流程（四步驟）：
  Step 1  (條件式): Environment Check（video_quality < 0.6 或 occlusion_detected）
  Step 2  (並行式): Action + Time&Emotion（human_interaction == True）
  Step 2b (新增)  : Semantic Agent 初步法律框架
                    → 根據 Step 2 結果確定候選罪名與需核查的構成要件
                    → 回傳 {candidate_articles, key_elements_to_verify}
                    → 注入 Action / TimeEmotion 的 _legal_framework
  Step 3  (新增)  : Action + TimeEmotion 針對性佐證補充
                    → 依 legal_framework 重新 refine，補充對應構成要件的視覺佐證
  Step 4  (循序式): Semantic Agent 最終法律定性
                    → 以補強後的完整報告做最終法條引用與 Rlegal 計算

衝突解決：
  SOFT → Reflector 直接傳回修正指令給目標代理人，重新 refine
  HARD → Planner 以衝突脈絡重新指派衝突代理人 re-analyze
  NONE → 直接整合最終報告

獎勵函數：R = w1*Racc + w2*Rcons + w3*Rlegal - w4*Rcost
動態權重：依犯罪類型調整（暴力 vs. 財產型）
"""
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .base_agent import AgentReport
from .reflector import ReflectorAgent, ReflectorOutput
from config import cfg

logger = logging.getLogger(__name__)

# ── 動態獎勵權重（依犯罪類型）────────────────────────────
# 設計原則：
#   violent  → Racc 高（辨識精準優先），Rlegal 次之
#   property → Rlegal 高（法律構成要件複雜），Racc 次之
#   public_safety → Rlegal 高（縱火/爆炸環境影響大），Rcons 略高
#   default  → Racc = Rlegal 並列（司法應用核心：辨識準確與法律正確同等重要）
DYNAMIC_WEIGHTS = {
    "violent": {           # Fighting, Assault, Shooting
        "w1": 0.45, "w2": 0.20, "w3": 0.25, "w4": 0.10,
    },
    "property": {          # Robbery, Burglary, Stealing, Shoplifting
        "w1": 0.25, "w2": 0.20, "w3": 0.45, "w4": 0.10,
    },
    "public_safety": {     # Arson, Explosion, Vandalism（公共安全；環境因素影響大）
        "w1": 0.30, "w2": 0.25, "w3": 0.35, "w4": 0.10,
    },
    "default": {           # Abuse, Arrest, RoadAccidents 等
        "w1": 0.35, "w2": 0.20, "w3": 0.35, "w4": 0.10,
    },
}
VIOLENT_CRIMES       = {"Fighting", "Assault", "Shooting"}
PROPERTY_CRIMES      = {"Robbery", "Burglary", "Stealing", "Shoplifting"}
PUBLIC_SAFETY_CRIMES = {"Arson", "Explosion", "Vandalism"}
# HIGH_SEVERITY（用於 max_reassignment 分級）
HIGH_SEVERITY_CRIMES = {"Shooting", "Robbery", "Assault"}


# ── Rcost 分段懲罰 ────────────────────────────────────────

def compute_rcost(
    total_turns: int,
    threshold_low: int = 3,
    threshold_high: int = 8,
) -> float:
    """
    Rcost 分段懲罰（門檻值待 pilot experiment 確認）：
      turns ≤ low    → 0              （免懲罰區）
      low < turns ≤ high → 線性 [0, 1]（線性懲罰區）
      turns > high   → 1.0            （最大懲罰）

    線性公式：(turns - threshold_low) / (threshold_high - threshold_low)
      turns = threshold_low → 0 / span = 0.0
      turns = threshold_high → span / span = 1.0
      （修正前為除以 threshold_high，導致 turns=8 時僅 0.625）

    預設門檻（pilot experiment 建議值）：
      threshold_low  = 3  ← 正常案例（無衝突）平均輪次
      threshold_high = 8  ← HARD 衝突案例平均輪次
      → 實際值應以 20-30 筆 pilot cases 統計後調整
    """
    if total_turns <= threshold_low:
        return 0.0
    span = threshold_high - threshold_low
    if span <= 0:
        return 1.0
    elif total_turns <= threshold_high:
        return (total_turns - threshold_low) / span
    else:
        return 1.0


# ── PlannerAgent ─────────────────────────────────────────

class PlannerAgent:
    """
    Planner 不繼承 BaseAgent（角色是協調者，不直接分析影像）

    建構時接受 agents_dict（依角色索引）：
      {
        "environment": EnvironmentAgent（可選）,
        "action":      ActionAgent,
        "time_emotion": TimeEmotionAgent,
        "semantic":    SemanticAgent,
      }
    """

    PLANNER_SYSTEM_PROMPT = """
You are a senior criminal investigation coordinator.
Your task is to orchestrate specialist agents to analyze
crime surveillance videos and generate legally-grounded
forensic reports.
"""

    def __init__(
        self,
        agents_dict: Dict[str, Any],
        reflector: ReflectorAgent,
        reward_weights=None,
    ):
        self.agents = agents_dict          # {"environment": ..., "action": ..., ...}
        self.reflector = reflector
        self.weights = reward_weights or cfg.reward
        self.model_name = cfg.model.planner_model
        self._total_turns: int = 0
        self._case_low_reliability: bool = False
        self._rcost_threshold_high: int = 8    # 動態調整（HIGH_SEVERITY = 10）

    # ── 主要入口 ─────────────────────────────────────────

    def run(self, frames: List, video_metadata: Dict) -> Dict[str, Any]:
        """
        完整分析流程（四步驟）：
        Step 1 → Step 2 → Step 2b → Step 3 → Step 4 → Reflector → 衝突解決 → 最終報告
        """
        self._total_turns = 0
        self._case_low_reliability = False

        # 重置所有代理人
        for agent in self.agents.values():
            if hasattr(agent, "reset"):
                agent.reset()
        self.reflector.reset()

        case_id  = video_metadata.get("case_id", "unknown")
        reports: Dict[str, AgentReport] = {}

        # ── Step 1: Environment Check（條件式）─────────
        env_confidence = 1.0
        reports, env_confidence = self._step1_environment(
            frames, video_metadata, reports
        )

        if env_confidence < 0.4:
            self._case_low_reliability = True
            logger.warning(
                f"[Planner] env_confidence={env_confidence:.2f} < 0.4 → "
                "LOW_RELIABILITY 標記啟動"
            )

        # ── Step 2: Behavioral Analysis（並行）─────────
        human_interaction = video_metadata.get("human_interaction", True)
        if human_interaction and env_confidence >= 0.4:
            reports = self._step2_behavior_parallel(
                frames, video_metadata, reports
            )
        elif human_interaction:
            # env_confidence 太低但仍需分析（低可信度標記）
            reports = self._step2_behavior_parallel(
                frames, video_metadata, reports
            )

        # ── Step 2b: Semantic 初步法律框架（新增）────────
        # 門檻：action_confidence >= 0.4（中等以上視覺基礎才做初步框架，
        # 避免視覺幾乎無信心時 Semantic 框架引入誤導）
        action_report = reports.get("action")
        action_confidence = action_report.confidence if action_report else 0.0
        legal_framework: Optional[Dict] = None

        if action_confidence >= 0.4:
            legal_framework = self._step2b_preliminary_legal(
                reports, video_metadata
            )
            if legal_framework:
                # 將法律框架注入 Action / TimeEmotion Agent
                for role in ("action", "time_emotion"):
                    agent = self.agents.get(role)
                    if agent and hasattr(agent, "set_legal_framework"):
                        agent.set_legal_framework(legal_framework)

        # ── Step 3: 針對性視覺佐證補充（新增）───────────
        # Action + TimeEmotion 依 legal_framework 執行 refine，
        # 補充各自對 key_elements_to_verify 的視覺佐證。
        if legal_framework and action_confidence >= 0.4:
            reports = self._step3_targeted_evidence(reports)

        # ── Step 4: Legal Classification（最終定性）─────
        # 原 Step 3 → 改為 Step 4，此時 Semantic 可看到所有補強資訊。
        # 門檻從 0.5 提高至 0.6：法律分析需建立在可靠的視覺基礎上。
        action_report = reports.get("action")  # 取 refine 後最新版
        action_confidence = action_report.confidence if action_report else 0.0

        if action_confidence >= 0.6:
            reports = self._step4_final_legal(
                frames, video_metadata, reports, legal_framework
            )
        elif action_confidence >= 0.4:
            # 中等信心：先嘗試讓 Action Agent refine（補充分析），再呼叫 Semantic
            action_agent = self.agents.get("action")
            if action_agent:
                logger.info(
                    f"[Planner] action_confidence={action_confidence:.2f} < 0.6，"
                    "嘗試 Action refine 後再呼叫 Semantic"
                )
                others = [r for r in reports.values()
                          if r.agent_name != (action_report.agent_name if action_report else "")]
                action_report = action_agent.refine(list(others))
                reports["action"] = action_report
                self._total_turns += 1
                action_confidence = action_report.confidence

            if action_confidence >= 0.6:
                reports = self._step4_final_legal(
                    frames, video_metadata, reports, legal_framework
                )
            else:
                # 仍不足 → 標記 LOW_VISUAL_CONFIDENCE，讓 Semantic 調整法律信心
                logger.info(
                    f"[Planner] refine 後 action_confidence={action_confidence:.2f}，"
                    "以 LOW_VISUAL_CONFIDENCE 呼叫 Semantic Agent"
                )
                video_metadata = dict(video_metadata)
                video_metadata["low_visual_confidence"] = True
                reports = self._step4_final_legal(
                    frames, video_metadata, reports, legal_framework
                )
        else:
            logger.info(
                f"[Planner] action_confidence={action_confidence:.2f} < 0.4，"
                "跳過 Semantic Agent（視覺基礎過弱）"
            )

        # ── Reflector CASAM + 衝突解決 ─────────────────
        # 高嚴重犯罪（Shooting/Robbery/Assault）給予 2 次重新指派機會
        action_cat = reports.get("action")
        _crime_type = action_cat.crime_category if action_cat else "Normal"
        max_reassign = 2 if _crime_type in HIGH_SEVERITY_CRIMES else 1

        # 高嚴重犯罪的 Rcost 門檻相應提高（容許更多輪次）
        rcost_high = 10 if _crime_type in HIGH_SEVERITY_CRIMES else 8

        all_reports = list(reports.values())
        final_audit = self._resolve_conflicts(
            frames, video_metadata, all_reports,
            max_reassignment=max_reassign,
            rcost_threshold_high=rcost_high,
        )

        # ── 最終報告整合 ────────────────────────────────
        return self._synthesize_final_report(
            list(reports.values()), final_audit, case_id
        )

    # ── Step 1 ───────────────────────────────────────────

    def _step1_environment(
        self,
        frames: List,
        video_metadata: Dict,
        reports: Dict[str, AgentReport],
    ) -> tuple:
        env_agent = self.agents.get("environment")
        video_quality = video_metadata.get("video_quality", 1.0)
        occlusion = video_metadata.get("occlusion_detected", False)

        if (video_quality < 0.6 or occlusion) and env_agent:
            logger.info(
                f"[Planner] Step 1: Environment Check "
                f"(quality={video_quality:.2f}, occlusion={occlusion})"
            )
            env_report = env_agent.analyze(frames, video_metadata)
            reports["environment"] = env_report
            self._total_turns += 1
            return reports, env_report.confidence

        return reports, 1.0

    # ── Step 2 ───────────────────────────────────────────

    def _step2_behavior_parallel(
        self,
        frames: List,
        video_metadata: Dict,
        reports: Dict[str, AgentReport],
    ) -> Dict[str, AgentReport]:
        """並行呼叫 Action + Time&Emotion"""
        action_agent = self.agents.get("action")
        te_agent     = self.agents.get("time_emotion")
        tasks: Dict[str, Any] = {}
        if action_agent:
            tasks["action"] = action_agent
        if te_agent:
            tasks["time_emotion"] = te_agent

        if not tasks:
            return reports

        logger.info(f"[Planner] Step 2: Parallel analysis ({list(tasks.keys())})")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(agent.analyze, frames, video_metadata): role
                for role, agent in tasks.items()
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    reports[role] = future.result()
                    self._total_turns += 1
                    r = reports[role]
                    logger.info(
                        f"  [{role}] {r.crime_category} conf={r.confidence:.2f}"
                    )
                except Exception as e:
                    logger.error(f"  [{role}] 分析失敗：{e}")

        return reports

    # ── Step 2b ──────────────────────────────────────────

    def _step2b_preliminary_legal(
        self,
        reports: Dict[str, AgentReport],
        video_metadata: Dict,
    ) -> Optional[Dict]:
        """
        Step 2.5：Semantic Agent 根據 Action + TimeEmotion 初步結果，
        確定候選罪名範圍與需核查的構成要件。

        回傳 legal_framework dict，失敗時回傳 None（不中斷主流程）。
        """
        semantic_agent = self.agents.get("semantic")
        action_report  = reports.get("action")
        te_report      = reports.get("time_emotion")

        if not semantic_agent or not action_report:
            logger.info("[Planner] Step 2b 跳過：缺少 Semantic Agent 或 Action Report")
            return None

        logger.info("[Planner] Step 2b: Semantic 初步法律框架")
        try:
            framework = semantic_agent.preliminary_legal_framework(
                action_report=action_report,
                te_report=te_report,
                video_metadata=video_metadata,
            )
            self._total_turns += 1
            logger.info(
                f"  [Step 2b] 候選罪名={framework.get('candidate_categories')} "
                f"要件={framework.get('key_elements_to_verify')}"
            )
            return framework
        except Exception as e:
            logger.error(f"[Planner] Step 2b 失敗：{e}，繼續主流程")
            return None

    # ── Step 3 ───────────────────────────────────────────

    def _step3_targeted_evidence(
        self,
        reports: Dict[str, AgentReport],
    ) -> Dict[str, AgentReport]:
        """
        Step 3：Action + TimeEmotion 依注入的 legal_framework 執行 refine，
        補充各自對 key_elements_to_verify 的針對性視覺佐證。

        此步驟在 set_legal_framework() 之後呼叫，
        各 Agent 的 refine() 內部會自動讀取 self._legal_framework。
        """
        logger.info("[Planner] Step 3: 針對性視覺佐證補充（Action + TimeEmotion refine）")

        for role in ("action", "time_emotion"):
            agent = self.agents.get(role)
            report = reports.get(role)
            if not agent or not report:
                continue
            others = [r for key, r in reports.items() if key != role]
            try:
                updated = agent.refine(others)
                reports[role] = updated
                self._total_turns += 1
                logger.info(
                    f"  [{role}] refine 完成，conf={updated.confidence:.2f}，"
                    f"targeted_ev={sum(1 for ev in updated.evidence if 'targeted' in ev.get('type',''))}"
                )
            except Exception as e:
                logger.error(f"  [{role}] Step 3 refine 失敗：{e}")

        return reports

    # ── Step 4 (原 Step 3) ───────────────────────────────

    def _step4_final_legal(
        self,
        frames: List,
        video_metadata: Dict,
        reports: Dict[str, AgentReport],
        legal_framework: Optional[Dict] = None,
    ) -> Dict[str, AgentReport]:
        """
        Step 4（原 Step 3）：Semantic Agent 最終法律定性。
        此時可看到所有 Agent 補強後的完整報告，
        並將 legal_framework 注入 video_metadata 以聚焦 HyDE 查詢範圍。
        """
        semantic_agent = self.agents.get("semantic")
        if not semantic_agent:
            return reports

        # 傳遞低可信度脈絡給 Semantic Agent
        action_report = reports.get("action")
        enriched_meta = dict(video_metadata)
        if action_report:
            enriched_meta["action_confidence"] = action_report.confidence
            enriched_meta["action_category"]   = action_report.crime_category
        if self._case_low_reliability:
            enriched_meta["low_reliability"] = True

        # 注入 Step 2.5 的初步法律框架（Semantic 的 _generate_hyde_query 會使用）
        if legal_framework:
            enriched_meta["legal_framework"] = legal_framework

        logger.info("[Planner] Step 4: Legal Classification（Semantic Agent 最終定性）")
        sem_report = semantic_agent.analyze(frames, enriched_meta)
        reports["semantic"] = sem_report
        self._total_turns += 1
        logger.info(
            f"  [semantic] {sem_report.crime_category} conf={sem_report.confidence:.2f}"
        )
        return reports

    # ── 衝突解決流程 ─────────────────────────────────────

    def _resolve_conflicts(
        self,
        frames: List,
        video_metadata: Dict,
        all_reports: List[AgentReport],
        max_reassignment: int = 1,
        rcost_threshold_high: int = 8,
    ) -> ReflectorOutput:
        self._rcost_threshold_high = rcost_threshold_high
        """
        SOFT → Reflector 直接指派目標代理人 refine
        HARD → Planner 以衝突脈絡重新呼叫衝突代理人 analyze
        NONE → 直接回傳
        """
        retry_count = 0

        while True:
            audit = self.reflector.audit(all_reports, retry_count=retry_count)

            if audit.conflict_type == "NONE":
                logger.info("[Planner] 衝突解決：NONE，進入最終整合")
                return audit

            elif audit.conflict_type == "SOFT":
                logger.info(
                    f"[Planner] SOFT 衝突 → 直接指派 {audit.target_agent}，"
                    f"retry_count={retry_count}"
                )
                target = self._find_agent_by_name(audit.target_agent)
                if target and retry_count < 2:
                    others = [r for r in all_reports
                              if r.agent_name != audit.target_agent]
                    updated = target.refine(others)
                    all_reports = [
                        updated if r.agent_name == audit.target_agent else r
                        for r in all_reports
                    ]
                    self._total_turns += 1
                    retry_count += 1
                else:
                    logger.info("[Planner] SOFT retry 上限，進入最終整合")
                    return audit

            elif audit.conflict_type == "HARD":
                logger.warning(
                    f"[Planner] HARD 衝突 ({audit.conflict_layer}) → "
                    f"重新指派衝突代理人，max_reassignment={max_reassignment}"
                )
                if retry_count >= max_reassignment:
                    logger.warning("[Planner] 達到最大重新指派次數，標記 INSUFFICIENT_EVIDENCE")
                    return audit

                for pair in audit.conflict_pairs:
                    for agent_name in [pair["agent_a"], pair["agent_b"]]:
                        agent = self._find_agent_by_name(agent_name)
                        if agent and hasattr(agent, "analyze"):
                            conflict_meta = dict(video_metadata)
                            conflict_meta.update({
                                "reassignment_reason": "HARD_CONFLICT",
                                "conflict_context": pair["conflict_description"],
                                "focus_frames": audit.recommended_focus_frames,
                                "instruction": (
                                    f"Compare your analysis with {pair['agent_b']} "
                                    f"report below. Focus on frames "
                                    f"{audit.recommended_focus_frames}."
                                ),
                            })
                            try:
                                updated = agent.analyze(frames, conflict_meta)
                                all_reports = [
                                    updated if r.agent_name == agent_name else r
                                    for r in all_reports
                                ]
                                self._total_turns += 2  # 重新指派耗費更多輪次
                            except Exception as e:
                                logger.error(f"重新指派 {agent_name} 失敗：{e}")

                retry_count += 1

    # ── 最終報告整合 ─────────────────────────────────────

    def _synthesize_final_report(
        self,
        reports: List[AgentReport],
        final_audit: ReflectorOutput,
        case_id: str,
    ) -> Dict[str, Any]:
        """
        依設計規格的 final report 格式：
        fact_finding / behavior_analysis / legal_classification / uncertainty_notes
        動態調整整合權重
        """
        crime_type = final_audit.consensus_category
        weights = self._get_dynamic_weights(crime_type)

        # 如果環境可信度低，調整 Rcons/Rcost 權重
        env_report = next(
            (r for r in reports if r.crime_category == "ENVIRONMENTAL_ASSESSMENT"),
            None,
        )
        if env_report and env_report.confidence < 0.4:
            weights = dict(weights)
            weights["w2"] = weights.get("w2", 0.30) + 0.05
            weights["w4"] = max(0.0, weights.get("w4", 0.10) - 0.05)

        # 整合權重：confidence × evidence_coverage
        cat_reports = [
            r for r in reports if r.crime_category != "ENVIRONMENTAL_ASSESSMENT"
        ]
        best_report = (
            max(cat_reports, key=lambda r: r.confidence * (1 + len(r.evidence) * 0.1))
            if cat_reports else None
        )

        # Semantic 報告
        semantic_report = next(
            (r for r in reports if "法律" in r.agent_name or "Semantic" in r.agent_name),
            None,
        )
        rlegal = (
            semantic_report.metadata.get("rlegal", 0.0)
            if semantic_report else 0.0
        )
        # 高嚴重犯罪允許更多輪次，threshold_high 相應調整至 10
        rcost   = compute_rcost(
            self._total_turns,
            threshold_low=3,
            threshold_high=self._rcost_threshold_high,
        )
        rcons   = final_audit.rcons_score

        # 法律條文
        applicable_articles, elements_covered = [], []
        if semantic_report:
            for ev in semantic_report.evidence:
                if ev.get("type") == "legal_article":
                    applicable_articles.append(ev.get("article_id", ""))
                if ev.get("type") == "legal_elements":
                    elements_covered.extend(ev.get("elements", []))

        # 低可信度 / 衝突佐證
        uncertainty_notes = self._build_uncertainty_notes(
            reports, final_audit, self._case_low_reliability
        )

        # action / time_emotion 報告
        action_r = next(
            (r for r in reports if "行為" in r.agent_name), None
        )
        te_r = next(
            (r for r in reports if "時間" in r.agent_name or "情緒" in r.agent_name),
            None,
        )

        causal_chain = (
            te_r.metadata.get("causal_chain", "") if te_r else ""
        )
        timeline = (
            f"異常幀分布：{te_r.frame_references[:5]}" if te_r else ""
        )

        final_report = {
            "case_id": case_id,
            "fact_finding": {
                "description": best_report.reasoning if best_report else "分析不完整",
                "supporting_frames": self._collect_all_key_frames(reports),
                "confidence": best_report.confidence if best_report else 0.0,
            },
            "behavior_analysis": {
                "causal_chain": (
                    str(causal_chain) if causal_chain
                    else action_r.reasoning if action_r else ""
                ),
                "timeline": timeline,
            },
            "legal_classification": {
                "applicable_articles": applicable_articles,
                "elements_covered": elements_covered,
                "coverage_rate": rlegal,
            },
            "uncertainty_notes": uncertainty_notes,
            # 獎勵訊號
            "rcons":  rcons,
            "rlegal": rlegal,
            "rcost":  rcost,
            "reward_weights": weights,
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
            f"[Planner] 最終報告 | case={case_id} | "
            f"category={crime_type} | rcons={rcons:.3f} | "
            f"rlegal={rlegal:.3f} | rcost={rcost:.3f} | turns={self._total_turns}"
        )
        return final_report

    # ── 獎勵函數 ─────────────────────────────────────────

    def compute_reward(self, result: Dict, ground_truth: str) -> float:
        """
        R_i = w1*Racc + w2*Rcons + w3*Rlegal - w4*Rcost
        GRPO 訓練時對每條 Rollout 計算此分數。
        """
        racc  = 1.0 if result.get("final_category") == ground_truth else 0.0
        rcons  = result.get("rcons", 0.0)
        rlegal = result.get("rlegal", 0.0)
        rcost  = result.get("rcost", 0.0)
        w = result.get("reward_weights", {
            "w1": self.weights.w1, "w2": self.weights.w2,
            "w3": self.weights.w3, "w4": self.weights.w4,
        })
        return (
            w["w1"] * racc
            + w["w2"] * rcons
            + w["w3"] * rlegal
            - w["w4"] * rcost
        )

    # ── 輔助方法 ─────────────────────────────────────────

    def _find_agent_by_name(self, agent_name: str):
        """依代理人名稱（或 role）找到對應 Agent 物件"""
        # 先查 role dict
        name_lower = agent_name.lower()
        for role, agent in self.agents.items():
            if (role in name_lower
                    or name_lower in role
                    or agent_name == getattr(agent, "name", "")):
                return agent
        # Fallback：遍歷 agent.name
        for agent in self.agents.values():
            if getattr(agent, "name", "") == agent_name:
                return agent
        return None

    def _get_dynamic_weights(self, crime_type: str) -> Dict[str, float]:
        if crime_type in VIOLENT_CRIMES:
            return DYNAMIC_WEIGHTS["violent"]
        if crime_type in PROPERTY_CRIMES:
            return DYNAMIC_WEIGHTS["property"]
        if crime_type in PUBLIC_SAFETY_CRIMES:
            return DYNAMIC_WEIGHTS["public_safety"]
        return DYNAMIC_WEIGHTS["default"]

    def _collect_all_key_frames(self, reports: List[AgentReport]) -> List[int]:
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
        """
        ALWAYS 填入（即使全部空）→ 提升司法可信度。
        """
        low_conf_items  = []
        conflicting_ev  = []
        insufficient_ev = []

        for r in reports:
            if r.confidence < 0.6 and r.crime_category != "ENVIRONMENTAL_ASSESSMENT":
                low_conf_items.append(
                    f"{r.agent_name}: {r.crime_category}（信心 {r.confidence:.2f}）"
                )
            for flag in r.conflict_flags:
                conflicting_ev.append(f"{r.agent_name}: {flag}")

        if low_reliability:
            insufficient_ev.append("環境可信度過低，視覺證據品質不足")

        if audit.conflict_type == "HARD" and not audit.is_convergent:
            insufficient_ev.append(
                f"HARD 衝突無法解決（層級：{audit.conflict_layer}），"
                "結論可信度受影響"
            )

        for c in audit.conflicts:
            if c.severity > 0.7:
                conflicting_ev.append(c.description)

        return {
            "low_confidence_items": low_conf_items,
            "conflicting_evidence": conflicting_ev,
            "insufficient_evidence": insufficient_ev,
        }

    # ── 向後相容：接受 local_agents list ─────────────────

    @classmethod
    def from_list(
        cls,
        local_agents: List,
        reflector: ReflectorAgent,
        reward_weights=None,
    ) -> "PlannerAgent":
        """
        向後相容介面：傳入代理人 list，自動依 name 分配角色。
        """
        role_map: Dict[str, Any] = {}
        for agent in local_agents:
            name = getattr(agent, "name", "")
            if "環境" in name or "Environment" in name.lower():
                role_map["environment"] = agent
            elif "行為" in name or "Action" in name.lower():
                role_map["action"] = agent
            elif "時間" in name or "情緒" in name or "Emotion" in name.lower():
                role_map["time_emotion"] = agent
            elif "法律" in name or "Semantic" in name.lower():
                role_map["semantic"] = agent
        return cls(role_map, reflector, reward_weights)
