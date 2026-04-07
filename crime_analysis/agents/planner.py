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
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import AgentReport
from .action_emotion_agent import UCF_CATEGORIES
from .reflector import ReflectorAgent, ReflectorOutput
from config import cfg
from rag.rag_module import LEGAL_ELEMENTS, GROUP_LEGAL_CONTEXT

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
    threshold_low: int = 4,
    threshold_high: int = 6,
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


# ── Step 3b Prompt Template ────────────────────────────────
REPORT_SYSTEM_PROMPT = """\
你是一位台灣刑事鑑定報告撰寫專家。根據影片行為分析結果、法律檢索資料、與內部審查結論，\
撰寫一份結構化的初步鑑定報告。

報告要求：
1. 必須以繁體中文撰寫
2. 必須涵蓋所有適用的法律構成要件（逐一論述是否該當）
3. 引用具體法條條號
4. 區分「影片可觀察事實」與「推論」
5. 若有衝突或不確定性，須明確說明
"""

REPORT_USER_TEMPLATE = """\
## 案件資訊
- 案件編號：{case_id}
- 判定犯罪類型：{crime_type}
- 分析信心：{confidence:.2f}

## 行為分析摘要
{rationale}

## 適用法條
{articles}

## 構成要件檢核清單
請逐一論述以下要件是否該當：
{elements_checklist}

## RAG 檢索法條摘要
{rag_laws_summary}

## 內部審查結果
- 衝突類型：{conflict_type}
- 一致性分數（Rcons）：{rcons:.2f}
{conflict_detail}

## 輸出格式
請依下列結構撰寫報告：

### 一、事實認定
（根據影片觀察到的客觀事實）

### 二、構成要件分析
（逐一論述每個構成要件是否該當，引用影片證據）

### 三、法律適用
（引用具體法條條號與裁判見解）

### 四、不確定性與限制
（影片分析的限制、無法確認的事項）

### 五、初步結論
（綜合判斷）
"""


def build_report_prompt(
    case_id: str,
    crime_type: str,
    confidence: float,
    rationale: str,
    rag_results: Dict,
    conflict_type: str,
    rcons: float,
    conflict_detail: str = "",
) -> List[Dict[str, str]]:
    """組裝 Step 3b 報告生成的 chat messages。"""
    elements = LEGAL_ELEMENTS.get(crime_type, [])
    articles = GROUP_LEGAL_CONTEXT.get(crime_type, [])

    elements_checklist = "\n".join(f"- [ ] {e}" for e in elements) if elements else "（無對應構成要件）"
    articles_str = "、".join(articles) if articles else "（無對應法條）"

    # 整理 RAG 檢索到的法條摘要
    rag_laws = rag_results.get("laws", [])
    if rag_laws:
        rag_laws_summary = "\n".join(
            f"- {l.get('article_id', '未知條號')}：{l.get('text', '')[:120]}"
            for l in rag_laws[:5]
        )
    else:
        rag_laws_summary = "（未檢索到相關法條）"

    user_content = REPORT_USER_TEMPLATE.format(
        case_id=case_id,
        crime_type=crime_type,
        confidence=confidence,
        rationale=rationale or "（無行為分析摘要）",
        articles=articles_str,
        elements_checklist=elements_checklist,
        rag_laws_summary=rag_laws_summary,
        conflict_type=conflict_type,
        rcons=rcons,
        conflict_detail=conflict_detail,
    )

    return [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def _extract_uca_guided_frames(video_path, uca_segments, n_frames, cv2_mod, pil_mod):
    """從影片用 UCA 引導抽幀（70% 犯罪時段 + 30% 背景），回傳 PIL Image list。"""
    if not video_path or not Path(video_path).exists():
        return []

    cap = cv2_mod.VideoCapture(video_path)
    total = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2_mod.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        return []

    if uca_segments:
        indices = PlannerAgent._uca_guided_indices(total, uca_segments, fps, n_frames)
    else:
        indices = [int(i * total / n_frames) for i in range(n_frames)]

    keyframes = []
    for idx in indices:
        cap.set(cv2_mod.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            keyframes.append(pil_mod.fromarray(cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2RGB)))
    cap.release()
    return keyframes


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
        self._rcost_threshold_high: int = 6  # Pilot P75

        # Step 3b 報告生成模型（延遲載入）
        self._report_tokenizer = None
        self._report_model = None

        # 消融實驗 flags（由 pilot_experiment.py 設定）
        self._skip_vlm_classify: bool = False
        self._skip_vlm_report: bool = False
        self._vlm_reason: str = ""

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

        low_reliability = env_confidence < cfg.inference.confidence_low_threshold
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
        confidence_ref_threshold = cfg.inference.confidence_mid_threshold
        confidence_low = cfg.inference.confidence_low_threshold
        if ae_report and confidence_low <= ae_confidence < confidence_ref_threshold:
            logger.info(f"[Planner] conf={ae_confidence:.2f} < {confidence_ref_threshold}，嘗試 ActionEmotion refine")
            ae_agent = self.agents.get("action_emotion")
            if ae_agent:
                ae_report = ae_agent.refine(list(reports.values()))
                reports["action_emotion"] = ae_report
                ae_confidence = ae_report.confidence
                self._total_turns += 1

        if ae_confidence < confidence_low:
            logger.info(f"[Planner] conf={ae_confidence:.2f} < {confidence_low} → 跳過 RAG 查詢")

        # ── Step 2b：VLM 分類（Qwen3-VL 覆核 MIL Head）──────
        mil_crime_type = ae_report.crime_category if ae_report else "Normal"
        mil_confidence = ae_confidence
        crime_type = mil_crime_type

        # 載入 Qwen3-VL（後續 Step 3b 報告生成也用同一個模型）
        if not self._skip_vlm_classify or not self._skip_vlm_report:
            self._load_report_model()

        if self._report_model is not None and not self._skip_vlm_classify:
            vlm_result = self._vlm_classify(frames, video_metadata)
            if vlm_result:
                crime_type, vlm_conf = vlm_result
                ae_confidence = vlm_conf
                if ae_report:
                    ae_report.crime_category = crime_type
                    ae_report.confidence = vlm_conf
                    ae_report.metadata["mil_crime_type"] = mil_crime_type
                    ae_report.metadata["mil_confidence"] = mil_confidence
                    ae_report.metadata["vlm_used"] = True
                logger.info(
                    f"[Planner] Step 2b: VLM 分類 MIL={mil_crime_type}({mil_confidence:.2f}) "
                    f"→ VLM={crime_type}({vlm_conf:.2f})"
                )
                self._total_turns += 1

        # ── Step 2c-2d：RAG-guided Classification Verification ──
        # 暫時關閉（測試顯示 VLM 對構成要件全報 0 matched，反而降低準確率）
        # TODO: 改善 VLM 對法律要件的理解後重新啟用
        if False and (self._report_model is not None
                and not self._skip_vlm_classify
                and crime_type != "Normal"
                and self.rag):
            verified = self._rag_verify_classification(
                frames, video_metadata, crime_type,
            )
            if verified and verified != crime_type:
                old_type = crime_type
                crime_type = verified
                if ae_report:
                    ae_report.crime_category = crime_type
                    ae_report.metadata["rag_verified"] = True
                    ae_report.metadata["pre_verify_type"] = old_type
                logger.info(
                    f"[Planner] Step 2d: RAG 驗證修正 {old_type} → {crime_type}"
                )
                self._total_turns += 1
            else:
                logger.info(f"[Planner] Step 2c: RAG 驗證通過 → {crime_type}")

        # ── Step 3：法律整合（3a RAG → 3b 報告生成 → 3c Rlegal）──
        rag_results: Dict = {"laws": [], "judgments": []}
        rlegal = 0.0

        # Step 3a：RAG 查詢取候選法條
        if ae_confidence >= cfg.inference.confidence_low_threshold and self.rag and crime_type != "Normal":
            rationale = ae_report.metadata.get("rationale", "") if ae_report else ""
            query_text = (
                self.rag.generate_hypothetical_doc(rationale, crime_type)
                if hasattr(self.rag, "generate_hypothetical_doc") else rationale
            )
            rag_results = self.rag.query(query_text, query_type="semantic")
            self._total_turns += 1
            logger.info(f"[Planner] Step 3a: RAG 查詢 → {len(rag_results.get('laws', []))} 條法條")

        # Step 3b：Qwen3-VL 帶影片幀生成鑑定報告
        conflict_detail = ""
        if hasattr(self, '_last_audit') and self._last_audit:
            conflict_detail = f"- 衝突層：{self._last_audit.conflict_layer}"

        # 不傳 MIL Head 的 rationale（常有錯誤分類描述，會誤導 VLM 報告生成）
        # VLM 直接根據影片幀 + RAG 法條 + VLM 自己的分類來寫報告
        report_messages = build_report_prompt(
            case_id=case_id,
            crime_type=crime_type,
            confidence=ae_confidence,
            rationale="",  # 留空，讓 VLM 根據影片內容自行分析
            rag_results=rag_results,
            conflict_type="NONE",
            rcons=0.0,
            conflict_detail=conflict_detail,
        )
        temperature = video_metadata.get("temperature", cfg.model.temperature)
        if not self._skip_vlm_report:
            generated_report_text = self._call_qwen3_vl(
                report_messages, frames, temperature=temperature,
                video_metadata=video_metadata,
            )
        else:
            generated_report_text = ""  # 消融⑤：強制 fallback

        # Fallback：VLM 未載入或生成失敗時，用 RAG + 構成要件組裝基礎報告
        if not generated_report_text:
            generated_report_text = self._build_fallback_report(
                crime_type, ae_confidence,
                ae_report.metadata.get("rationale", "") if ae_report else "",
                rag_results,
            )
            self._report_method = "fallback"
            logger.info(f"[Planner] Step 3b: 使用結構化 fallback（{len(generated_report_text)} chars）")
        else:
            self._report_method = "qwen3-vl"
            logger.info(f"[Planner] Step 3b: Qwen3-VL 生成完成（{len(generated_report_text)} chars）")

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
            generated_report_text=generated_report_text,
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
        quality_threshold = cfg.inference.video_quality_threshold
        occlusion = video_metadata.get("occlusion_detected", False)

        if (quality < quality_threshold or occlusion) and env_agent:
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
                    # 保存 VLM 分類結果，re-analyze 後恢復
                    vlm_category = None
                    vlm_conf = None
                    for r in all_reports:
                        if hasattr(r, 'metadata') and r.metadata.get("vlm_used"):
                            vlm_category = r.crime_category
                            vlm_conf = r.confidence
                            break

                    conflict_meta = dict(video_metadata)
                    conflict_meta.update({
                        "reassignment_reason": "HARD_CONFLICT",
                        "conflict_context": audit.conflict_layer,
                        "focus_frames": audit.recommended_focus_frames,
                    })
                    try:
                        updated = ae_agent.analyze(frames, conflict_meta)
                        # 恢復 VLM 分類（MIL re-analyze 不應覆蓋 VLM 結果）
                        if vlm_category:
                            updated.crime_category = vlm_category
                            updated.confidence = vlm_conf
                            updated.metadata["vlm_used"] = True
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
        generated_report_text: str = "",
    ) -> Dict[str, Any]:
        # VLM 分類優先於 Reflector consensus（Reflector 用的是 MIL Head 的結果）
        # 如果 VLM 有分類結果，以 VLM 為準
        ae_report_for_cat = next(
            (r for r in reports
             if "行為情緒" in r.agent_name or "ActionEmotion" in r.agent_name),
            None,
        )
        vlm_used = ae_report_for_cat and ae_report_for_cat.metadata.get("vlm_used", False)
        if vlm_used:
            crime_type = ae_report_for_cat.crime_category  # VLM 分類結果
        else:
            crime_type = final_audit.consensus_category  # fallback 到 Reflector
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
            threshold_low=4,  # Pilot median
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
                "description": generated_report_text or rationale or (ae_report.reasoning if ae_report else "分析不完整"),
                "rationale": rationale,
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
            "report_generation_method": getattr(self, "_report_method", "unknown"),
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

    def evaluate(
        self,
        result: Dict,
        ground_truth: str,
        llm_judge=None,
    ) -> Dict[str, Any]:
        """
        四指標評估框架（測試階段使用，非訓練信號）：
        R = w1·Racc + w2·Rcons + w3·Rlegal − w4·Rcost

        Args:
            result: pipeline.run() 的輸出
            ground_truth: 正確犯罪類別
            llm_judge: 可選的 LLMJudge 實例，若提供則同時執行外部語意評分
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
        eval_result = {
            "R": total,
            "Racc": racc,
            "Rcons": rcons,
            "Rlegal": rlegal,
            "Rcost": rcost,
        }

        # 可選：外部 LLM-as-Judge 語意評分
        if llm_judge is not None:
            crime_type = result.get("final_category", "Normal")
            report_text = result.get("fact_finding", {}).get("description", "")
            prompt = f"犯罪類別：{crime_type}，正確答案：{ground_truth}"
            try:
                judge_scores = llm_judge.rubric_score(prompt, report_text, crime_type)
                eval_result["llm_judge"] = judge_scores
            except Exception as e:
                logger.warning(f"[evaluate] LLM Judge 評分失敗：{e}")
                eval_result["llm_judge"] = None

        return eval_result

    # ── Step 3b: Fallback 報告組裝 ────────────────────────────

    def _build_fallback_report(
        self,
        crime_type: str,
        confidence: float,
        rationale: str,
        rag_results: Dict,
    ) -> str:
        """
        當 Qwen3 不可用時，用 RAG 結果 + LEGAL_ELEMENTS 組裝基礎報告。
        目的：讓 compute_rlegal 的 substring matching 能命中法律關鍵字。
        """
        elements = LEGAL_ELEMENTS.get(crime_type, [])
        articles = GROUP_LEGAL_CONTEXT.get(crime_type, [])

        parts = [f"一、事實認定\n本案經影片行為分析，判定犯罪類別為{crime_type}，信心程度{confidence:.2f}。"]

        if rationale:
            parts.append(f"行為分析摘要：{rationale}")

        # 只放 RAG 查到的法條內容，不直接列出 LEGAL_ELEMENTS
        # 讓 Rlegal 的 substring matching 自然命中或不命中
        parts.append(f"\n二、法律適用")
        if articles:
            parts.append(f"本案可能適用{'、'.join(articles)}。")

        rag_laws = rag_results.get("laws", [])
        if rag_laws:
            for law in rag_laws[:3]:
                text = law.get("text", law.get("content", ""))[:200]
                parts.append(f"{law.get('article_id', '')}：{text}")

        parts.append(f"\n三、初步結論")
        parts.append(f"綜合以上分析，本案涉及{crime_type}類型犯罪之可能性為{confidence:.0%}。")

        return "\n".join(parts)

    # ── MIL-guided Frame Sampling ──────────────────────────

    def _mil_guided_frames(self, frames, video_path, video_metadata, n, cv2_mod, pil_mod):
        """
        用 MIL Head 的 snippet anomaly scores 引導 VLM 抽幀。
        取 50% 最可疑片段 + 30% UCA 犯罪時段 + 20% 均勻背景。
        """
        import numpy as np

        ae_agent = self.agents.get("action_emotion")
        ae_report = getattr(ae_agent, "_position", None) if ae_agent else None
        snippet_scores = ae_report.metadata.get("snippet_scores", []) if ae_report else []

        # 如果有 snippet scores → 用最可疑的片段抽幀
        if snippet_scores and video_path and Path(video_path).exists():
            cap = cv2_mod.VideoCapture(video_path)
            total = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return self._fallback_frames(frames, n, cv2_mod, pil_mod)

            # 分配：4 幀最可疑 + 2 幀 UCA + 2 幀均勻
            n_suspicious = n // 2      # 4
            n_uca = n // 4             # 2
            n_context = n - n_suspicious - n_uca  # 2

            # 最可疑的 snippet 對應的幀
            frames_per_snippet = max(1, total // len(snippet_scores))
            top_indices = sorted(range(len(snippet_scores)), key=lambda i: snippet_scores[i], reverse=True)
            suspicious_frames = []
            for si in top_indices[:n_suspicious]:
                mid_frame = si * frames_per_snippet + frames_per_snippet // 2
                suspicious_frames.append(min(mid_frame, total - 1))

            # UCA 犯罪時段的幀
            uca_segments = (video_metadata or {}).get("uca_segments", [])
            fps = cap.get(cv2_mod.CAP_PROP_FPS) or 25.0
            uca_frame_set = set()
            for seg in uca_segments:
                sf = int(seg.get("start", 0) * fps)
                ef = int(seg.get("end", total / fps) * fps)
                for i in range(max(0, sf), min(ef + 1, total)):
                    uca_frame_set.add(i)
            if uca_frame_set:
                uca_list = sorted(uca_frame_set)
                uca_indices = [uca_list[int(i * len(uca_list) / n_uca)] for i in range(n_uca)]
            else:
                uca_indices = []

            # 均勻背景幀
            context_indices = [int(i * total / n_context) for i in range(n_context)]

            # 合併去重
            all_indices = sorted(set(suspicious_frames + uca_indices + context_indices))[:n]

            keyframes = []
            for idx in all_indices:
                cap.set(cv2_mod.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    keyframes.append(pil_mod.fromarray(cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2RGB)))
            cap.release()

            if keyframes:
                logger.info(
                    f"[VLM classify] MIL-guided {len(keyframes)} 幀"
                    f"（{len(suspicious_frames)} suspicious + {len(uca_indices)} UCA + {len(context_indices)} context）"
                )
                return keyframes

        # Fallback
        return self._fallback_frames(frames, n, cv2_mod, pil_mod)

    def _fallback_frames(self, frames, n, cv2_mod, pil_mod):
        """均勻取幀的 fallback。"""
        valid = [f for f in frames if f is not None and hasattr(f, "shape")]
        if not valid:
            return []
        total = len(valid)
        indices = [int(i * total / min(n, total)) for i in range(min(n, total))]
        keyframes = [pil_mod.fromarray(cv2_mod.cvtColor(valid[idx], cv2_mod.COLOR_BGR2RGB)) for idx in indices]
        logger.info(f"[VLM classify] Fallback {len(keyframes)} 張均勻幀")
        return keyframes

    # ── RAG-guided Classification Verification ──────────────

    def _rag_verify_classification(
        self,
        frames: List,
        video_metadata: Dict,
        initial_category: str,
    ) -> Optional[str]:
        """
        Step 2c-2d: 用 RAG 的法律構成要件驗證 VLM 分類。

        流程：
        1. 查出 initial_category 的構成要件
        2. 問 VLM：「你在影片中看到這些要件嗎？」
        3. 如果多數要件不符合 → 讓 VLM 重新選擇類別
        """
        if self._report_model is None or self._report_tokenizer is None:
            return None

        import torch, re, cv2
        from PIL import Image as PILImage

        elements = LEGAL_ELEMENTS.get(initial_category, [])
        if not elements:
            return None

        # 取幀（跟分類用的一樣，8 幀）
        keyframes = _extract_uca_guided_frames(
            (video_metadata or {}).get("video_path", ""),
            (video_metadata or {}).get("uca_segments", []),
            8, cv2, PILImage,
        )
        if not keyframes:
            valid = [f for f in frames if f is not None and hasattr(f, "shape")]
            if not valid:
                return None
            n = len(valid)
            indices = [int(i * n / 8) for i in range(8)]
            keyframes = [PILImage.fromarray(cv2.cvtColor(valid[idx], cv2.COLOR_BGR2RGB)) for idx in indices]

        # 找出 top-3 候選類別（排除 initial_category，提供有限選項）
        from rag.rag_module import GROUP_LEGAL_CONTEXT
        # 同群組的替代類別
        crime_groups = {
            "violent": {"Assault", "Fighting", "Shooting", "Robbery", "Abuse", "Arrest"},
            "property": {"Stealing", "Shoplifting", "Burglary", "Vandalism"},
            "public_safety": {"Arson", "Explosion", "RoadAccidents"},
        }
        current_group = ""
        for g, cats in crime_groups.items():
            if initial_category in cats:
                current_group = g
                break
        # 同群組的替代 + 其他群組各一
        alternatives = [c for c in crime_groups.get(current_group, set()) if c != initial_category][:3]

        elements_str = "\n".join(f"- {e}" for e in elements)
        alt_str = ", ".join(alternatives) if alternatives else ", ".join(UCF_CATEGORIES[:5])

        verify_prompt = (
            f"You classified this CCTV footage as: {initial_category}\n\n"
            f"Required legal elements for {initial_category}:\n"
            f"{elements_str}\n\n"
            f"Look at these frames carefully. For each element, is it VISIBLE?\n"
            f"Count how many elements you can actually see.\n\n"
            f"If you can see MOST elements → keep {initial_category}\n"
            f"If you CANNOT see most elements → choose from: {alt_str}\n\n"
            f"Reply:\n"
            f"ELEMENTS_MATCHED: <number>/{len(elements)}\n"
            f"FINAL_CATEGORY: <category name>"
        )

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in keyframes],
                {"type": "text", "text": verify_prompt},
            ],
        }]

        try:
            processor = self._report_tokenizer
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self._report_model.device)

            with torch.no_grad():
                output_ids = self._report_model.generate(
                    **inputs, max_new_tokens=128, temperature=0.1, do_sample=False,
                )
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True).strip()
            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

            logger.info(f"[RAG verify] {response[:150]}")

            # 解析 ELEMENTS_MATCHED
            em_match = re.search(r"ELEMENTS_MATCHED:\s*(\d+)\s*/\s*(\d+)", response, re.IGNORECASE)
            if em_match:
                matched = int(em_match.group(1))
                total_el = int(em_match.group(2))
                ratio = matched / max(total_el, 1)
                # 如果超過一半要件符合 → 保留原分類
                if ratio >= 0.5:
                    logger.info(f"[RAG verify] {matched}/{total_el} 要件符合 → 保留 {initial_category}")
                    return None

            # 解析 FINAL_CATEGORY
            cat_match = re.search(r"FINAL_CATEGORY:\s*(\w+)", response, re.IGNORECASE)
            if cat_match:
                raw = cat_match.group(1)
                for cat in UCF_CATEGORIES:
                    if cat.lower() == raw.lower():
                        return cat

        except Exception as e:
            logger.warning(f"[RAG verify] 失敗：{e}")

        return None

    # ── UCA 引導抽幀 ──────────────────────────────────────

    @staticmethod
    def _uca_guided_indices(total_frames: int, uca_segments: list, fps: float, n: int) -> list:
        """計算 UCA 引導的幀索引（70% 犯罪時段 + 30% 背景）。"""
        n_event = int(n * 0.7)
        n_context = n - n_event

        event_indices = set()
        for seg in uca_segments:
            start_f = int(seg.get("start", seg.get("start_frame", 0)) * fps)
            end_f = int(seg.get("end", seg.get("end_frame", total_frames)) * fps)
            for i in range(max(0, start_f), min(end_f + 1, total_frames)):
                event_indices.add(i)

        if not event_indices:
            return [int(i * total_frames / n) for i in range(n)]

        event_list = sorted(event_indices)
        if len(event_list) >= n_event:
            selected_event = [event_list[int(i * len(event_list) / n_event)] for i in range(n_event)]
        else:
            selected_event = event_list

        context = [int(i * total_frames / n_context) for i in range(n_context)]
        return sorted(set(selected_event + context))[:n]

    # ── Qwen3-VL 統一模型（分類 + 報告生成）─────────────────

    def _load_report_model(self):
        """
        延遲載入 Qwen3-VL-8B-Instruct（統一 VLM）。
        同一個模型負責 Step 2b 分類 + Step 3b 報告生成。
        """
        if self._report_model is not None:
            return
        try:
            import torch as _torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            model_name = cfg.model.report_model
            logger.info(f"[Planner] 載入 VLM：{model_name}")

            self._report_tokenizer = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True,
            )

            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
            }
            if cfg.model.torch_dtype == "bfloat16" and _torch.cuda.is_available():
                load_kwargs["torch_dtype"] = _torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = "auto"

            self._report_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, **load_kwargs,
            )
            logger.info(f"[Planner] VLM 載入完成：{model_name} (dtype={load_kwargs.get('torch_dtype', 'auto')})")
        except Exception as e:
            logger.warning(f"[Planner] 無法載入 VLM：{e}，將使用 fallback")
            self._report_model = None

    def _vlm_classify(self, frames: List, video_metadata: Dict = None) -> Optional[tuple]:
        """
        Qwen3-VL 犯罪分類。
        優先使用原生影片輸入（動態資訊），fallback 到 8 張靜態幀。
        """
        if self._report_model is None or self._report_tokenizer is None:
            return None

        import torch, re, cv2
        from PIL import Image as PILImage

        # 混淆感知 prompt — 定義精確 + 常混淆組別特別說明
        categories_str = ", ".join(UCF_CATEGORIES)
        prompt = (
            "You are a forensic surveillance video analyst.\n"
            "Look at these frames from a CCTV video and determine what crime is occurring.\n\n"
            f"Choose ONE category from: {categories_str}\n\n"
            "Definitions:\n"
            "- Assault: One person attacking another, one-sided, purpose is to HURT\n"
            "- Robbery: Attacking + TAKING belongings (violence is the MEANS, stealing is the GOAL)\n"
            "- Stealing: Secretly taking items, NO confrontation, thief acts casual\n"
            "- Shoplifting: Inside a STORE, hiding merchandise, leaving without paying\n"
            "- Burglary: BREAKING INTO a building (climbing wall, forcing door/window)\n"
            "- Fighting: MUTUAL combat, BOTH sides throwing punches\n"
            "- Arson: Person deliberately SETTING FIRE, flames spreading\n"
            "- Explosion: Sudden BLAST, shockwave, debris flying\n"
            "- RoadAccidents: VEHICLE collision or hitting pedestrian/animal\n"
            "- Vandalism: SMASHING or DESTROYING property (not fire, not explosion)\n"
            "- Abuse: REPEATED harm to helpless person (elderly/child), over time\n"
            "- Shooting: GUN visible, muzzle flash, victim drops\n"
            "- Arrest: POLICE/uniform restraining suspect, handcuffs\n\n"
            "CRITICAL: Ask yourself these questions:\n"
            "- Is someone TAKING an object? → Robbery/Stealing/Shoplifting\n"
            "- Is there a VEHICLE involved? → RoadAccidents\n"
            "- Is there FIRE or SMOKE? → Arson or Explosion\n"
            "- Are people hitting each other? → Fighting (mutual) or Assault (one-sided)\n"
            "- Is someone in UNIFORM? → Arrest\n"
            "- Is someone BREAKING IN? → Burglary\n\n"
            "Reply with ONLY: CATEGORY: <name>"
        )

        # MIL-guided + UCA 引導抽幀
        # 優先用 MIL snippet scores 找最可疑時段，搭配 UCA 背景
        video_path = (video_metadata or {}).get("video_path", "")
        n_keyframes = 8

        keyframes = self._mil_guided_frames(
            frames, video_path, video_metadata, n_keyframes, cv2, PILImage,
        )

        video_content = [{"type": "image", "image": img} for img in keyframes]

        # 組裝 messages
        if isinstance(video_content, dict):
            user_content = [video_content, {"type": "text", "text": prompt}]
        else:
            user_content = [*video_content, {"type": "text", "text": prompt}]

        messages = [{"role": "user", "content": user_content}]

        try:
            processor = self._report_tokenizer
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self._report_model.device)

            with torch.no_grad():
                output_ids = self._report_model.generate(
                    **inputs, max_new_tokens=128, temperature=0.1, do_sample=False,
                )
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True).strip()
            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
            logger.info(f"[VLM classify] {response}")

            # 提取 REASON 供報告使用
            reason_match = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE)
            if reason_match:
                self._vlm_reason = reason_match.group(1).strip()
            else:
                self._vlm_reason = ""

            # 解析
            cat_match = re.search(r"CATEGORY:\s*(\w+)", response, re.IGNORECASE)
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)

            if cat_match:
                raw = cat_match.group(1)
                for cat in UCF_CATEGORIES:
                    if cat.lower() == raw.lower():
                        conf = float(conf_match.group(1)) if conf_match else 0.7
                        return cat, min(conf, 1.0)

            # Fallback: 搜尋類別名稱
            for cat in UCF_CATEGORIES:
                if cat.lower() in response.lower():
                    return cat, 0.6

        except Exception as e:
            logger.warning(f"[VLM classify] 失敗：{e}")

        return None

    def _call_qwen3_vl(
        self,
        messages: List[Dict[str, str]],
        frames: List,
        temperature: Optional[float] = None,
        video_metadata: Dict = None,
    ) -> str:
        """
        Qwen3-VL 帶影片生成鑑定報告。
        優先用原生影片輸入，fallback 到 8 張靜態幀。
        """
        if self._report_model is None or self._report_tokenizer is None:
            logger.warning("[Planner] VLM 未就緒，使用 fallback")
            return ""

        import re, torch, cv2
        from PIL import Image as PILImage

        # UCA 引導抽幀（報告生成用 8 張）
        video_path = (video_metadata or {}).get("video_path", "")
        uca_segments = (video_metadata or {}).get("uca_segments", [])
        n_report_frames = 8

        report_frames = _extract_uca_guided_frames(video_path, uca_segments, n_report_frames, cv2, PILImage)
        if report_frames:
            visual_content = [{"type": "image", "image": img} for img in report_frames]
        else:
            valid = [f for f in frames if f is not None and hasattr(f, "shape")]
            visual_content = []
            if valid:
                n = len(valid)
                indices = [int(i * n / min(n_report_frames, n)) for i in range(min(n_report_frames, n))]
                visual_content = [
                    {"type": "image", "image": PILImage.fromarray(cv2.cvtColor(valid[idx], cv2.COLOR_BGR2RGB))}
                    for idx in indices
                ]

        # 從 build_report_prompt 的 messages 提取文字
        system_text = ""
        user_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                user_text = msg["content"]

        # 組裝 VLM messages（Qwen3-VL 要求所有 content 為 list 格式）
        vlm_messages = []
        if system_text:
            vlm_messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})

        user_content = list(visual_content)  # video or images
        user_content.append({
            "type": "text",
            "text": "以下是監視器影片。請根據影像內容與下方的分析資料撰寫鑑定報告。\n\n" + user_text,
        })
        vlm_messages.append({"role": "user", "content": user_content})

        try:
            processor = self._report_tokenizer
            temp = temperature or cfg.model.temperature
            max_new = cfg.model.max_new_tokens

            inputs = processor.apply_chat_template(
                vlm_messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self._report_model.device)

            with torch.no_grad():
                output_ids = self._report_model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=temp,
                    do_sample=temp > 0,
                    top_p=0.8,
                    top_k=20,
                )

            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            result = processor.decode(generated, skip_special_tokens=True).strip()
            result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL).strip()

            if not result:
                logger.warning("[Planner] Qwen3-VL 生成空字串")
                return ""
            return result

        except Exception as e:
            logger.error(f"[Planner] Qwen3-VL 生成失敗：{e}")
            return ""

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
