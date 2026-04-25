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
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentReport
from .action_emotion_agent import UCF_CATEGORIES
from .reflector import ReflectorAgent, ReflectorOutput
from config import cfg, is_crime, is_non_crime_anomaly, NON_CRIME_CATEGORIES
from rag.rag_module import LEGAL_ELEMENTS, GROUP_LEGAL_CONTEXT

# ── VLM 分類用類別清單（13 類 UCF + Normal）────────────────
# 不等同於 action_emotion_agent.UCF_CATEGORIES（13 類，MIL head 權重維度綁定）。
# 這裡多加 Normal 讓 VLM 可以主動回答「場景正常，無異常事件」，
# 避免當 VLM 對所有類別都不確定時仍被迫挑一個 crime 造成 FP。
VLM_CATEGORIES = list(UCF_CATEGORIES) + ["Normal"]

logger = logging.getLogger(__name__)


# ── Visual-to-Legal mapping 載入 ──────────────────────────
# 把 data/rag/mappings/visual_to_legal.json 的專家手編 visual cue
# 注入 VLM classify prompt，做 visual grounding。
# 同樣的 mapping 也會被 LLM-as-Judge 的 open-book γ 使用（見
# scripts/run_cross_evaluation.py），確保 VLM 與 Judge 看同一份
# 視覺-法律對應表，評分可對稱審計。
_VISUAL_TO_LEGAL_CACHE: Optional[Dict] = None


def _load_visual_to_legal() -> Dict:
    global _VISUAL_TO_LEGAL_CACHE
    if _VISUAL_TO_LEGAL_CACHE is not None:
        return _VISUAL_TO_LEGAL_CACHE
    import json as _json
    path = Path("data/rag/mappings/visual_to_legal.json")
    if not path.exists():
        logger.warning(f"[visual_to_legal] 找不到 {path}，VLM prompt 會少 visual grounding")
        _VISUAL_TO_LEGAL_CACHE = {}
        return {}
    data = _json.loads(path.read_text(encoding="utf-8"))
    # 過濾掉 _description / _usage 等 meta key
    _VISUAL_TO_LEGAL_CACHE = {k: v for k, v in data.items() if not k.startswith("_")}
    return _VISUAL_TO_LEGAL_CACHE


def format_priming_section(mil_top3: List[Tuple[str, float]]) -> str:
    """
    Pre-classification RAG Priming（Fix F1, 2026-04-24）。

    從 MIL Head top-3 預測 + `visual_to_legal.json` 取出對應 visual cues，
    作為 VLM 分類的先驗提示。與 `format_visual_cues_section` 的全類別定義互補：
      - 全類別定義：VLM 需要知道所有選項
      - Priming：聚焦 MIL 認為最可能的 3 類，告訴 VLM「特別留意這些特徵」

    設計注意：
      - 用「preliminary analysis」字樣避免 VLM 盲從
      - 保留 VLM 根據畫面推翻的空間（不強制）
      - MIL top-1 conf 過低（< 0.5）時 top-3 屬於 MIL 「無意見」雜訊，跳過 priming
        以避免錯誤先驗汙染 VLM（pilot_v8 觀察：MIL conf 0.32–0.69 時 priming
        會把錯誤類別 Visual cues 強塞給 VLM，導致 VLM 跟著錯，14-way 從 v6 的
        48.2% 降到 33.9%）。
    """
    if not mil_top3:
        return ""
    top1_conf = mil_top3[0][1] if mil_top3 else 0.0
    PRIMING_MIN_CONF = 0.5  # 從 0.2 提升至 0.5（pilot_v8 RCA, 2026-04-25）
    if top1_conf < PRIMING_MIN_CONF:
        return ""

    data = _load_visual_to_legal()
    lines = [
        "Preliminary feature analysis (motion + appearance embeddings) suggests the scene is most likely one of these:",
    ]
    for cat, prob in mil_top3:
        entry = data.get(cat)
        cue_bits = []
        if entry:
            mappings = entry.get("visual_mappings", [])
            direct_cues = [
                m["visual_cue"] for m in mappings
                if m.get("evidence_type") == "直接證據" and m.get("visual_cue")
            ]
            cue_bits = direct_cues[:2]
        cues_str = "；".join(cue_bits) if cue_bits else "(no cue data)"
        lines.append(f"  • {cat} (prior={prob:.2f}): look for — {cues_str}")
    lines.append(
        "Use these hints to focus attention, BUT rely on what you actually see "
        "in the frames. If the scene clearly does not match any of the above, "
        "choose the most fitting category from the full list."
    )
    return "\n".join(lines)


def format_visual_cues_section(categories: List[str]) -> str:
    """
    從 visual_to_legal.json 產出 VLM prompt 用的 Category Definitions section。

    格式（每行一類）：
        - Cat (刑法第X條/刑法第Y條): direct_cue_1；direct_cue_2 ...
    """
    data = _load_visual_to_legal()
    lines = []
    for cat in categories:
        if cat == "Normal":
            # Normal 是 residual class，列出「看到哪些 cue 就不要選 Normal」
            # 每類取第一個 direct-evidence cue（使用完整 cue，避免全形括號被切斷）
            excl_bits = []
            for other_cat in categories:
                if other_cat == "Normal" or other_cat not in data:
                    continue
                mappings = data[other_cat].get("visual_mappings", [])
                for m in mappings:
                    if m.get("evidence_type") == "直接證據" and m.get("visual_cue"):
                        excl_bits.append(f"{other_cat}: {m['visual_cue']}")
                        break
            excl_str = "；".join(excl_bits)
            lines.append(
                f"- Normal: Clearly ordinary scene with NO suspicious behavior. "
                f"**Do NOT pick Normal if you see ANY of** — {excl_str}."
            )
            continue
        entry = data.get(cat)
        if not entry:
            lines.append(f"- {cat}: (no cue data)")
            continue
        articles = entry.get("applicable_articles", [])
        articles_str = "／".join(articles[:2])
        mappings = entry.get("visual_mappings", [])
        direct_cues = [
            m["visual_cue"] for m in mappings
            if m.get("evidence_type") == "直接證據" and m.get("visual_cue")
        ]
        cues_str = "；".join(direct_cues[:3])
        lines.append(f"- {cat}（{articles_str}）：{cues_str}")

    # 重要 disambiguation（VLM 歷史混淆對）
    lines.append("")
    lines.append("IMPORTANT disambiguation (VLM has shown confusion on these pairs):")
    lines.append(
        "  • Shooting events often occur near parked vehicles — "
        "do NOT default to RoadAccidents just because cars appear."
    )
    lines.append(
        "  • Arson requires VISIBLE flames/smoke/ignition action — "
        "if you only see generic property damage, it's Vandalism, not Arson."
    )
    lines.append(
        "  • Burglary requires evidence of unauthorized entry (forced, "
        "broken windows, climbing) — if entry is normal, it may be "
        "Shoplifting (in stores) or Stealing (elsewhere)."
    )
    return "\n".join(lines)

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


# ── Non-Crime 異常描述 Prompt ──────────────────────────────
# 用於 Arrest / RoadAccidents / Explosion 等非犯罪類異常。
# 不涉及構成要件與法條對應，僅產出可觀察事實描述。
NON_CRIME_SYSTEM_PROMPT = """\
你是一位影片事件描述專家。當系統判定影片為「非犯罪類異常事件」時，\
請根據畫面內容撰寫客觀的事實描述，不作任何刑法上的推論。
"""

NON_CRIME_USER_TEMPLATE = """\
## 案件資訊
- 案件編號：{case_id}
- 事件類型：{event_type}（非犯罪類異常）
- 分析信心：{confidence:.2f}

## 描述要求
1. 以繁體中文撰寫
2. 僅描述影片可觀察事實，不作法律推論、不引用法條
3. 涵蓋：人物、動作、場所、關鍵行為與時間順序

## 輸出格式
### 一、事件摘要
（一句話說明事件性質，例如「交通事故」「執法人員逮捕行動」「不明原因爆炸」）

### 二、可觀察事實
（描述畫面中發生的事，包含人物、動作、場景）

### 三、備註
本事件經系統判定為非犯罪類異常（{event_type}），不進入刑法條文對應流程，\
亦不進行構成要件審查。
"""


def build_non_crime_report_prompt(
    case_id: str,
    event_type: str,
    confidence: float,
) -> List[Dict[str, str]]:
    """組裝非犯罪異常之描述 chat messages（不含法條、構成要件）。"""
    user_content = NON_CRIME_USER_TEMPLATE.format(
        case_id=case_id,
        event_type=event_type,
        confidence=confidence,
    )
    return [
        {"role": "system", "content": NON_CRIME_SYSTEM_PROMPT},
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
        self._rcost_threshold_high: int = 4  # pilot_v6 turns p75

        # Step 3b 報告生成模型（延遲載入）
        self._report_tokenizer = None
        self._report_model = None

        # 消融實驗 flags（由 pilot_experiment.py 設定）
        self._skip_vlm_classify: bool = False
        self._skip_vlm_report: bool = False
        self._vlm_reason: str = ""

        # Bias correction（由 pilot_experiment.py 用 --bias-correction 注入；
        # None 或空 dict 代表不做校正）
        self._bias_corrections: Optional[Dict[str, float]] = None

        # 2-stage anomaly detection（由 pilot_experiment.py 用 --anomaly-threshold
        # 注入；None 代表不 gate，全部走 13 類分類；float → escalation_score 低於
        # 此值直接輸出 Normal 並跳過 VLM 分類 + 報告生成）
        self._anomaly_threshold: Optional[float] = None

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

        # ── Step 2a：2-stage anomaly gate ──
        # 若 escalation_score 低於門檻 → 判為 Normal，跳過 VLM 分類與報告生成。
        # 注意：用 local variable bypass_vlm_{classify,report}，不能覆寫 self._skip_*
        # 因為 Planner instance 會在多個影片間共用。
        escalation_score = (
            ae_report.metadata.get("escalation_score", 0.0) if ae_report else 0.0
        )
        gated_normal = (
            self._anomaly_threshold is not None
            and escalation_score < self._anomaly_threshold
        )
        bypass_vlm_classify = self._skip_vlm_classify
        bypass_vlm_report = self._skip_vlm_report
        if gated_normal:
            crime_type = "Normal"
            if ae_report:
                ae_report.crime_category = "Normal"
                ae_report.metadata["anomaly_gated"] = True
                ae_report.metadata["anomaly_threshold"] = self._anomaly_threshold
            logger.info(
                f"[Planner] Step 2a: anomaly gate triggered — "
                f"escalation={escalation_score:.3f} < τ={self._anomaly_threshold:.3f} "
                f"→ Normal（跳過 VLM 分類/報告）"
            )
            bypass_vlm_classify = True
            bypass_vlm_report = True

        # 載入 Qwen3-VL（後續 Step 3b 報告生成也用同一個模型）
        if not bypass_vlm_classify or not bypass_vlm_report:
            self._load_report_model()

        if self._report_model is not None and not bypass_vlm_classify:
            vlm_result = self._vlm_classify(frames, video_metadata, ae_report=ae_report)
            if vlm_result:
                crime_type, vlm_conf = vlm_result
                # ── MIL + VLM Weighted Ensemble ──
                # 當 VLM 信心 < 0.7 且 MIL 給不同類別 + MIL 信心夠高（>0.55），
                # 走加權投票而非 VLM-only。歷史證據：MIL alone 15.4% + VLM alone
                # 34.6% → ensemble 42.3%（CLAUDE.md 紀錄）
                ensemble_used = False
                ensemble_score = {}
                if (
                    vlm_conf < 0.7
                    and mil_crime_type != crime_type
                    and mil_confidence > 0.55
                    and mil_crime_type not in ("Normal", "")
                ):
                    # 加權：VLM 0.7 + MIL 0.3，分數高者勝
                    ensemble_score[crime_type] = ensemble_score.get(crime_type, 0) + 0.7 * vlm_conf
                    ensemble_score[mil_crime_type] = ensemble_score.get(mil_crime_type, 0) + 0.3 * mil_confidence
                    winner = max(ensemble_score, key=ensemble_score.get)
                    if winner != crime_type:
                        logger.info(
                            f"[Planner] Step 2b ensemble: VLM={crime_type}({vlm_conf:.2f}) "
                            f"+ MIL={mil_crime_type}({mil_confidence:.2f}) → {winner}"
                        )
                        crime_type = winner
                        ensemble_used = True
                ae_confidence = vlm_conf
                if ae_report:
                    ae_report.crime_category = crime_type
                    ae_report.confidence = vlm_conf
                    ae_report.metadata["mil_crime_type"] = mil_crime_type
                    ae_report.metadata["mil_confidence"] = mil_confidence
                    ae_report.metadata["vlm_used"] = True
                    ae_report.metadata["ensemble_used"] = ensemble_used
                    ae_report.metadata["ensemble_score"] = ensemble_score
                if not ensemble_used:
                    logger.info(
                        f"[Planner] Step 2b: VLM 分類 MIL={mil_crime_type}({mil_confidence:.2f}) "
                        f"→ VLM={crime_type}({vlm_conf:.2f})"
                    )
                self._total_turns += 1

                # ── VLM 判定為 Normal：跳過 Step 3b 的犯罪報告生成 ──
                # VLM 主動回 Normal 代表系統認為場景無異常，不需要產出犯罪鑑定報告。
                # 仍會進入 _synthesize_final_report 組一份 Normal 最終報告。
                if crime_type == "Normal":
                    bypass_vlm_report = True
                    if ae_report:
                        ae_report.metadata["vlm_predicted_normal"] = True
                    logger.info(
                        "[Planner] Step 2b: VLM 判定 Normal → 跳過犯罪報告生成"
                    )

        # ── Step 2c-2d：RAG-guided Classification Verification（verify-only）──
        # 對所有信心 > 0.3 的 crime 類做要件符合度檢核，結果存入 ae_report.metadata：
        #   metadata["rag_element_match"]        = (matched, total) | None
        #   metadata["rag_element_match_ratio"]  = matched/total    | None
        # Reflector L2 會用此 ratio 觸發 SOFT（ratio 低 + 高信心 = 分類可能錯誤）
        # 並在 Dempster-Shafer _mass 中對 m("crime") 做校準乘數。
        # **不強換分類**（Fix #3, pilot_v3 Shooting033 案例）— 見 _rag_verify_classification。
        if (self._report_model is not None
                and not bypass_vlm_classify
                and is_crime(crime_type)
                and self.rag
                and ae_confidence > 0.3):
            rag_result = self._rag_verify_classification(
                frames, video_metadata, crime_type,
            )
            if rag_result is not None and ae_report is not None:
                matched, total = rag_result
                ratio = matched / max(total, 1)
                ae_report.metadata["rag_element_match"] = (matched, total)
                ae_report.metadata["rag_element_match_ratio"] = ratio

        # ── Non-Crime 短路分支 ──────────────────────────────
        # crime_type ∈ {Arrest, RoadAccidents, Explosion}：系統仍視為異常
        # （MIL Head 已偵測到），但不進入 H-RAG + Rlegal + Reflector。
        # 僅生成事實描述並直接回傳最終報告。
        # 詳見 /home/yuting/.claude/plans/ucf-polished-nova.md。
        if is_non_crime_anomaly(crime_type):
            logger.info(
                f"[Planner] Non-crime anomaly detected: {crime_type} → "
                f"skip H-RAG + Rlegal + Reflector"
            )
            temperature = video_metadata.get("temperature", cfg.model.temperature)
            non_crime_text = ""
            if self._report_model is not None and not bypass_vlm_report:
                non_crime_messages = build_non_crime_report_prompt(
                    case_id=case_id,
                    event_type=crime_type,
                    confidence=ae_confidence,
                )
                non_crime_text = self._call_qwen3_vl(
                    non_crime_messages, frames, temperature=temperature,
                    video_metadata=video_metadata,
                )
            if not non_crime_text:
                non_crime_text = (
                    f"本案經系統判定為非犯罪類異常事件（{crime_type}），"
                    f"未進入刑法條文對應流程。"
                )
                self._report_method = "non-crime-template"
            else:
                self._report_method = "qwen3-vl-non-crime"
            return self._synthesize_non_crime_report(
                case_id=case_id,
                crime_type=crime_type,
                ae_confidence=ae_confidence,
                ae_report=ae_report,
                reports=list(reports.values()),
                generated_report_text=non_crime_text,
                low_reliability=low_reliability,
            )

        # ── Step 3：法律整合（3a RAG → 3b 報告生成 → 3c Rlegal）──
        rag_results: Dict = {"laws": [], "judgments": []}
        rlegal = 0.0

        # Step 3a：RAG 查詢取候選法條
        if ae_confidence >= cfg.inference.confidence_low_threshold and self.rag and is_crime(crime_type):
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
        if not bypass_vlm_report:
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
        # 非犯罪類已在 Step 2d 後短路離開，此處僅處理 CRIME_CATEGORIES。
        if self.rag and is_crime(crime_type):
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
        anomaly_gated = (
            ae_report_for_cat
            and ae_report_for_cat.metadata.get("anomaly_gated", False)
        )
        if anomaly_gated:
            # 2-stage gate 已判 Normal：覆蓋 Reflector / VLM 的任何分類
            crime_type = "Normal"
        elif vlm_used:
            # ── HARD conflict re-vote ──
            # Pilot v2 觀察：10/56 HARD 衝突發生但 VLM 仍是 final_category。
            # 改進：HARD 且 VLM confidence < 0.6 時，採信 Reflector consensus
            # （MIL 加 audit 的綜合判斷）。high-conf VLM (≥ 0.6) 仍用 VLM 結果。
            vlm_cat = ae_report_for_cat.crime_category
            vlm_conf = ae_report_for_cat.confidence
            ref_consensus = final_audit.consensus_category
            if (
                final_audit.conflict_type == "HARD"
                and vlm_conf < 0.6
                and ref_consensus
                and ref_consensus != vlm_cat
                and ref_consensus != "Normal"
            ):
                logger.info(
                    f"[Planner] HARD revote: VLM={vlm_cat}({vlm_conf:.2f}) "
                    f"→ Reflector consensus={ref_consensus}"
                )
                crime_type = ref_consensus
                ae_report_for_cat.metadata["hard_revote"] = True
            else:
                crime_type = vlm_cat
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
            # 類別性質旗標（供 evaluation script 過濾；非犯罪類別會在 run() 中短路）
            "is_non_crime_anomaly": False,
            "event_category": "normal" if crime_type == "Normal" else "crime",
            # 2-stage anomaly detection outputs（D 計畫；供 AUROC / NDCF 評估用）
            "escalation_score": escalation_score,
            "anomaly_gated": bool(
                ae_report and ae_report.metadata.get("anomaly_gated", False)
            ),
            "anomaly_threshold": self._anomaly_threshold,
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

    # ── Non-Crime 最終報告整合 ───────────────────────────
    #
    # 非犯罪類（Arrest / RoadAccidents / Explosion）短路分支使用。
    # 不經 Reflector 審核，不做 Rlegal / 法條引用；輸出與 _synthesize_final_report
    # 同形狀，差異在：
    #   - is_non_crime_anomaly = True
    #   - legal_classification 所有欄位空置
    #   - rcons = 1.0 (無衝突)、rlegal = 0.0、conflict_type = "NONE"

    def _synthesize_non_crime_report(
        self,
        case_id: str,
        crime_type: str,
        ae_confidence: float,
        ae_report: Optional[AgentReport],
        reports: List[AgentReport],
        generated_report_text: str,
        low_reliability: bool,
    ) -> Dict[str, Any]:
        weights = self._get_eval_weights(crime_type)
        rcost = compute_rcost(
            self._total_turns,
            threshold_low=4,
            threshold_high=self._rcost_threshold_high,
        )

        causal_chain = ae_report.metadata.get("causal_chain", "") if ae_report else ""
        escalation_score = ae_report.metadata.get("escalation_score", 0.0) if ae_report else 0.0
        rationale = ae_report.metadata.get("rationale", "") if ae_report else ""

        # 保持與 _build_uncertainty_notes 同樣的 Dict[str, List[str]] 結構，
        # 供 _save_case_report 用 .get() 查表渲染。
        low_conf = []
        insufficient = [
            f"本案為非犯罪類異常（{crime_type}），未進行構成要件審查與法條對應"
        ]
        if low_reliability:
            insufficient.append("環境可信度過低，視覺證據品質不足")
        for r in reports:
            if getattr(r, "confidence", 1.0) < 0.6 and r.crime_category != "ENVIRONMENTAL_ASSESSMENT":
                low_conf.append(f"{r.agent_name}: conf={r.confidence:.2f}")
        uncertainty_notes = {
            "low_confidence_items": low_conf,
            "conflicting_evidence": [],
            "insufficient_evidence": insufficient,
        }

        final_report = {
            "case_id": case_id,
            "fact_finding": {
                "description": generated_report_text,
                "rationale": rationale,
                "supporting_frames": self._collect_key_frames(reports),
                "confidence": ae_confidence,
            },
            "behavior_analysis": {
                "causal_chain": causal_chain,
                "escalation_score": escalation_score,
                "pre_crime_indicators": (
                    ae_report.metadata.get("pre_crime_indicators", []) if ae_report else []
                ),
                "post_crime_indicators": (
                    ae_report.metadata.get("post_crime_indicators", []) if ae_report else []
                ),
                "crime_group": ae_report.metadata.get("crime_group", "") if ae_report else "",
                "severity": ae_report.metadata.get("crime_type_severity", "") if ae_report else "",
            },
            "legal_classification": {
                "applicable_articles": [],
                "elements_covered": [],
                "coverage_rate": 0.0,
                "rag_laws": [],
            },
            "uncertainty_notes": uncertainty_notes,
            "rcons":  1.0,
            "rlegal": 0.0,
            "rcost":  rcost,
            "eval_weights": weights,
            "total_turns": self._total_turns,
            "final_category": crime_type,
            # 非犯罪異常旗標（供 evaluation script 過濾）
            "is_non_crime_anomaly": True,
            "event_category": "non_crime",
            "escalation_score": escalation_score,
            "anomaly_gated": bool(
                ae_report and ae_report.metadata.get("anomaly_gated", False)
            ),
            "anomaly_threshold": self._anomaly_threshold,
            "is_convergent": True,
            "conflict_type": "NONE",
            "report_generation_method": getattr(self, "_report_method", "unknown"),
            "agent_reports": [r.to_dict() for r in reports],
            "audit_log": [],
            "debate_log": self.reflector.get_debate_log(),
        }

        logger.info(
            f"[Planner] 最終報告（非犯罪） | case={case_id} | category={crime_type} | "
            f"confidence={ae_confidence:.3f} | turns={self._total_turns}"
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
    ) -> Optional[Tuple[int, int]]:
        """
        Step 2c-2d: 用 RAG 的法律構成要件驗證 VLM 分類。

        回傳：
            `(matched, total)` 解析成功時的要件匹配統計，供 Reflector Rcons
            校準 + 低匹配率 SOFT 觸發；解析失敗或不適用回傳 None。
            **不強換分類**（Fix #3, pilot_v3 Shooting033 案例）。
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

        elements_str = "\n".join(f"- {e}" for e in elements)

        # 僅驗證要件符合度，不強迫 VLM 改換類別（見 pilot_v3 Shooting033 案例）。
        # 若要件不符，後續流程會保留 VLM 原判，由信心分數與 Reflector 處理。
        verify_prompt = (
            f"You classified this CCTV footage as: {initial_category}\n\n"
            f"Required legal elements for {initial_category}:\n"
            f"{elements_str}\n\n"
            f"Look at these frames carefully. For each element, is it VISIBLE?\n"
            f"Count how many elements you can actually see.\n\n"
            f"Reply with ONLY:\n"
            f"ELEMENTS_MATCHED: <number>/{len(elements)}"
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
                if ratio >= 0.5:
                    logger.info(
                        f"[RAG verify] {matched}/{total_el} 要件符合 → 保留 {initial_category}"
                    )
                    return (matched, total_el)
                # 要件不足：不強換另一類別（pilot_v3 Shooting033 案例）。
                # 回傳 (matched, total) 供 Reflector 做 Rcons 校準與 SOFT 觸發，
                # 但 VLM 分類本身仍保留。
                logger.info(
                    f"[RAG verify] {matched}/{total_el} 要件符合 → 要件不足，"
                    f"保留 VLM 原判 {initial_category}（Reflector 會降低 Rcons）"
                )
                return (matched, total_el)

            # ELEMENTS_MATCHED 格式解析失敗：保守保留原判
            logger.warning(
                f"[RAG verify] 無法解析 ELEMENTS_MATCHED → 保留 {initial_category}"
            )

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
        延遲載入 Qwen3-VL-32B-Instruct + QLoRA adapter（統一 VLM）。
        同一個模型負責 Step 2b 分類 + Step 3b 報告生成。
        支援 INT4 量化 + LoRA adapter 載入。
        """
        if self._report_model is not None:
            return
        try:
            import torch as _torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            model_name = cfg.model.report_model
            adapter_path = getattr(cfg.model, "vlm_adapter", "")
            logger.info(f"[Planner] 載入 VLM：{model_name}")

            self._report_tokenizer = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True,
            )

            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
            }

            # INT4 量化（32B 模型需要）
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=_torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("[Planner] INT4 量化已啟用")
            except ImportError:
                # 沒有 bitsandbytes → 用 BF16（僅 8B 以下模型可用）
                if cfg.model.torch_dtype == "bfloat16" and _torch.cuda.is_available():
                    load_kwargs["torch_dtype"] = _torch.bfloat16
                else:
                    load_kwargs["torch_dtype"] = "auto"

            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, **load_kwargs,
            )

            # 載入 QLoRA adapter（如果存在）
            if adapter_path and Path(adapter_path).exists():
                from peft import PeftModel
                self._report_model = PeftModel.from_pretrained(base_model, adapter_path)
                self._report_model.eval()
                logger.info(f"[Planner] VLM + LoRA adapter 載入完成：{adapter_path}")
            else:
                self._report_model = base_model
                logger.info(f"[Planner] VLM 載入完成（無 adapter）：{model_name}")
        except Exception as e:
            logger.warning(f"[Planner] 無法載入 VLM：{e}，將使用 fallback")
            self._report_model = None

    def _vlm_classify(
        self,
        frames: List,
        video_metadata: Dict = None,
        ae_report=None,
    ) -> Optional[tuple]:
        """
        Qwen3-VL 犯罪分類。
        優先使用原生影片輸入（動態資訊），fallback 到 8 張靜態幀。

        ae_report 若提供，會讀 metadata["mil_top3"] 做 Pre-classification RAG Priming。
        """
        if self._report_model is None or self._report_tokenizer is None:
            return None

        import torch, re, cv2
        from PIL import Image as PILImage

        # Prompt with auto-generated visual cue section from
        # data/rag/mappings/visual_to_legal.json (專家手編 visual → legal mapping)
        # — same source fed to LLM-as-Judge open-book γ for symmetric auditability.
        categories_str = ", ".join(sorted(VLM_CATEGORIES))
        cue_section = format_visual_cues_section(sorted(VLM_CATEGORIES))

        # Pre-classification RAG Priming：若 MIL top-3 可用，加入先驗提示
        priming_section = ""
        if ae_report is not None:
            mil_top3 = ae_report.metadata.get("mil_top3", [])
            priming_section = format_priming_section(mil_top3)

        prompt_parts = [
            "You are a forensic surveillance video analyst.",
            "Look at these frames from a CCTV video and determine what category the scene falls into.",
            "If the footage shows no criminal, hazardous, or law-enforcement activity, reply Normal.",
            "",
        ]
        if priming_section:
            prompt_parts += [priming_section, ""]
        prompt_parts += [
            f"Choose ONE category from: {categories_str}",
            "",
            "Category definitions (visual cues → applicable articles):",
            cue_section,
            "",
            "Reply with ONLY: CATEGORY: <name>",
        ]
        prompt = "\n".join(prompt_parts)

        # 均勻 8 幀（跟 standalone 診斷一致，MIL-guided 反而引入偏差）
        from .frame_utils import uniform_keyframes, fallback_from_frame_list
        video_path = (video_metadata or {}).get("video_path", "")
        n_keyframes = 8
        keyframes = uniform_keyframes(video_path, n_keyframes)
        if not keyframes:
            keyframes = fallback_from_frame_list(frames, n_keyframes)

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
                gen_out = self._report_model.generate(
                    **inputs, max_new_tokens=128, temperature=0.1, do_sample=False,
                    output_scores=True, return_dict_in_generate=True,
                )
            output_ids = gen_out.sequences
            scores = gen_out.scores  # tuple[tensor(1, vocab_size), ...] per generation step
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

            # 解析類別（含 Normal）
            cat_match = re.search(r"CATEGORY:\s*(\w+)", response, re.IGNORECASE)
            predicted_cat = None
            if cat_match:
                raw = cat_match.group(1)
                for cat in VLM_CATEGORIES:
                    if cat.lower() == raw.lower():
                        predicted_cat = cat
                        break
            if predicted_cat is None:
                # Fallback: 在 response 中找任一類別名
                for cat in VLM_CATEGORIES:
                    if cat.lower() in response.lower():
                        predicted_cat = cat
                        break
            if predicted_cat is None:
                return None

            # 從 generate 的 logits 計算真實 confidence
            # 幾何平均 p(category_token | prefix)，範圍 [0, 1]；
            # 若抽不到則退回 0.5（中性），不再硬編 0.7
            diag = self._classify_diagnostics(
                scores, generated, processor.tokenizer, predicted_cat,
                list(VLM_CATEGORIES),
                bias_corrections=self._bias_corrections,
            )
            conf = diag["confidence"]
            logger.info(
                f"[VLM classify] conf={conf:.3f}  top1={diag['top1_prob']:.3f}  "
                f"top2={diag['top2_prob']:.3f}  margin={diag['margin']:.3f}  "
                f"entropy={diag['entropy']:.3f}  "
                f"top3={diag['top3']}"
            )

            # Bias correction：校正後若 top-1 與 greedy 預測不同，覆寫並重新計算 conf
            if self._bias_corrections and diag.get("corrected_top1"):
                new_cat, new_prob = diag["corrected_top1"]
                if new_cat != predicted_cat:
                    logger.info(
                        f"[VLM classify] bias-correct: {predicted_cat} → {new_cat}"
                        f"  (corrected top1={new_prob:.3f}, "
                        f"orig prob of {predicted_cat}="
                        f"{diag['all_probs'].get(predicted_cat, 0):.3f})"
                    )
                    predicted_cat = new_cat
                    conf = new_prob  # 改用校正後 softmax 機率當 confidence
            return predicted_cat, conf

        except Exception as e:
            logger.warning(f"[VLM classify] 失敗：{e}")

        return None

    @staticmethod
    def _classify_diagnostics(
        scores,
        generated_ids,
        tokenizer,
        predicted_cat: str,
        all_categories,
        bias_corrections: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        對第一個類別 token 的位置抽 top-k 診斷資訊，給 calibration / 門檻調整用。

        Returns dict with:
          confidence       — 幾何平均 token prob（同 _confidence_from_scores）
          top1_prob        — 在該位置、13 類 first-token 的 softmax 中，top-1 機率
          top2_prob        — 同上 top-2
          margin           — top1 − top2（越大越確定）
          entropy          — 13 類限定的 entropy
          top3             — [(cat, prob), ...] 前三（未校正）
          all_probs        — {cat: prob} 13 類完整分布（未校正）
          corrected_top1   — (cat, prob) bias 校正後的 top-1，未啟用時為 None
          corrected_all    — {cat: prob} bias 校正後的完整分布
        """
        import math
        import torch

        fallback = {
            "confidence": PlannerAgent._confidence_from_scores(
                scores, generated_ids, tokenizer, predicted_cat
            ),
            "top1_prob": 0.0, "top2_prob": 0.0, "margin": 0.0,
            "entropy": 0.0, "top3": [], "all_probs": {},
            "corrected_top1": None, "corrected_all": {},
        }
        if scores is None or len(scores) == 0:
            return fallback

        # 預先建每類第一個 token 的 id（前後綴空白兩種嘗試）
        cat_first: Dict[str, int] = {}
        for cat in all_categories:
            chosen = None
            for prefix in (" ", ""):
                try:
                    ids = tokenizer.encode(prefix + cat, add_special_tokens=False)
                    if ids:
                        chosen = ids[0]
                        break
                except Exception:
                    continue
            if chosen is not None:
                cat_first[cat] = chosen
        if predicted_cat not in cat_first:
            return fallback

        # 找到 predicted_cat 第一個 token 在 generated_ids 的位置
        gen_list = generated_ids.tolist() if hasattr(generated_ids, "tolist") else list(generated_ids)
        pred_tid = cat_first[predicted_cat]
        try:
            decision_pos = gen_list.index(pred_tid)
        except ValueError:
            return fallback
        if decision_pos >= len(scores):
            return fallback

        logits = scores[decision_pos][0]  # (vocab_size,)
        # 限定在 13 類 first-token 上做 softmax，避免和無關 token 稀釋
        ids_list = list(cat_first.values())
        cat_list = list(cat_first.keys())
        # 有些類別第一個 token id 可能重複（A 開頭的多類同 token）—— 去重並對應回最早的類
        unique: Dict[int, str] = {}
        for cat, tid in cat_first.items():
            if tid not in unique:
                unique[tid] = cat
        unique_ids = list(unique.keys())
        unique_cats = [unique[tid] for tid in unique_ids]

        sub_logits = torch.stack([logits[tid] for tid in unique_ids])
        probs = torch.softmax(sub_logits, dim=-1)
        probs_list = probs.tolist()

        ranked = sorted(
            zip(unique_cats, probs_list), key=lambda x: x[1], reverse=True
        )
        top1_prob = ranked[0][1] if ranked else 0.0
        top2_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top1_prob - top2_prob
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs_list)
        top3 = [(c, round(p, 3)) for c, p in ranked[:3]]
        all_probs = {cat: round(p, 4) for cat, p in zip(unique_cats, probs_list)}

        # ── Bias correction（過度預測 Robbery/Burglary 的對策） ──
        corrected_top1 = None
        corrected_all: Dict[str, float] = {}
        if bias_corrections:
            bias_vec = torch.tensor(
                [bias_corrections.get(cat, 0.0) for cat in unique_cats],
                dtype=sub_logits.dtype, device=sub_logits.device,
            )
            adj_logits = sub_logits - bias_vec
            adj_probs = torch.softmax(adj_logits, dim=-1).tolist()
            ranked_adj = sorted(
                zip(unique_cats, adj_probs), key=lambda x: x[1], reverse=True
            )
            corrected_top1 = (ranked_adj[0][0], round(float(ranked_adj[0][1]), 4))
            corrected_all = {c: round(p, 4) for c, p in zip(unique_cats, adj_probs)}

        conf_geo = PlannerAgent._confidence_from_scores(
            scores, generated_ids, tokenizer, predicted_cat
        )
        return {
            "confidence": conf_geo,
            "top1_prob": float(top1_prob),
            "top2_prob": float(top2_prob),
            "margin": float(margin),
            "entropy": float(entropy),
            "top3": top3,
            "all_probs": all_probs,
            "corrected_top1": corrected_top1,
            "corrected_all": corrected_all,
        }

    @staticmethod
    def _confidence_from_scores(
        scores,
        generated_ids,
        tokenizer,
        predicted_cat: str,
    ) -> float:
        """
        從 greedy 生成的 per-step logits 抽出預測類別 tokens 的幾何平均機率。

        Returns
        -------
        float in [0, 1]
            幾何平均 ∏ p(token_i | prefix) ^ (1 / n_tokens)
            找不到匹配時回 0.5（中性）
        """
        import math
        import torch

        if scores is None or len(scores) == 0:
            return 0.5

        # 嘗試兩種 tokenization（前綴空白 / 無空白）以適配不同 tokenizer
        candidates = []
        for prefix in (" ", ""):
            try:
                ids = tokenizer.encode(prefix + predicted_cat, add_special_tokens=False)
                if ids:
                    candidates.append(ids)
            except Exception:
                continue
        if not candidates:
            return 0.5

        gen_list = generated_ids.tolist() if hasattr(generated_ids, "tolist") else list(generated_ids)

        start_pos = None
        matched_ids = None
        for target in candidates:
            n = len(target)
            if n == 0 or n > len(gen_list):
                continue
            for i in range(len(gen_list) - n + 1):
                if gen_list[i:i + n] == target:
                    start_pos = i
                    matched_ids = target
                    break
            if start_pos is not None:
                break

        if start_pos is None or matched_ids is None:
            return 0.5

        log_probs = []
        for offset, tid in enumerate(matched_ids):
            step = start_pos + offset
            if step >= len(scores):
                break
            logits = scores[step][0]
            prob = torch.softmax(logits, dim=-1)[tid].item()
            log_probs.append(math.log(max(prob, 1e-10)))

        if not log_probs:
            return 0.5
        return float(math.exp(sum(log_probs) / len(log_probs)))

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
