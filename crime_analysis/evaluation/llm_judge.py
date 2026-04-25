"""
LLM-as-a-Judge 評估模組
兩種評估方式：
  1. Pairwise 比較 - 用於 DPO 偏好對建構（AB/BA 雙向比較校正 Position Bias）
  2. Rubric 評分 - 用於消融實驗量化（五維度，總分 25）

Judge Model：Claude（避免 Self-Enhancement Bias，支援 Gemini/OpenAI fallback）
Position Bias 控制：正反各跑一次取平均
穩定性標記：兩次分數差距 > 3 分時警告
"""
from typing import Any, Dict, List, Optional
import logging
import json

# 自動從 .env 載入 API keys；shell exports 仍優先（override=False）
try:
    from env_loader import load_dotenv
    load_dotenv()
except ImportError:
    pass  # env_loader 可能在部分 import 路徑下不可達，讓使用者自行 export

from config import cfg

logger = logging.getLogger(__name__)

# ── 7 題 Rubric（2026-04-23 更新）──────────────────────────
# 每題有具體 anchor + 浮動最高分；評分後正規化到 0–1 相加取平均得 overall。
# 設計理由：
#   - 法律無絕對 GT，Q1 評「情景合理性」而非「絕對正確」
#   - Q2 要件覆蓋率 max 依報告自己主張的罪名（動態 k）
#   - Q7 用 UCA 情景描述當事實 GT（唯一錨點題）
#   - 與人類評分者用相同 rubric，ICC 才有意義
#
# Q2 的 max_score=None 表示動態：由 open_book_context 裡法條的 elements 決定
RUBRIC_QUESTIONS = {
    "q1_legal_relevance": {
        "max_score": 3,
        "description": "法條引用情景合理性（A=3 條號+項完整且對情景合理；B=2 條號正確但格式不完整；C=1 條號本身有效但對情景不相關；D=0 未引用）",
    },
    "q2_element_coverage": {
        "max_score": None,   # 動態，由報告主張罪名的要件數 k 決定
        "description": "構成要件覆蓋率（每覆蓋一個要件 1 分；要件清單依報告自己主張的罪名）",
    },
    "q3_evidence_traceability": {
        "max_score": 3,
        "description": "影像證據可追溯性（A=3 全部主張有幀號；B=2 多數>50%；C=1 少數<50%；D=0 無）",
    },
    "q4_causal_chain": {
        "max_score": 4,
        "description": "因果鏈完整性（犯罪前/中/後各 1 分 + 三階段有清楚因果連結 +1 分，共 4 分）",
    },
    "q5_uncertainty_marking": {
        "max_score": 3,
        "description": "不確定性標記適當性（A=3 有說明限制+建議複查；B=2 有提但無具體原因；C=3 影片品質好不需標記；D=0 影片有問題卻仍強肯定）",
    },
    "q6_judicial_language": {
        "max_score": 3,
        "description": "司法語言規範性（A=3 用語精確、主體統一；B=2 1-2 處口語化；C=1 多處口語但有實質內容；D=0 術語明顯錯誤）",
    },
    "q7_scenario_fidelity": {
        "max_score": 3,
        "description": "情景事實吻合度（對照 UCA 情景 GT。A=3 全部事實吻合；B=2 絕大部分吻合 1-2 處小出入；C=1 多處出入或過度推論；D=0 明顯與情景矛盾）",
    },
}

# 正規化差距門檻（超過視為兩次評分不穩定）
# raw 最大差 2 分、max=7 題平均後約 0.29
STABILITY_THRESHOLD_NORM = 0.29

# ── Backward compat（舊 5 維 rubric；DPO 相關 legacy code 可能還用到）──
RUBRIC_DIMENSIONS = {
    "logical_consistency": "報告的推理鏈是否邏輯自洽，無矛盾（對應 Rcons）",
    "legal_coverage": "報告是否涵蓋足夠的法律構成要件並引用具體法條條號（對應 Rlegal）",
    "evidence_citation": "是否精確標註對應的影像幀與時間點，區分觀察事實與推論",
    "causal_reasoning": "是否建立清晰的因果鏈（前兆行為 → 犯罪實施 → 事後反應）",
    "uncertainty_marking": "是否明確標記不確定性、分析限制、和需進一步調查的事項",
}
STABILITY_THRESHOLD = 3.0  # 舊 raw 1-5 差距門檻（for pairwise_compare double-check 等舊路徑）


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    從 LLM 回應中提取 JSON。
    處理三種常見格式：純 JSON、```json``` 包裹、混合文字中的 JSON。
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"[LLMJudge] 無法解析 JSON 回應：{text[:200]}")
    return {}


class LLMJudge:
    """
    LLM-as-a-Judge

    預設 Judge Model：Claude（避免使用 Qwen 作為 Judge，防止 Self-Enhancement Bias）
    支援：Claude / Gemini / OpenAI 三後端

    Rubric：5 維度 × 1-5 分 = 總分 25
    """

    def __init__(self, judge_model: str = None, api_key: str = None, budget_limit_usd: float = None):
        self.judge_model = judge_model or cfg.dpo.judge_model
        self._api_key = api_key
        self._client = None

        # 判斷 backend 類型
        if "claude" in self.judge_model.lower():
            self._backend = "claude"
        elif "gemini" in self.judge_model.lower():
            self._backend = "gemini"
        else:
            self._backend = "openai"

        # ── Token Budgeting ──
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_calls: int = 0
        self._budget_limit_usd: float = budget_limit_usd or 20.0  # 預設上限 $20

        # 各模型定價（USD per 1M tokens）
        self._pricing = {
            "claude": {"input": 3.0, "output": 15.0},
            "gemini": {"input": 0.15, "output": 0.60},  # Flash
            "openai": {"input": 2.50, "output": 10.0},   # GPT-4o
        }

    @property
    def total_cost_usd(self) -> float:
        """目前累計花費（USD）。"""
        p = self._pricing.get(self._backend, {"input": 3.0, "output": 15.0})
        return (self._total_input_tokens * p["input"] + self._total_output_tokens * p["output"]) / 1_000_000

    @property
    def token_summary(self) -> dict:
        """Token 使用摘要。"""
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "budget_limit_usd": self._budget_limit_usd,
            "budget_remaining_usd": round(self._budget_limit_usd - self.total_cost_usd, 4),
        }

    def _check_budget(self):
        """檢查是否超出預算。"""
        if self.total_cost_usd >= self._budget_limit_usd:
            raise RuntimeError(
                f"[LLMJudge] 已超出預算上限！"
                f"已花費 ${self.total_cost_usd:.2f} / 上限 ${self._budget_limit_usd:.2f}"
                f"（{self._total_calls} 次呼叫）"
            )

    def _get_client(self):
        """延遲初始化 LLM client（支援 Claude / Gemini / OpenAI）。"""
        if self._client is not None:
            return self._client

        if self._backend == "claude":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
                logger.info(f"[LLMJudge] 初始化 Claude client，模型：{self.judge_model}")
            except ImportError:
                raise ImportError("需要安裝 anthropic：pip install anthropic")
        elif self._backend == "gemini":
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                logger.info(f"[LLMJudge] 初始化 Gemini client，模型：{self.judge_model}")
            except ImportError:
                raise ImportError("需要安裝 google-genai：pip install google-genai")
        else:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
                logger.info(f"[LLMJudge] 初始化 OpenAI client，模型：{self.judge_model}")
            except ImportError:
                raise ImportError("需要安裝 openai：pip install openai")

        return self._client

    # ── Pairwise 比較 ─────────────────────────────────────

    def pairwise_compare(
        self,
        prompt: str,
        report_a: str,
        report_b: str,
        video_id: str = "",
        double_check: bool = False,
    ) -> Dict[str, Any]:
        """
        Pairwise 比較。預設單向（省 token），可選 AB/BA 雙向校正。
        """
        result_ab = self._call_judge(
            prompt, report_a, report_b, first_label="A", second_label="B"
        )

        if not double_check:
            return {
                "winner": result_ab.get("winner", "tie"),
                "score_a": result_ab.get("score_first", 0.0),
                "score_b": result_ab.get("score_second", 0.0),
                "is_consistent": True,
                "is_stable": True,
                "rationale": result_ab.get("rationale", ""),
                "video_id": video_id,
            }

        # 雙向校正（DPO 偏好對建構時使用）
        result_ba = self._call_judge(
            prompt, report_b, report_a, first_label="B", second_label="A"
        )
        winner_ab = result_ab.get("winner", "tie")
        winner_ba = result_ba.get("winner", "tie")
        is_consistent = winner_ab == winner_ba

        avg_score_a = (result_ab.get("score_first", 0.0) + result_ba.get("score_second", 0.0)) / 2
        avg_score_b = (result_ab.get("score_second", 0.0) + result_ba.get("score_first", 0.0)) / 2

        return {
            "winner": winner_ab if is_consistent else "tie",
            "score_a": avg_score_a,
            "score_b": avg_score_b,
            "is_consistent": is_consistent,
            "is_stable": True,
            "rationale": result_ab.get("rationale", ""),
            "video_id": video_id,
        }

    # ── Rubric 評分（7 題 rubric，支援 open-book 參考）────────

    def rubric_score(
        self,
        prompt: str,
        report: str,
        crime_type: str = "",
        double_check: bool = False,
        open_book_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        7 題 Rubric 評分（2026-04-23 版）。

        open_book_context（選填）格式：
            {
              "scenario_description": "...",  # UCA 情景描述
              "cited_articles": [             # 報告引用的法條
                {"article": "刑法第 277 條", "full_text": "...", "elements": [...]}
              ],
            }

        回傳：
            q_scores:   {q_key: raw_score}
            q_max:      {q_key: max}                   # Q2 動態
            q_norm:     {q_key: normalized [0,1]}
            overall:    平均正規化分數 [0,1]
            feedback:   裁判文字評語
            is_stable:  兩次評分 max_norm_diff < STABILITY_THRESHOLD_NORM
        """
        # 決定 Q2 的 max_score（若報告引用法條帶有 elements 清單，依其長度）
        q2_max = 4  # 預設 fallback
        if open_book_context and open_book_context.get("cited_articles"):
            elements_counts = [
                len(a.get("elements", []))
                for a in open_book_context["cited_articles"]
                if a.get("elements")
            ]
            if elements_counts:
                q2_max = max(elements_counts)   # 取引用法條中要件最多者
        q_max = {
            k: (q2_max if k == "q2_element_coverage" else v["max_score"])
            for k, v in RUBRIC_QUESTIONS.items()
        }

        judge_prompt = self._build_rubric_prompt(prompt, report, open_book_context, q_max)
        scores_1 = self._call_rubric(judge_prompt, q_max)

        if not double_check:
            return self._pack_rubric_result(scores_1, None, q_max, scores_1.get("feedback", ""))

        # 穩定性檢查（2x token）
        scores_2 = self._call_rubric(judge_prompt, q_max)
        return self._pack_rubric_result(scores_1, scores_2, q_max, scores_1.get("feedback", ""))

    @staticmethod
    def _pack_rubric_result(
        scores_1: Dict[str, int],
        scores_2: Optional[Dict[str, int]],
        q_max: Dict[str, int],
        feedback: str,
    ) -> Dict[str, Any]:
        """把一次或兩次評分壓成最終 dict。"""
        q_raw_avg = {}
        q_norm = {}
        max_norm_diff = 0.0

        for q_key in RUBRIC_QUESTIONS:
            max_s = q_max[q_key]
            # 中間分 fallback：ceiling 除 2，避免 max_s=1 時給 0
            midpoint = (max_s + 1) // 2 if max_s else 0
            raw_1 = scores_1.get(q_key, midpoint)
            if scores_2 is not None:
                raw_2 = scores_2.get(q_key, midpoint)
                raw_avg = (raw_1 + raw_2) / 2
                n1 = raw_1 / max_s if max_s else 0
                n2 = raw_2 / max_s if max_s else 0
                max_norm_diff = max(max_norm_diff, abs(n1 - n2))
            else:
                raw_avg = raw_1
            q_raw_avg[q_key] = raw_avg
            q_norm[q_key] = raw_avg / max_s if max_s else 0

        overall = sum(q_norm.values()) / len(q_norm) if q_norm else 0
        return {
            "q_scores": q_raw_avg,
            "q_max": q_max,
            "q_norm": q_norm,
            "overall": overall,
            "feedback": feedback,
            "is_stable": (scores_2 is None) or (max_norm_diff < STABILITY_THRESHOLD_NORM),
            "max_norm_diff": max_norm_diff if scores_2 is not None else 0.0,
            # ── 與舊欄位相容（run_judge.py 舊呼叫可能讀這些）──
            "dimension_scores": q_raw_avg,
            "overall_score": overall * 5,  # 換回舊「1-5 比例」讓舊碼不爆
        }

    def batch_rubric_score(
        self,
        prompts: List[str],
        reports: List[str],
        crime_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """批次評分，用於消融實驗的各 config 比較。"""
        results = []
        for i, (prompt, report) in enumerate(zip(prompts, reports)):
            ct = crime_types[i] if crime_types else ""
            result = self.rubric_score(prompt, report, crime_type=ct)
            results.append(result)
        return results

    def compute_ablation_scores(
        self,
        config_reports: Dict[str, List[str]],
        prompts: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        消融實驗：對每個 config 計算平均 Rubric 分數。
        """
        summary = {}
        for config_name, reports in config_reports.items():
            scores = self.batch_rubric_score(prompts, reports)
            avg_by_dim = {}
            for dim in RUBRIC_DIMENSIONS:
                vals = [s["dimension_scores"].get(dim, 0) for s in scores]
                avg_by_dim[dim] = sum(vals) / len(vals) if vals else 0.0
            avg_by_dim["overall"] = sum(
                avg_by_dim[d] for d in RUBRIC_DIMENSIONS
            ) / len(RUBRIC_DIMENSIONS)
            avg_by_dim["stability_rate"] = (
                sum(1 for s in scores if s.get("is_stable", True)) / len(scores)
                if scores else 0.0
            )
            summary[config_name] = avg_by_dim

        return summary

    # ── 內部方法 ──────────────────────────────────────────

    def _call_judge(
        self,
        prompt: str,
        report_first: str,
        report_second: str,
        first_label: str,
        second_label: str,
    ) -> Dict[str, Any]:
        """呼叫 LLM 進行 Pairwise 比較。"""
        judge_prompt = self._build_pairwise_prompt(
            prompt, report_first, report_second, first_label, second_label
        )
        response_text = self._call_llm(judge_prompt, json_mode=True)
        parsed = _parse_json_response(response_text)

        if not parsed:
            return {
                "winner": "tie",
                "score_first": 0.0,
                "score_second": 0.0,
                "rationale": "JSON 解析失敗",
            }

        score_first = parsed.get("score_first", parsed.get(f"score_{first_label.lower()}", 0.0))
        score_second = parsed.get("score_second", parsed.get(f"score_{second_label.lower()}", 0.0))

        return {
            "winner": parsed.get("winner", "tie"),
            "score_first": float(score_first),
            "score_second": float(score_second),
            "rationale": parsed.get("rationale", ""),
        }

    def _call_rubric(self, judge_prompt: str, q_max: Dict[str, int]) -> Dict[str, Any]:
        """呼叫 LLM 進行 7 題 Rubric 評分；回傳 raw 分數 + feedback。"""
        response_text = self._call_llm(judge_prompt, json_mode=True)
        parsed = _parse_json_response(response_text)

        # midpoint fallback：ceiling 除 2 避免 max=1 時給 0
        def _midpoint(max_s: int) -> int:
            return (max_s + 1) // 2 if max_s else 0

        if not parsed:
            # fallback: 每題給中間分
            fallback = {q_key: _midpoint(q_max[q_key]) for q_key in RUBRIC_QUESTIONS}
            fallback["feedback"] = "[parse failed]"
            return fallback

        result = {}
        for q_key in RUBRIC_QUESTIONS:
            max_s = q_max[q_key]
            raw = parsed.get(q_key, _midpoint(max_s))
            try:
                raw_int = int(float(raw))
                raw_int = max(0, min(max_s, raw_int))
            except (ValueError, TypeError):
                raw_int = _midpoint(max_s)
            result[q_key] = raw_int

        result["feedback"] = parsed.get("feedback", "")
        return result

    def _call_llm(self, prompt: str, json_mode: bool = False) -> str:
        """呼叫 LLM API（支援 Claude / Gemini / OpenAI），含 token tracking。"""
        self._check_budget()
        client = self._get_client()

        if self._backend == "claude":
            try:
                kwargs = {
                    "model": self.judge_model,
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}],
                }
                response = client.messages.create(**kwargs)
                text = response.content[0].text if response.content else ""

                # Token tracking（Claude 回傳 usage）
                usage = getattr(response, "usage", None)
                if usage:
                    self._total_input_tokens += getattr(usage, "input_tokens", 0)
                    self._total_output_tokens += getattr(usage, "output_tokens", 0)
                self._total_calls += 1
                if self._total_calls % 10 == 0:
                    logger.info(f"[Budget] {self._total_calls} calls, ${self.total_cost_usd:.3f} / ${self._budget_limit_usd}")
                return text
            except Exception as e:
                logger.error(f"[LLMJudge] Claude API 呼叫失敗：{e}")
                return "{}"

        elif self._backend == "gemini":
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    ],
                )
                if json_mode:
                    config.response_mime_type = "application/json"

                response = client.models.generate_content(
                    model=self.judge_model,
                    contents=prompt,
                    config=config,
                )
                # Gemini token tracking（估算）
                text = response.text or ""
                self._total_input_tokens += len(prompt) // 3  # 粗估
                self._total_output_tokens += len(text) // 3
                self._total_calls += 1
                return text
            except Exception as e:
                logger.error(f"[LLMJudge] Gemini API 呼叫失敗：{e}")
                return "{}"

        else:  # OpenAI
            kwargs = {
                "model": self.judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                # OpenAI token tracking
                usage = getattr(response, "usage", None)
                if usage:
                    self._total_input_tokens += getattr(usage, "prompt_tokens", 0)
                    self._total_output_tokens += getattr(usage, "completion_tokens", 0)
                self._total_calls += 1
                return text
            except Exception as e:
                logger.error(f"[LLMJudge] OpenAI API 呼叫失敗：{e}")
                return "{}"

    def _build_pairwise_prompt(
        self,
        case_prompt: str,
        report_first: str,
        report_second: str,
        first_label: str,
        second_label: str,
    ) -> str:
        return f"""你是一位台灣刑事鑑識專家。請比較以下兩份鑑定報告的品質。

【案件描述】
{case_prompt}

【報告 {first_label}】
{report_first}

【報告 {second_label}】
{report_second}

【評分標準】
1. 邏輯一致性：推理鏈是否自洽
2. 法律覆蓋率：是否引用正確法條條號
3. 證據引用：是否精確標註影像幀
4. 因果推理：是否建立清晰因果鏈
5. 不確定性標記：是否誠實標記分析限制

請以 JSON 格式回傳：
{{"winner":"{first_label}"或"{second_label}"或"tie","score_first":1-5,"score_second":1-5,"rationale":"50字內說明理由"}}"""

    def _build_rubric_prompt(
        self,
        case_prompt: str,
        report: str,
        open_book_context: Optional[Dict[str, Any]] = None,
        q_max: Optional[Dict[str, int]] = None,
    ) -> str:
        """7 題 rubric prompt，含 open-book γ 參考資料。"""
        q_max = q_max or {
            k: (v["max_score"] or 4) for k, v in RUBRIC_QUESTIONS.items()
        }

        # Open-book 參考資料（情景 + 引用法條）
        reference_block = ""
        if open_book_context:
            lines = ["【參考資料】"]
            scenario = open_book_context.get("scenario_description", "")
            if scenario:
                lines.append(f"\nUCA 情景描述（事實 GT）：\n{scenario}")

            articles = open_book_context.get("cited_articles", [])
            if articles:
                lines.append("\n報告中引用的法條全文：")
                for art in articles:
                    lines.append(f"\n{art.get('article', '?')}")
                    if art.get("full_text"):
                        lines.append(f"  條文：{art['full_text']}")
                    if art.get("elements"):
                        lines.append(f"  構成要件（共 {len(art['elements'])} 項）：")
                        for i, e in enumerate(art["elements"], 1):
                            lines.append(f"    {i}. {e}")
            reference_block = "\n".join(lines) + "\n"

        # JSON schema 提示
        schema_items = []
        for q_key in RUBRIC_QUESTIONS:
            schema_items.append(f'"{q_key}":0-{q_max[q_key]}')
        schema_json = ",".join(schema_items) + ',"feedback":"50字內整體評語"'

        return f"""你是一位台灣刑事鑑識評分專家。請用 7 題 rubric 評估以下鑑定報告。

{reference_block}
【案件類別】
{case_prompt}

【待評估報告】
{report}

【評分規則（每題請給 raw 分數，整數）】

Q1 法條引用情景合理性（0–3）：
  3=條號+項完整，且對情景合理
  2=條號正確但格式不完整（如只寫「刑法 277 條」）
  1=條號本身有效但對情景不相關
  0=完全未引用法條

Q2 構成要件覆蓋率（0–{q_max['q2_element_coverage']}）：
  每覆蓋一個構成要件得 1 分
  要件清單：依【報告自己主張的罪名】對應的要件（見上方參考資料）
  最高 {q_max['q2_element_coverage']}（該罪名的要件總數）

Q3 影像證據可追溯性（0–3）：
  3=所有主要法律主張都有具體幀號／時間點對應
  2=多數主張有幀號（>50%）
  1=少數主張有幀號（<50%）
  0=完全無幀號或具體時間點

Q4 因果鏈完整性（0–4）：
  犯罪前階段（情緒前兆、準備行為）：+1 分
  犯罪行為本身：+1 分
  犯罪後階段（逃跑、被害人反應）：+1 分
  三階段都有且因果連結清楚：再 +1 分（共 4 分）

Q5 不確定性標記適當性（0–3）：
  3=明確說明視覺限制並建議人工複查，或影片品質確實良好不需標記（視情況）
  2=有提到不確定性但沒具體原因
  1=其他中間情況
  0=影片有遮蔽／過暗等明顯問題卻仍給出強烈肯定結論

Q6 司法語言規範性（0–3）：
  3=用語精確、主體統一（行為人／被害人）、動詞精確（施以／造成）
  2=整體專業但 1–2 處口語化
  1=多處口語但內容仍有實質意義
  0=用語混亂、術語使用明顯錯誤

Q7 情景事實吻合度（0–3，對照上方 UCA 情景 GT）：
  3=全部事實皆吻合情景
  2=絕大部分吻合，1–2 處小出入
  1=多處出入或過度推論
  0=明顯與情景矛盾

請以 JSON 格式回傳（整數分數，不要加 quote）：
{{{schema_json}}}"""
