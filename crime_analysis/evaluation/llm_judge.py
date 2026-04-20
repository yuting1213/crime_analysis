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

# Rubric 評分維度（5 維度，每項 1-5 分，總分 25）
RUBRIC_DIMENSIONS = {
    "logical_consistency": "報告的推理鏈是否邏輯自洽，無矛盾（對應 Rcons）",
    "legal_coverage": "報告是否涵蓋足夠的法律構成要件並引用具體法條條號（對應 Rlegal）",
    "evidence_citation": "是否精確標註對應的影像幀與時間點，區分觀察事實與推論",
    "causal_reasoning": "是否建立清晰的因果鏈（前兆行為 → 犯罪實施 → 事後反應）",
    "uncertainty_marking": "是否明確標記不確定性、分析限制、和需進一步調查的事項",
}

# 兩次評分差距超過此值時標記為不穩定
STABILITY_THRESHOLD = 3.0


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

    # ── Rubric 評分（單份報告）────────────────────────────

    def rubric_score(
        self,
        prompt: str,
        report: str,
        crime_type: str = "",
        double_check: bool = False,
    ) -> Dict[str, Any]:
        """
        Prometheus 風格的 Rubric 評分（5 維度，每項 1-5 分）。
        """
        judge_prompt = self._build_rubric_prompt(prompt, report)
        scores_1 = self._call_rubric(judge_prompt)

        if not double_check:
            dim_scores = {dim: float(scores_1.get(dim, 3)) for dim in RUBRIC_DIMENSIONS}
            overall = sum(dim_scores.values()) / len(dim_scores)
            return {
                "dimension_scores": dim_scores,
                "overall_score": overall,
                "feedback": scores_1.get("feedback", ""),
                "is_stable": True,
            }

        # 穩定性檢查（2x token）
        scores_2 = self._call_rubric(judge_prompt)
        avg_scores = {}
        for dim in RUBRIC_DIMENSIONS:
            avg_scores[dim] = (scores_1.get(dim, 3) + scores_2.get(dim, 3)) / 2

        overall = sum(avg_scores.values()) / len(avg_scores)
        max_diff = max(
            abs(scores_1.get(dim, 3) - scores_2.get(dim, 3))
            for dim in RUBRIC_DIMENSIONS
        )
        is_stable = max_diff < STABILITY_THRESHOLD

        return {
            "dimension_scores": avg_scores,
            "overall_score": overall,
            "feedback": scores_1.get("feedback", ""),
            "is_stable": is_stable,
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

    def _call_rubric(self, judge_prompt: str) -> Dict[str, Any]:
        """呼叫 LLM 進行 Rubric 評分，回傳維度分數。"""
        response_text = self._call_llm(judge_prompt, json_mode=True)
        parsed = _parse_json_response(response_text)

        if not parsed:
            return {dim: 3 for dim in RUBRIC_DIMENSIONS}

        result = {}
        for dim in RUBRIC_DIMENSIONS:
            score = parsed.get(dim, 3)
            try:
                score = int(float(score))
                score = max(1, min(5, score))
            except (ValueError, TypeError):
                score = 3
            result[dim] = score

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

    def _build_rubric_prompt(self, case_prompt: str, report: str) -> str:
        dims_json = ", ".join(f'"{d}":int' for d in RUBRIC_DIMENSIONS)
        return f"""你是一位台灣刑事鑑識專家。請評估以下鑑定報告的品質，每項 1-5 分。

【案件描述】
{case_prompt}

【鑑定報告】
{report}

【評分標準（每項 1-5 分）】
1. logical_consistency（邏輯一致性）：推理鏈是否自洽，無矛盾
   1=完全矛盾 2=部分矛盾 3=基本通順 4=邏輯清晰 5=完整論證含替代解釋排除
2. legal_coverage（法律覆蓋率）：是否涵蓋足夠法律構成要件並引用具體法條條號
   1=未提及法條 2=提及但未說明 3=基本關聯 4=完整涵蓋要件 5=引用裁判書判例
3. evidence_citation（證據引用）：是否精確標註影像幀與時間點
   1=無引用 2=有但不精確 3=精確引用 4=引用+視覺證據說明 5=引用+排除誤判
4. causal_reasoning（因果推理）：是否建立清晰因果鏈
   1=僅相關性 2=簡單因果 3=完整前因後果 4=含前兆行為 5=含反事實推論
5. uncertainty_marking（不確定性標記）：是否誠實標記限制和不確定性
   1=未標記 2=泛泛提及 3=列出具體限制 4=限制+影響評估 5=限制+替代解釋+後續建議

請以 JSON 格式回傳：
{{{dims_json},"feedback":"50字內整體評語"}}"""
