"""
LLM-as-a-Judge 評估模組
兩種評估方式：
  1. Pairwise 比較 - 用於 DPO 偏好對建構（AB/BA 雙向比較校正 Position Bias）
  2. Rubric 評分 - 用於消融實驗量化（四個維度對應四個獎勵函數）

Judge Model：GPT-4o（避免 Self-Enhancement Bias）
Position Bias 控制：正反各跑一次取平均
穩定性標記：兩次分數差距 > 3 分時警告
"""
from typing import Any, Dict, List, Optional
import logging
import json

from config import cfg

logger = logging.getLogger(__name__)

# Rubric 評分維度（與 dpo_trainer.py 中的定義保持一致）
RUBRIC_DIMENSIONS = {
    "logical_consistency": "報告的推理鏈是否邏輯自洽，無矛盾（對應 Rcons）",
    "legal_coverage": "報告是否涵蓋足夠的法律構成要件（對應 Rlegal）",
    "evidence_citation": "是否精確標註對應的影像幀與時間點",
    "causal_reasoning": "是否建立清晰的因果鏈（Pearl 因果階梯）",
}

# 兩次評分差距超過此值時標記為不穩定
STABILITY_THRESHOLD = 3.0


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    從 LLM 回應中提取 JSON。

    處理三種常見格式：
    1. 純 JSON 字串
    2. ```json ... ``` 包裹
    3. 混合文字中的 JSON 區塊
    """
    text = text.strip()

    # 嘗試直接 parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 嘗試 ```json ... ```
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 嘗試找第一個 { 到最後一個 }
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

    Judge Model：Gemini 2.0 Flash（高速 + 低成本）或 GPT-4o
    理由：避免使用 Qwen（訓練模型）作為 Judge，防止 Self-Enhancement Bias
    Judge Prompt 角色：「台灣刑事鑑識專家」（確保以台灣法律邏輯評判）

    支援兩種 backend：
      - "gemini" 開頭 → 使用 google-generativeai
      - 其他 → 使用 OpenAI API
    """

    def __init__(self, judge_model: str = None, api_key: str = None):
        self.judge_model = judge_model or cfg.dpo.judge_model
        self._api_key = api_key
        self._client = None
        self._backend = "gemini" if "gemini" in self.judge_model.lower() else "openai"

    def _get_client(self):
        """延遲初始化 LLM client（支援 Gemini / OpenAI）。"""
        if self._client is not None:
            return self._client

        if self._backend == "gemini":
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

        Args:
            double_check: True 時做 AB/BA 雙向比較（2x token 成本）
        """
        # 截斷報告節省 token
        report_a = report_a[:800]
        report_b = report_b[:800]

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
        Prometheus 風格的 Rubric 評分。
        預設跑一次（省 token），可選 double_check=True 跑兩次取平均。

        Args:
            double_check: True 時跑兩次檢查穩定性（2x token 成本）
        """
        # 截斷報告節省 token
        report = report[:800]

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

        Returns:
            每個 config 在每個維度的平均分 + overall + stability_rate
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
        """呼叫 GPT-4o 進行 Pairwise 比較。"""
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

        # 標準化 score keys
        score_first = parsed.get("score_first", parsed.get(f"score_{first_label.lower()}", 0.0))
        score_second = parsed.get("score_second", parsed.get(f"score_{second_label.lower()}", 0.0))

        return {
            "winner": parsed.get("winner", "tie"),
            "score_first": float(score_first),
            "score_second": float(score_second),
            "rationale": parsed.get("rationale", ""),
        }

    def _call_rubric(self, judge_prompt: str) -> Dict[str, Any]:
        """呼叫 GPT-4o 進行 Rubric 評分，回傳維度分數。"""
        response_text = self._call_llm(judge_prompt, json_mode=True)
        parsed = _parse_json_response(response_text)

        if not parsed:
            return {dim: 3 for dim in RUBRIC_DIMENSIONS}

        # 確保每個維度都有合法的 1-5 分數
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
        """呼叫 LLM API（支援 Gemini / OpenAI）。"""
        client = self._get_client()

        if self._backend == "gemini":
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=256,
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
                return response.text or ""
            except Exception as e:
                logger.error(f"[LLMJudge] Gemini API 呼叫失敗：{e}")
                return "{}"
        else:
            kwargs = {
                "model": self.judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 256,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
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
        return f"""比較兩份台灣刑事鑑定報告，以JSON回傳。
案件：{case_prompt[:100]}
報告{first_label}：{report_first}
報告{second_label}：{report_second}
JSON格式：{{"winner":"{first_label}"/"{second_label}"/"tie","score_first":1-5,"score_second":1-5,"rationale":"30字內"}}"""

    def _build_rubric_prompt(self, case_prompt: str, report: str) -> str:
        return f"""評估台灣刑事鑑定報告，每項1-5分，以JSON回傳。
案件：{case_prompt[:100]}
報告：{report}
評分項：logical_consistency(邏輯)、legal_coverage(法律要件)、evidence_citation(證據引用)、causal_reasoning(因果推理)
JSON格式：{{"logical_consistency":int,"legal_coverage":int,"evidence_citation":int,"causal_reasoning":int,"feedback":"30字內"}}"""
