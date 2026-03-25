"""
LLM-as-a-Judge 評估模組
兩種評估方式：
  1. Pairwise 比較 - 用於 DPO 偏好對建構（AB/BA 雙向比較校正 Position Bias）
  2. Rubric 評分 - 用於消融實驗量化（四個維度對應四個獎勵函數）
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


class LLMJudge:
    """
    LLM-as-a-Judge

    Judge Model 選擇：GPT-4o
    理由：避免使用 Qwen3（訓練模型）作為 Judge，防止 Self-Enhancement Bias
    Judge Prompt 角色：「台灣刑事鑑識專家」（確保以台灣法律邏輯評判）
    """

    def __init__(self, judge_model: str = None):
        self.judge_model = judge_model or cfg.dpo.judge_model
        # TODO: self.client = OpenAI()

    # ── Pairwise 比較 ─────────────────────────────────────

    def pairwise_compare(
        self,
        prompt: str,
        report_a: str,
        report_b: str,
        video_id: str = "",
    ) -> Dict[str, Any]:
        """
        AB/BA 雙向比較，校正 Position Bias

        Returns:
            {
                "winner": "A" | "B" | "tie",
                "score_a": float,
                "score_b": float,
                "is_consistent": bool,   # AB 和 BA 結果是否一致
                "rationale": str,
            }
        """
        # 正向：A 在前
        result_ab = self._call_judge(
            prompt, report_a, report_b, first_label="A", second_label="B"
        )
        # 反向：B 在前（校正 Position Bias）
        result_ba = self._call_judge(
            prompt, report_b, report_a, first_label="B", second_label="A"
        )

        winner_ab = result_ab.get("winner", "tie")
        winner_ba = result_ba.get("winner", "tie")

        # 一致性檢查
        is_consistent = winner_ab == winner_ba
        if not is_consistent:
            logger.debug(f"Video {video_id}: Position Bias detected, skipping")

        return {
            "winner": winner_ab if is_consistent else "tie",
            "score_a": result_ab.get("score_first", 0.0),
            "score_b": result_ab.get("score_second", 0.0),
            "is_consistent": is_consistent,
            "rationale": result_ab.get("rationale", ""),
            "video_id": video_id,
        }

    # ── Rubric 評分 ───────────────────────────────────────

    def rubric_score(
        self, prompt: str, report: str
    ) -> Dict[str, Any]:
        """
        Prometheus 風格的 Rubric 評分
        對每個維度給出 1-5 分，用於消融實驗量化比較

        Returns:
            {
                "dimension_scores": {"logical_consistency": 4, ...},
                "overall_score": float,
                "feedback": str,
            }
        """
        # TODO: 呼叫 GPT-4o 進行 Rubric 評分
        judge_prompt = self._build_rubric_prompt(prompt, report)
        response = self._call_llm(judge_prompt)

        # placeholder 回傳
        dimension_scores = {dim: 3 for dim in RUBRIC_DIMENSIONS}
        overall = sum(dimension_scores.values()) / len(dimension_scores)

        return {
            "dimension_scores": dimension_scores,
            "overall_score": overall,
            "feedback": "placeholder feedback",
        }

    def batch_rubric_score(
        self,
        prompts: List[str],
        reports: List[str],
    ) -> List[Dict[str, Any]]:
        """批次評分，用於消融實驗的各 config 比較"""
        results = []
        for prompt, report in zip(prompts, reports):
            result = self.rubric_score(prompt, report)
            results.append(result)
        return results

    def compute_ablation_scores(
        self,
        config_reports: Dict[str, List[str]],
        prompts: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        消融實驗：對每個 config 計算平均 Rubric 分數

        Args:
            config_reports: {"full_system": [...], "no_action": [...], ...}
            prompts: 對應的影像描述

        Returns:
            每個 config 在每個維度的平均分
        """
        summary = {}
        for config_name, reports in config_reports.items():
            scores = self.batch_rubric_score(prompts, reports)
            avg_by_dim = {}
            for dim in RUBRIC_DIMENSIONS:
                vals = [s["dimension_scores"].get(dim, 0) for s in scores]
                avg_by_dim[dim] = sum(vals) / len(vals) if vals else 0.0
            avg_by_dim["overall"] = sum(avg_by_dim.values()) / len(avg_by_dim)
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
        """
        TODO:
        judge_prompt = self._build_pairwise_prompt(
            prompt, report_first, report_second, first_label, second_label
        )
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
        """
        return {
            "winner": first_label,
            "score_first": 4.0,
            "score_second": 3.0,
            "rationale": "placeholder",
        }

    def _call_llm(self, prompt: str) -> str:
        """TODO: 呼叫 LLM API"""
        return "{}"  # placeholder

    def _build_pairwise_prompt(
        self,
        case_prompt: str,
        report_first: str,
        report_second: str,
        first_label: str,
        second_label: str,
    ) -> str:
        return f"""你是一位台灣刑事鑑識專家，請依照台灣刑事訴訟法與刑法的標準，
比較以下兩份鑑定報告的品質。

【案件描述】
{case_prompt}

【報告 {first_label}】
{report_first}

【報告 {second_label}】
{report_second}

請從以下四個維度評估，並以 JSON 回傳：
1. logical_consistency（邏輯一致性）
2. legal_coverage（法律要件覆蓋率）
3. evidence_citation（影像幀引用精確度）
4. causal_reasoning（因果推理完整性）

回傳格式：
{{
  "winner": "{first_label}" 或 "{second_label}" 或 "tie",
  "score_{first_label.lower()}": 1-5 分,
  "score_{second_label.lower()}": 1-5 分,
  "dimension_comparison": {{"維度": "{first_label} 優/劣"}},
  "rationale": "選擇理由"
}}"""

    def _build_rubric_prompt(self, case_prompt: str, report: str) -> str:
        rubric_text = "\n".join(
            f"- **{dim}**（1-5分）：{desc}"
            for dim, desc in RUBRIC_DIMENSIONS.items()
        )
        return f"""你是一位台灣刑事鑑識專家，請依照以下 Rubric 評估這份鑑定報告。

【案件描述】
{case_prompt}

【鑑定報告】
{report}

【評分標準】
{rubric_text}

請以 JSON 回傳每個維度的分數（1-5整數）與整體回饋：
{{
  "logical_consistency": 整數,
  "legal_coverage": 整數,
  "evidence_citation": 整數,
  "causal_reasoning": 整數,
  "feedback": "具體改進建議"
}}"""
