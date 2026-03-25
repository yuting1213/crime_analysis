"""
DPO Trainer - 最終鑑定報告品質對齊
Direct Preference Optimization：
  GRPO 負責 Planner 的任務調度決策（可量化獎勵）
  DPO 負責 Semantic Agent 的報告生成品質（偏好比較，難量化）

偏好對來源：
  - GPT-4o 作為 Pairwise Judge（避免 Self-Enhancement Bias）
  - AB/BA 雙向比較校正 Position Bias
  - Judge Prompt 明確指定「台灣刑事鑑識專家」角色
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """DPO 訓練用的偏好對"""
    video_id: str
    prompt: str          # 影像描述 + 犯罪類別
    chosen: str          # 較佳的鑑定報告
    rejected: str        # 較差的鑑定報告
    judge_score_chosen: float
    judge_score_rejected: float
    judgment_criteria: List[str]  # 評分維度


# Prometheus 風格的 Rubric 評分維度（對應四個獎勵函數）
RUBRIC_DIMENSIONS = {
    "logical_consistency": {
        "description": "報告的推理鏈是否邏輯自洽，無矛盾",
        "levels": {
            1: "完全無邏輯，各證據間明顯矛盾",
            2: "部分矛盾，推理不完整",
            3: "基本邏輯通順，但有小瑕疵",
            4: "邏輯清晰，有完整推理鏈",
            5: "完整論證，包含替代解釋的排除",
        },
    },
    "legal_coverage": {
        "description": "報告是否涵蓋足夠的法律構成要件",
        "levels": {
            1: "未提及任何法條",
            2: "提及法條但未說明關聯",
            3: "提及法條並說明基本關聯",
            4: "完整涵蓋主要構成要件",
            5: "完整涵蓋並引用裁判書判例支撐",
        },
    },
    "evidence_citation": {
        "description": "是否精確標註對應的影像幀與時間點",
        "levels": {
            1: "完全無幀引用",
            2: "有幀引用但不精確",
            3: "有精確幀引用",
            4: "有幀引用並說明視覺證據",
            5: "有幀引用、視覺證據說明，並排除誤判可能",
        },
    },
    "causal_reasoning": {
        "description": "是否建立清晰的因果鏈（Pearl 因果階梯）",
        "levels": {
            1: "只有相關性描述，無因果論述",
            2: "有簡單因果描述",
            3: "有完整的前因後果描述",
            4: "有因果鏈並識別前兆行為",
            5: "完整因果分析，包含反事實推論",
        },
    },
}


class DPOTrainer:
    """
    DPO 訓練器

    訓練流程：
    1. 收集報告對（同一影片，不同生成結果）
    2. 以 GPT-4o Judge 進行 Pairwise 比較
    3. 校正 Position Bias（AB/BA 雙向比較）
    4. 建立偏好對 (chosen, rejected)
    5. 以 DPO Loss 更新 Semantic Agent（LoRA）
    """

    def __init__(self, judge_model: str = None):
        self.judge_model = judge_model or cfg.dpo.judge_model
        self.beta = cfg.dpo.beta
        self._preference_dataset: List[PreferencePair] = []

    def collect_preference_pair(
        self,
        video_id: str,
        prompt: str,
        report_a: str,
        report_b: str,
    ) -> Optional[PreferencePair]:
        """
        收集一對偏好對：
        1. 正向比較：Judge 以 A 在前評分
        2. 反向比較：Judge 以 B 在前評分（校正 Position Bias）
        3. 若兩次結果一致，加入訓練集
        """
        # 正向比較（A 在前）
        score_ab = self._judge_pairwise(prompt, report_a, report_b, order="AB")

        # 反向比較（B 在前，校正 Position Bias）
        score_ba = self._judge_pairwise(prompt, report_b, report_a, order="BA")

        # 校正：若兩次結果一致才採用
        ab_prefers_a = score_ab["winner"] == "A"
        ba_prefers_a = score_ba["winner"] == "B"  # BA 順序下 B 位置對應原始 A

        if ab_prefers_a != ba_prefers_a:
            logger.debug(f"Video {video_id}: Position Bias 偵測，跳過此對")
            return None

        chosen = report_a if ab_prefers_a else report_b
        rejected = report_b if ab_prefers_a else report_a

        pair = PreferencePair(
            video_id=video_id,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            judge_score_chosen=score_ab.get("score_winner", 0.0),
            judge_score_rejected=score_ab.get("score_loser", 0.0),
            judgment_criteria=list(RUBRIC_DIMENSIONS.keys()),
        )
        self._preference_dataset.append(pair)
        return pair

    def compute_dpo_loss(
        self, policy_logprob_chosen: float, policy_logprob_rejected: float,
        ref_logprob_chosen: float, ref_logprob_rejected: float,
    ) -> float:
        """
        DPO Loss 公式：
        L_DPO = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                         - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))
        其中 y_w = chosen, y_l = rejected

        TODO: 在實際訓練中整合 trl 的 DPOTrainer
        from trl import DPOTrainer as TRLDPOTrainer
        """
        import math

        delta_chosen = policy_logprob_chosen - ref_logprob_chosen
        delta_rejected = policy_logprob_rejected - ref_logprob_rejected
        logit = self.beta * (delta_chosen - delta_rejected)

        # -log(sigmoid(logit))
        loss = -math.log(1 / (1 + math.exp(-logit)) + 1e-8)
        return loss

    def get_dataset(self) -> List[PreferencePair]:
        return self._preference_dataset

    def export_to_jsonl(self, output_path: str):
        """匯出偏好對為 JSONL 格式（供 trl DPOTrainer 使用）"""
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in self._preference_dataset:
                record = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"匯出 {len(self._preference_dataset)} 筆偏好對 → {output_path}")

    # ── 內部方法 ──────────────────────────────────────────

    def _judge_pairwise(
        self, prompt: str, report_first: str, report_second: str, order: str
    ) -> Dict[str, Any]:
        """
        呼叫 GPT-4o 進行 Pairwise 比較

        Judge Prompt 設計原則：
        - 明確指定「台灣刑事鑑識專家」角色（避免非台灣法律邏輯評判）
        - 使用 Rubric 四個維度逐一評分
        - 要求給出 winner 和分項分數

        TODO:
        from openai import OpenAI
        client = OpenAI()
        judge_prompt = self._build_judge_prompt(
            prompt, report_first, report_second
        )
        response = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
        """
        # placeholder：模擬 Judge 回傳
        return {
            "winner": "A",
            "score_winner": 4.2,
            "score_loser": 3.1,
            "rationale": "Report A provides clearer legal element coverage...",
        }

    def _build_judge_prompt(
        self, prompt: str, report_a: str, report_b: str
    ) -> str:
        """建立 Judge Prompt，指定台灣刑事鑑識專家角色"""
        rubric_text = "\n".join(
            f"- {dim}: {info['description']}"
            for dim, info in RUBRIC_DIMENSIONS.items()
        )
        return f"""你是一位台灣刑事鑑識專家，請依照以下評分標準，比較兩份鑑定報告的品質。

【案件描述】
{prompt}

【報告 A】
{report_a}

【報告 B】
{report_b}

【評分標準】
{rubric_text}

請以 JSON 格式回傳：
{{
  "winner": "A" 或 "B",
  "score_A": 1-5 的平均分,
  "score_B": 1-5 的平均分,
  "dimension_scores": {{維度名稱: {{"A": 分數, "B": 分數}}}},
  "rationale": "選擇理由"
}}"""
