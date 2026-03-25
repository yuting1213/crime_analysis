"""
獎勵函數模組
R_i = w1*Racc + w2*Rcons + w3*Rlegal - w4*Rcost

GRPO 訓練時：對同一個影片生成 G 條 Rollout，
計算每條 Rollout 的獎勵，再以相對優勢函數更新 Planner 策略。
"""
import math
from typing import Any, Dict, List
from config import cfg


class RewardCalculator:
    """
    四項獎勵函數的統一計算器
    """

    def __init__(self, weights=None):
        self.w = weights or cfg.reward

    def compute(
        self,
        final_category: str,
        ground_truth: str,
        rcons: float,
        rlegal: float,
        total_turns: int,
    ) -> Dict[str, float]:
        """
        計算各項獎勵分數並彙整

        Returns:
            {
                "racc": float,
                "rcons": float,
                "rlegal": float,
                "rcost": float,
                "total": float,
            }
        """
        racc = self._compute_racc(final_category, ground_truth)
        rcost = self._compute_rcost(total_turns)

        total = (
            self.w.w1 * racc
            + self.w.w2 * rcons
            + self.w.w3 * rlegal
            - self.w.w4 * rcost
        )

        return {
            "racc": racc,
            "rcons": rcons,
            "rlegal": rlegal,
            "rcost": rcost,
            "total": total,
        }

    def compute_group_advantages(
        self, rewards: List[float]
    ) -> List[float]:
        """
        GRPO 核心：計算組內相對優勢
        advantage_i = (reward_i - mean) / (std + ε)

        這讓 Planner 學習「哪條決策路徑比平均更好」，
        而非只學習「固定答案」。
        """
        if len(rewards) == 0:
            return []

        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(variance) + 1e-8

        return [(r - mean) / std for r in rewards]

    # ── 各項指標計算 ──────────────────────────────────────

    def _compute_racc(self, predicted: str, ground_truth: str) -> float:
        """
        分類準確率獎勵
        - 完全匹配：1.0
        - 同大類（財產犯罪 vs 財產犯罪）：0.5
        - 不同大類：0.0

        犯罪大類分組（根據 UCF-Crime）
        """
        if predicted == ground_truth:
            return 1.0

        # 大類分組
        category_groups = {
            "property": {"Robbery", "Stealing", "Burglary", "Shoplifting"},
            "violence": {"Assault", "Fighting", "Abuse", "Shooting"},
            "public_safety": {"Arson", "Explosion", "RoadAccidents", "Vandalism"},
            "law_enforcement": {"Arrest"},
            "normal": {"Normal"},
        }

        pred_group = self._get_group(predicted, category_groups)
        gt_group = self._get_group(ground_truth, category_groups)

        return 0.5 if pred_group == gt_group and pred_group != "normal" else 0.0

    def _compute_rcost(self, total_turns: int) -> float:
        """
        溝通成本懲罰：Rcost = log(Total Turns)
        輪次越多，懲罰越大（但以對數壓縮，避免懲罰過大）
        """
        return math.log(max(1, total_turns))

    def _get_group(self, category: str, groups: Dict) -> str:
        for group, cats in groups.items():
            if category in cats:
                return group
        return "unknown"
