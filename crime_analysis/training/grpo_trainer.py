"""
GRPO Trainer - Planner Agent 的強化學習訓練器
Group Relative Policy Optimization：
  對同一個影片生成 G 條偵查 Rollout，以相對獎勵優勢函數更新策略
"""
from typing import Any, Dict, List
import logging

from .reward_functions import RewardCalculator
from config import cfg

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    GRPO 訓練循環：

    for each batch of videos:
        for each video:
            1. 生成 G 條 Rollout（不同的代理人任務分配策略）
            2. 對每條 Rollout 執行完整辯論流程
            3. 計算每條 Rollout 的獎勵 R_i
            4. 計算組內相對優勢 A_i = (R_i - mean) / std
            5. 以 PPO-Clip 梯度更新 Planner 策略
            6. KL 散度懲罰，防止策略偏離 Reference Model 太遠
    """

    def __init__(self, planner, reward_calc: RewardCalculator = None):
        self.planner = planner
        self.reward_calc = reward_calc or RewardCalculator()
        self.cfg = cfg.grpo
        self._step = 0
        self._training_log: List[Dict] = []

    def train_step(
        self,
        frames: List,
        video_metadata: Dict,
        ground_truth: str,
    ) -> Dict[str, float]:
        """
        單步 GRPO 訓練：
        生成 G 條 Rollout，計算相對優勢，回傳 loss metrics
        """
        group_size = self.cfg.group_size

        # Step 1: 生成 G 條 Rollout
        rollout_results = []
        for g in range(group_size):
            # 每次 Rollout 用不同的 temperature/sampling 策略
            result = self.planner.run(frames, video_metadata)
            rollout_results.append(result)

        # Step 2: 計算每條 Rollout 的獎勵
        rewards = []
        for result in rollout_results:
            reward_info = self.reward_calc.compute(
                final_category=result["final_category"],
                ground_truth=ground_truth,
                rcons=result.get("rcons", 0.0),
                rlegal=result.get("rlegal", 0.0),
                total_turns=int(result.get("rcost", 1)),
            )
            rewards.append(reward_info["total"])

        # Step 3: 計算組內相對優勢
        advantages = self.reward_calc.compute_group_advantages(rewards)

        # Step 4: 計算 GRPO Loss（PPO-Clip）
        policy_loss = self._compute_policy_loss(rollout_results, advantages)

        # Step 5: KL 懲罰（防止策略偏離過遠）
        kl_loss = self._compute_kl_divergence()

        total_loss = policy_loss + self.cfg.kl_coef * kl_loss

        metrics = {
            "step": self._step,
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "best_category": rollout_results[rewards.index(max(rewards))]["final_category"],
            "ground_truth": ground_truth,
        }

        self._training_log.append(metrics)
        self._step += 1

        logger.info(
            f"Step {self._step}: "
            f"mean_reward={metrics['mean_reward']:.3f}, "
            f"best={metrics['best_category']} (GT={ground_truth})"
        )

        return metrics

    def _compute_policy_loss(
        self, rollouts: List[Dict], advantages: List[float]
    ) -> float:
        """
        PPO-Clip Loss：
        L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

        TODO: 實際實作需要：
        1. 從 Planner LLM 取得每個決策的 log_prob
        2. 從 Reference Model 取得 reference log_prob
        3. 計算 ratio = exp(log_prob - ref_log_prob)
        4. 套用 PPO-Clip 公式
        """
        # placeholder：模擬 loss 下降
        return max(0.0, 1.0 - sum(advantages) / len(advantages))

    def _compute_kl_divergence(self) -> float:
        """
        TODO: 計算當前策略與 Reference Model 的 KL 散度
        KL(π || π_ref) = E[log π(a|s) - log π_ref(a|s)]
        """
        return 0.01  # placeholder

    def get_training_log(self) -> List[Dict]:
        return self._training_log
