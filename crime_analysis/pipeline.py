"""
主系統流程 - 組裝所有模組的入口點
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import cfg
from agents import (
    EnvironmentAgent, ActionAgent,
    TimeEmotionAgent, SemanticAgent,
    ReflectorAgent, PlannerAgent,
)
from rag import HierarchicalRAG, LawPreprocessor, JudgmentPreprocessor
from training import RewardCalculator, GRPOTrainer, DPOTrainer
from evaluation import MetricsCalculator, LLMJudge

logger = logging.getLogger(__name__)


class CrimeAnalysisPipeline:
    """
    系統主流程

    架構：
        [影片輸入]
            ↓
        [Planner] ←── GRPO 訓練
            ↓ 分配任務
        [Environment] [Action] [Time&Emotion] [Semantic←H-RAG]
            ↓ 各自回報
        [Reflector] ←── CASAM
            ↓ 衝突回饋
        [Planner 最終裁決]
            ↓
        [結構化鑑定報告] ←── DPO 對齊
    """

    def __init__(self):
        # 建立 H-RAG 系統
        self.rag = HierarchicalRAG(cfg.rag)

        # 建立四個 Local Solver
        self.local_agents = [
            EnvironmentAgent(),
            ActionAgent(),
            TimeEmotionAgent(),
            SemanticAgent(rag_system=self.rag),
        ]

        # 建立 Reflector
        self.reflector = ReflectorAgent()

        # 建立 Planner
        self.planner = PlannerAgent(
            local_agents=self.local_agents,
            reflector=self.reflector,
            reward_weights=cfg.reward,
        )

        # 訓練器
        self.reward_calc = RewardCalculator(cfg.reward)
        self.grpo_trainer = GRPOTrainer(self.planner, self.reward_calc)
        self.dpo_trainer = DPOTrainer(cfg.dpo.judge_model)

        # 評估器
        self.metrics = MetricsCalculator()
        self.judge = LLMJudge(cfg.dpo.judge_model)

    # ── 推論模式 ──────────────────────────────────────────

    def analyze(
        self, frames: List, video_metadata: Dict
    ) -> Dict[str, Any]:
        """
        單影片推論（不含訓練）
        Returns: 最終鑑定報告 + 各代理人分析結果
        """
        return self.planner.run(frames, video_metadata)

    # ── 訓練模式 ──────────────────────────────────────────

    def train_grpo(
        self,
        dataset: List[Dict],   # [{"frames": ..., "metadata": ..., "ground_truth": ...}]
        max_steps: int = None,
    ) -> List[Dict]:
        """
        GRPO 訓練 Planner Agent
        """
        steps = max_steps or cfg.grpo.max_steps
        training_log = []

        for step, sample in enumerate(dataset):
            if step >= steps:
                break

            metrics = self.grpo_trainer.train_step(
                frames=sample["frames"],
                video_metadata=sample.get("metadata", {}),
                ground_truth=sample["ground_truth"],
            )
            training_log.append(metrics)

            if step % 100 == 0:
                logger.info(f"GRPO Step {step}: {metrics}")

        return training_log

    def build_dpo_dataset(
        self,
        video_samples: List[Dict],
        output_path: str = "./data/dpo_pairs.jsonl",
    ) -> int:
        """
        建立 DPO 訓練資料集：
        對每個影片生成兩份報告，以 GPT-4o 比較後建立偏好對
        """
        count = 0
        for sample in video_samples:
            # 生成兩份報告（使用不同溫度/Prompt）
            result_a = self.planner.run(sample["frames"], sample.get("metadata", {}))
            result_b = self.planner.run(sample["frames"], sample.get("metadata", {}))

            report_a = result_a.get("forensic_report", "")
            report_b = result_b.get("forensic_report", "")

            if not report_a or not report_b:
                continue

            prompt = f"影片類型：{sample.get('ground_truth', '未知')}，{sample.get('description', '')}"
            pair = self.dpo_trainer.collect_preference_pair(
                video_id=sample.get("video_id", str(count)),
                prompt=prompt,
                report_a=report_a,
                report_b=report_b,
            )
            if pair:
                count += 1

        self.dpo_trainer.export_to_jsonl(output_path)
        logger.info(f"DPO 資料集：{count} 筆偏好對 → {output_path}")
        return count

    # ── RAG 初始化 ────────────────────────────────────────

    def build_rag_index(self):
        """
        從原始資料建立 H-RAG 索引
        須在首次執行前呼叫
        """
        law_preprocessor = LawPreprocessor()
        judgment_preprocessor = JudgmentPreprocessor()

        law_dir = Path(cfg.rag.law_data_dir)
        judgment_dir = Path(cfg.rag.judgment_data_dir)

        law_chunks = []
        for f in law_dir.glob("*.json"):
            law_chunks.extend(law_preprocessor.process_file(str(f)))

        judgment_chunks = []
        for f in judgment_dir.glob("*.json"):
            judgment_chunks.extend(judgment_preprocessor.process_file(str(f)))

        self.rag.build_index(
            [c.to_dict() for c in law_chunks],
            [c.to_dict() for c in judgment_chunks],
        )
        logger.info(
            f"RAG 索引建立完成：{len(law_chunks)} 法條，{len(judgment_chunks)} 裁判書段落"
        )

    # ── 消融實驗 ──────────────────────────────────────────

    def run_ablation(
        self,
        test_samples: List[Dict],
    ) -> Dict[str, Dict]:
        """
        Leave-One-Out 消融實驗
        逐一移除單一 Agent，量化各模組貢獻
        """
        configs = {
            "full_system": list(range(len(self.local_agents))),
            "no_environment": [1, 2, 3],   # 移除 EnvironmentAgent
            "no_action": [0, 2, 3],        # 移除 ActionAgent
            "no_time_emotion": [0, 1, 3],  # 移除 TimeEmotionAgent
            "no_semantic": [0, 1, 2],      # 移除 SemanticAgent
        }

        ablation_results: Dict[str, List[Dict]] = {}

        for config_name, agent_indices in configs.items():
            logger.info(f"消融實驗：{config_name}")
            active_agents = [self.local_agents[i] for i in agent_indices]

            temp_planner = PlannerAgent(
                local_agents=active_agents,
                reflector=self.reflector,
            )

            config_preds = []
            for sample in test_samples:
                result = temp_planner.run(
                    sample["frames"], sample.get("metadata", {})
                )
                config_preds.append(result)

            ablation_results[config_name] = config_preds

        # 計算各 config 的分類指標
        ground_truths = [s["ground_truth"] for s in test_samples]
        metrics_table = self.metrics.compute_ablation_table(
            ablation_results, ground_truths
        )

        return metrics_table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = CrimeAnalysisPipeline()

    # 建立 RAG 索引（首次執行）
    # pipeline.build_rag_index()

    # 測試單影片推論（以空白 placeholder 幀測試架構）
    dummy_frames = [None] * 32
    dummy_metadata = {"fps": 25.0, "duration": 1.28, "video_id": "test_001"}

    result = pipeline.analyze(dummy_frames, dummy_metadata)

    print("=== 測試結果 ===")
    print(f"最終犯罪類別：{result['final_category']}")
    print(f"Rcons：{result['rcons']:.3f}")
    print(f"Rlegal：{result['rlegal']:.3f}")
    print(f"是否收斂：{result['is_convergent']}")
    print("\n=== 鑑定報告 ===")
    print(result.get("forensic_report", "（無報告）"))
