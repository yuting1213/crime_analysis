"""
主系統流程 - 組裝所有模組的入口點

架構（簡化後）：
    [影片輸入]
        ↓
    [Planner（規則式）]
        ├─ Step 1: Environment Agent（條件式）
        ├─ Step 2: ActionEmotion Agent
        └─ Step 3: RAG 查詢 → Qwen3 報告生成 → Rlegal
        ↓
    [Reflector CASAM] ← 衝突解決
        ↓
    [結構化鑑定報告] ←── DPO 對齊
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# 自動從 .env 載入 API keys；shell exports 仍優先
try:
    from env_loader import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import cfg
from agents import (
    EnvironmentAgent, ActionEmotionAgent,
    ReflectorAgent, PlannerAgent,
)
from rag import HierarchicalRAG, LawPreprocessor, JudgmentPreprocessor
from rag.rag_module import RAGModule
from training import RewardCalculator, GRPOTrainer, DPOTrainer
from evaluation import MetricsCalculator, LLMJudge

logger = logging.getLogger(__name__)


def _init_cuda_backends():
    """
    RTX 5090 Blackwell CUDA 後端初始化。
    在任何模型載入之前呼叫，確保全域設定生效。
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，將使用 CPU 模式")
        return

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024 ** 3)
    logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    # cuDNN benchmark: 自動調優卷積核（R3D-18 3D conv 大幅受益）
    if cfg.model.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # TF32: Blackwell 預設啟用，確保 matmul 精度
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info(
        f"CUDA 後端初始化完成 | cuDNN benchmark={cfg.model.cudnn_benchmark} | "
        f"TF32=True | dtype={cfg.model.torch_dtype}"
    )


class CrimeAnalysisPipeline:
    """
    系統主流程

    架構：
        [影片輸入]
            ↓
        [Planner（規則式三步驟）]
            ├─ Step 1: Environment Agent（條件式）
            ├─ Step 2: ActionEmotion Agent
            └─ Step 3: RAG + Qwen3 報告生成 + Rlegal
            ↓
        [Reflector CASAM] ← 衝突解決
            ↓
        [結構化鑑定報告] ←── DPO 對齊
    """

    def __init__(self):
        # RTX 5090: 初始化 CUDA 後端（cuDNN benchmark, TF32）
        _init_cuda_backends()

        # 建立 H-RAG 系統
        self.rag = HierarchicalRAG(cfg.rag)
        self.rag_module = RAGModule(self.rag)

        # 建立 Agent（新架構：Environment + ActionEmotion）
        agents_dict = {
            "environment": EnvironmentAgent(),
            "action_emotion": ActionEmotionAgent(),
        }

        # 建立 Reflector
        self.reflector = ReflectorAgent()

        # 建立 Planner（規則式，接收 agents_dict + rag_module）
        self.planner = PlannerAgent(
            agents_dict=agents_dict,
            reflector=self.reflector,
            rag_module=self.rag_module,
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
        對每個影片用不同溫度生成兩份報告，以 GPT-4o 比較後建立偏好對。

        生成策略（定義在 DPOConfig）：
        - pairs_per_category: 每個犯罪類別的目標偏好對數
        - generation_temperatures: 用不同溫度產生多樣報告
        - min_score_gap: Judge 分差太小的對會被丟棄
        - position_bias_check: AB/BA 雙向校正

        預估總量：13 類 × 15 對/類 = ~195 對（扣除 position bias 約剩 120~150 對）
        """
        temps = cfg.dpo.generation_temperatures
        count = 0
        category_counts: Dict[str, int] = {}

        for sample in video_samples:
            cat = sample.get("ground_truth", "Normal")
            target = cfg.dpo.pairs_per_category
            if category_counts.get(cat, 0) >= target:
                continue

            meta = sample.get("metadata", {})

            # 選兩個不同溫度生成報告
            temp_a = temps[count % len(temps)]
            temp_b = temps[(count + 1) % len(temps)]
            if temp_a == temp_b and len(temps) > 1:
                temp_b = temps[(count + 2) % len(temps)]

            meta_a = {**meta, "temperature": temp_a}
            meta_b = {**meta, "temperature": temp_b}

            result_a = self.planner.run(sample["frames"], meta_a)
            result_b = self.planner.run(sample["frames"], meta_b)

            report_a = result_a.get("fact_finding", {}).get("description", "")
            report_b = result_b.get("fact_finding", {}).get("description", "")

            if not report_a or not report_b:
                continue

            prompt = f"影片類型：{cat}，{sample.get('description', '')}"
            pair = self.dpo_trainer.collect_preference_pair(
                video_id=sample.get("video_id", str(count)),
                prompt=prompt,
                report_a=report_a,
                report_b=report_b,
            )
            if pair:
                # 檢查分差是否足夠
                gap = abs(pair.judge_score_chosen - pair.judge_score_rejected)
                if gap < cfg.dpo.min_score_gap:
                    self.dpo_trainer.get_dataset().pop()  # 移除品質太接近的對
                    logger.debug(f"DPO 分差不足 ({gap:.2f} < {cfg.dpo.min_score_gap})，跳過")
                    continue
                count += 1
                category_counts[cat] = category_counts.get(cat, 0) + 1

        self.dpo_trainer.export_to_jsonl(output_path)
        logger.info(
            f"DPO 資料集：{count} 筆偏好對 → {output_path}\n"
            f"  各類別分布：{category_counts}"
        )
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
    # 完整消融實驗請使用 scripts/run_ablation.py（支援 5 種變體）
    # pipeline 保留簡易版供快速驗證


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
    print(result.get("fact_finding", {}).get("description", "（無報告）"))
