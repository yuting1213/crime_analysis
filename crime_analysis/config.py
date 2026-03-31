"""
全域設定檔 - 基於強化學習之層級式多代理人犯罪影像分析架構
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # 本地端 VLM（Action/Environment）
    report_model: str = "Qwen/Qwen2.5-7B-Instruct"     # Step 3b 報告生成（text-only）
    planner_model: str = "gpt-4o"                      # Planner 使用 GPT-4o
    embedding_model: str = "BAAI/bge-m3"               # BGE-M3 中文向量
    device: str = "cuda"
    max_new_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class RAGConfig:
    chroma_persist_dir: str = "./rag_db/chroma"
    law_data_dir: str = "./data/rag/laws"           # 台灣刑事法條
    judgment_data_dir: str = "./data/rag/judgments" # 最高法院裁判書
    manual_data_dir: str = "./data/rag/manuals"     # 司法鑑定操作手冊
    top_k_bm25: int = 5
    top_k_dense: int = 5
    top_k_final: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class RewardWeights:
    """獎勵函數權重: R_i = w1*Racc + w2*Rcons + w3*Rlegal - w4*Rcost"""
    w1: float = 0.4   # Racc  - 犯罪分類準確率
    w2: float = 0.3   # Rcons - 邏輯一致性（來自 Reflector）
    w3: float = 0.2   # Rlegal - 法律要件覆蓋率
    w4: float = 0.1   # Rcost  - log(對話輪次)


@dataclass
class GRPOConfig:
    group_size: int = 8           # 每個問題生成幾條 rollout 互相比較
    learning_rate: float = 1e-5
    max_steps: int = 5000
    clip_epsilon: float = 0.2
    kl_coef: float = 0.04


@dataclass
class DPOConfig:
    """用於最終報告品質對齊"""
    learning_rate: float = 5e-6
    beta: float = 0.1             # KL 懲罰係數
    max_steps: int = 2000
    judge_model: str = "gpt-4o"  # Pairwise judge，避免 Self-Enhancement Bias
    # ── 偏好對生成策略 ──
    pairs_per_category: int = 15  # 每個犯罪類別的偏好對數量
    generation_temperatures: List[float] = field(
        default_factory=lambda: [0.3, 0.7, 1.0]
    )  # 不同溫度產生報告多樣性，每對從中取兩個溫度組合
    min_score_gap: float = 0.5    # Judge 分差 < 此值的對被丟棄（品質太接近無法區分）
    position_bias_check: bool = True  # AB/BA 雙向比較校正


@dataclass
class DebateConfig:
    max_rounds: int = 3           # 最大辯論輪次
    conflict_threshold: float = 0.5   # Reflector 判定衝突的門檻
    consensus_threshold: float = 0.8  # Reflector 判定收斂的門檻


@dataclass
class DataConfig:
    ucf_crime_dir: str = "./data/ucf_crime"
    # UCF-Crime 13 類
    crime_categories: List[str] = field(default_factory=lambda: [
        "Normal", "Robbery", "RoadAccidents", "Stealing", "Burglary",
        "Abuse", "Assault", "Vandalism", "Arrest", "Fighting",
        "Arson", "Explosion", "Shoplifting", "Shooting"
    ])
    frames_per_clip: int = 32     # 每個影片片段抽幾幀
    frame_size: tuple = (224, 224)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "./outputs"
    log_level: str = "INFO"


# 全域設定實例
cfg = Config()
