"""
全域設定檔 - 基於強化學習之層級式多代理人犯罪影像分析架構
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct"    # 統一 VLM：分類 + 報告生成
    report_model: str = "Qwen/Qwen3-VL-8B-Instruct"  # 與 base_model 相同（向後相容）
    planner_model: str = "gpt-4o"                      # Planner 使用 GPT-4o
    embedding_model: str = "BAAI/bge-m3"               # BGE-M3 中文向量
    device: str = "cuda"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    # ── RTX 5090 優化參數 ──
    torch_dtype: str = "bfloat16"          # Blackwell 原生 BF16，吞吐量遠高於 FP32
    use_flash_attention: bool = True       # Flash Attention 2（Qwen 推理加速）
    compile_models: bool = True            # torch.compile（Blackwell inductor 後端）
    cudnn_benchmark: bool = True           # 自動調優卷積核（R3D-18 / ViT 受益）


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
    judge_model: str = "gemini-2.0-flash"  # Pairwise judge（Gemini 高速低成本）
    # ── 偏好對生成策略 ──
    pairs_per_category: int = 15  # 每個犯罪類別的偏好對數量
    generation_temperatures: List[float] = field(
        default_factory=lambda: [0.3, 0.7, 1.0]
    )  # 不同溫度產生報告多樣性，每對從中取兩個溫度組合
    min_score_gap: float = 0.5    # Judge 分差 < 此值的對被丟棄（品質太接近無法區分）
    position_bias_check: bool = True  # AB/BA 雙向比較校正


@dataclass
class TrainingConfig:
    """MIL Head 訓練配置（針對 RTX 5090 優化）"""
    batch_size: int = 32   # 降低以增加 gradient steps（576 anomaly / 32 = 18 batch/epoch）
    epochs: int = 60       # 18 × 60 = 1080 steps，匹配 13.3M 參數量
    learning_rate: float = 5e-4    # 配合較小 batch_size 稍微提高
    weight_decay: float = 1e-2     # 強正則化防 overfitting（13.3M params, ~1K samples）
    gradient_clip_norm: float = 1.0
    lambda_mil: float = 0.3        # 主任務是分類（CE），MIL ranking 為輔助
    label_smoothing: float = 0.1   # 防止分類頭過度自信
    warmup_ratio: float = 0.05     # 前 5% steps warmup（~54 steps）
    dropout: float = 0.2           # FusionEncoder dropout 加大
    # MIL 排序損失常數
    mil_mu1: float = 8e-5   # Smoothness 懲罰
    mil_mu2: float = 8e-5   # Sparsity 懲罰
    # ── RTX 5090 訓練優化 ──
    num_workers: int = 4             # DataLoader 並行載入
    pin_memory: bool = True          # 固定記憶體加速 Host→Device 傳輸
    mixed_precision: bool = True     # AMP 混合精度訓練（BF16）


@dataclass
class InferenceConfig:
    """推理配置"""
    # 特徵提取
    num_r3d_snippets: int = 32
    frames_per_clip: int = 16
    num_vit_keyframes: int = 8
    # 信心度閾值
    confidence_low_threshold: float = 0.4
    confidence_mid_threshold: float = 0.6
    video_quality_threshold: float = 0.6
    # 升溫分析
    escalation_calm_threshold: float = 0.35
    escalation_high_threshold: float = 0.7
    # 報告生成
    max_report_prompt_ratio: float = 0.75  # 最大 prompt token 佔引擎容量的比例
    # 正規化常數（ImageNet）
    imagenet_mean: tuple = (0.43216, 0.394666, 0.37645)
    imagenet_std: tuple = (0.22803, 0.22145, 0.216989)


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
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "./outputs"
    log_level: str = "INFO"


# 全域設定實例
cfg = Config()
