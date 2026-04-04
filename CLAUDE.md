# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent LLM system for forensic crime video analysis. Combines vision models (R3D-18, ViT), Qwen2.5-VL-7B-Instruct (local LLM), Qwen3-8B (report generation), Gemini 2.0 Flash (judge), hierarchical RAG over Taiwan Criminal Code (刑法), and DPO alignment. Targets the UCF-Crime dataset with UCA temporal annotations. Scope is limited to Taiwan criminal law.

## Setup & Commands

```bash
# Environment setup (one-time) — RTX 5090 requires CUDA 12.8+
conda create -n crime python=3.11
conda activate crime
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pip install flash-attn --no-build-isolation   # FA2（flash-attn 尚不支援 sm_120 Blackwell，自動 fallback）
pip install transformers accelerate Pillow scikit-learn opencv-python
pip install rank-bm25 jieba chromadb sentence-transformers
pip install mediapipe==0.10.14 deepface tf-keras

# IMPORTANT: conda activate sets SSL_CERT_FILE to a nonexistent path.
# Always run `unset SSL_CERT_FILE` before any HuggingFace download.
unset SSL_CERT_FILE

# Build RAG index (one-time, after law data is ready)
cd crime_analysis
python data/scripts/build_rag.py

# Train MIL Head (must run before meaningful inference)
# Params from config.py: bs=32, ep=60, lr=5e-4, λ_MIL=0.3, wd=1e-2
python -m scripts.train_mil --split Train

# Run analysis on a video
python pipeline.py

# Pilot experiment (small-sample calibration)
python -m scripts.pilot_experiment --n_samples 30 --split Test

# Ablation study (5 variants)
python -m scripts.run_ablation --n_samples 100 --split Test

# Type checking
pyright crime_analysis/
```

No formal test suite — testing is done via `pipeline.analyze(frames, metadata)` directly.

## Architecture

### Agent Workflow (3-Step, Rule-Based Planner)

```
Video Frames
  │
  ├─[Step 1] Environment Agent   — quality check, occlusion detection (conditional)
  ├─[Step 2] ActionEmotion Agent — R3D-18 + ViT feature fusion → crime_head (13 classes) + escalation_head
  └─[Step 3] Planner integrates:
       3a. H-RAG query (BM25 + BGE-M3 + RRF)
       3b. Qwen3-8B report generation (prompt in planner.py, fallback to rationale if model unavailable)
       3c. Rlegal computation (substring matching against LEGAL_ELEMENTS)
              │
          Reflector (CASAM, 3-layer audit)
              ├── NONE conflict  → final report
              ├── SOFT conflict  → refine() with target guidance
              └── HARD conflict  → re-analyze() with new context
```

**Planner** (`agents/planner.py`) is rule-based, not an LLM agent — by design to prevent hallucination in scheduling. It owns the Step 3b prompt template (`build_report_prompt()`) and `_call_qwen3()` for Qwen3-8B report generation.

**Reflector** (`agents/reflector.py`) audits three layers: temporal consistency, causal antecedents, and spurious environmental correlations.

### Reward Function

```
R = 0.4·Racc + 0.3·Rcons + 0.2·Rlegal − 0.1·Rcost
```

Weights vary by crime type (violent / property / public_safety) — see `training/reward_functions.py`.

### RAG System (H-RAG)

Dual-layer retrieval in `rag/h_rag.py`:
- **BM25** with jieba Chinese tokenization (keyword precision)
- **BGE-M3** dense embeddings → ChromaDB (semantic recall)
- **RRF** fusion of both layers
- HyDE query enhancement
- Data: Taiwan Criminal Code (per-article chunks) + visual-to-legal mappings

### Training

- **MIL Head** (`scripts/train_mil.py`): trains FusionEncoder + crime_head + escalation_head using R3D-18 + ViT features with CE loss + MIL ranking loss. Must be run before inference produces meaningful results. Weights saved to `outputs/mil_weights/` and auto-loaded by `ActionEmotionAgent`.
- **DPO** (`training/dpo_trainer.py`): primary alignment for report quality, using Gemini 2.0 Flash as pairwise judge (delegates to `evaluation/llm_judge.py`)
- **GRPO** (`training/grpo_trainer.py`): optional Planner policy optimization

### Key Files

| File | Role |
|------|------|
| `crime_analysis/pipeline.py` | Main entry point — `CrimeAnalysisPipeline` |
| `crime_analysis/config.py` | All configuration via dataclasses; global `cfg` instance |
| `crime_analysis/agents/planner.py` | 3-step workflow + Step 3b prompt template + `_call_qwen3()` |
| `crime_analysis/agents/reflector.py` | CASAM consistency auditing + Dempster-Shafer |
| `crime_analysis/agents/action_emotion_agent.py` | R3D-18 + ViT + FusionEncoder + crime/escalation heads |
| `crime_analysis/rag/rag_module.py` | `LEGAL_ELEMENTS`, `GROUP_LEGAL_CONTEXT`, `compute_rlegal()` |
| `crime_analysis/rag/h_rag.py` | Dual-layer RAG backend |
| `crime_analysis/training/reward_functions.py` | Reward calculation + relative advantage normalization |
| `crime_analysis/benchmark/gemini_baseline.py` | Gemini baseline for comparison |
| `crime_analysis/evaluation/llm_judge.py` | Gemini 2.0 Flash LLM-as-Judge (rubric + pairwise, supports Claude/Gemini/OpenAI) |
| `crime_analysis/scripts/train_mil.py` | MIL Head training (R3D-18 + ViT features) |
| `crime_analysis/scripts/pilot_experiment.py` | Small-sample calibration script |
| `crime_analysis/scripts/run_ablation.py` | Ablation study (5 variants) |
| `system_flow.html` | Mermaid diagram of full system flow |

## Data Layout

```
UCA/
├── UCFCrime_{Train|Test|Val}.json  # UCA temporal sentence annotations
├── UCF_Crimes/UCF_Crimes/
│   ├── Videos/{Category}/*.mp4     # 13 crime categories + Normal
│   ├── Action_Regnition_splits/    # ClassIDs.txt, train/test splits
│   └── Anomaly_Detection_splits/   # Anomaly_Train.txt, Anomaly_Test.txt

crime_analysis/data/rag/
├── laws/criminal_code.json         # 33 articles from 刑法 + 家暴法, mapped to 13 categories
└── mappings/visual_to_legal.json   # visual cue → legal element mapping (13 categories)

crime_analysis/outputs/
├── mil_weights/                    # Trained FusionEncoder + crime_head + escalation_head
├── feature_cache/                  # Pre-extracted R3D-18 + ViT features (.npy per video)
├── pilot_stats.json                # Per-case pilot results
└── pilot_summary.{txt,json}        # Aggregated pilot statistics + threshold suggestions
```

UCA JSON format per video: `{duration, timestamps: [[start, end], ...], sentences: [...]}`
Video ID → category: `Abuse001_x264` → `Abuse`, `Normal_Videos_003_x264` → `Normal`

## RTX 5090 部署優化

目標 GPU：NVIDIA RTX 5090（Blackwell, SM 12.0, 32GB GDDR7）

| 優化項目 | 設定位置 | 說明 |
|----------|----------|------|
| CUDA 12.8 | `requirements_freeze.txt` | Blackwell 架構需要 cu128+ |
| BF16 推理 | `config.py → ModelConfig.torch_dtype` | Blackwell 原生 BF16 吞吐量 >> FP32，VRAM 減半 |
| Flash Attention 2 | `config.py → ModelConfig.use_flash_attention` | Qwen 推理 KV-cache 記憶體↓ + 注意力加速 |
| torch.compile | `config.py → ModelConfig.compile_models` | Blackwell inductor 後端，vision models 加速 ~30-50% |
| cuDNN benchmark | `config.py → ModelConfig.cudnn_benchmark` | 自動調優 3D 卷積核（R3D-18 受益最大） |
| TF32 matmul | `pipeline.py → _init_cuda_backends()` | Blackwell 預設 TF32 精度 |
| 混合精度訓練 | `config.py → TrainingConfig.mixed_precision` | BF16 autocast（Blackwell 不需要 GradScaler） |
| DataLoader | `config.py → TrainingConfig.pin_memory/num_workers` | pin_memory + 4 workers 加速資料載入 |
| batch_size=64 | `config.py → TrainingConfig.batch_size` | 32GB VRAM 可支撐 |

降級到其他 GPU 時，修改 `config.py`：
```python
cfg.model.torch_dtype = "float32"       # 不支援 BF16 的 GPU
cfg.model.use_flash_attention = False    # 無 Flash Attention
cfg.model.compile_models = False         # torch.compile 不穩定時關閉
cfg.training.batch_size = 16             # VRAM < 16GB
cfg.training.num_workers = 2             # CPU 核心少時降低
```

## Design Decisions

- `action_agent.py`, `time_emotion_agent.py`, `semantic_agent.py` have been **removed**; use `action_emotion_agent.py` (early-fusion, 1386D → 512D)
- Planner is intentionally **not** an LLM to avoid task-scheduling hallucinations
- Primary training path is **DPO**, not GRPO; GRPO is optional
- RAG uses **both** BM25 and dense retrieval — do not remove either layer; precision and recall are complementary
- `LEGAL_ELEMENTS` is defined once in `rag/rag_module.py` — all other modules import from there
- DS m_env no_crime is a derived value: `0.88 − 0.85 × ae_conf` (not a fixed constant)
- `evaluate()` supports optional `llm_judge` parameter for external semantic scoring alongside internal metrics
- `pipeline.py` uses the new 2-agent architecture (`agents_dict` with `environment` + `action_emotion`), not the old 4-agent list
- MIL Head weights are auto-loaded from `outputs/mil_weights/` if present; without them, classification is random
- `ViTFeatureExtractor` was renamed to `ViTImageProcessor` in transformers v5+; code handles both
- `criminal_code.json` contains only substantive criminal law (刑法 + 家暴法), not procedural law (刑事訴訟法)
