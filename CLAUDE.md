# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Codex will review your output once you are done.

## Project Overview

Multi-agent LLM system for forensic crime video analysis. Uses Qwen3-VL-32B-Instruct (QLoRA INT4) as unified VLM for crime classification and forensic report generation. Combines vision models (R3D-18, ViT) for MIL Head pre-screening, Qwen3-VL (local VLM, classification + report), Gemini 2.0 Flash (judge), hierarchical RAG over Taiwan Criminal Code (刑法), and DPO alignment. Targets the UCF-Crime dataset with UCA temporal annotations. Scope is limited to Taiwan criminal law. All models run locally on RTX 5090 — no API tokens required for inference.

## Setup & Commands

```bash
# Environment setup (one-time) — RTX 5090 requires CUDA 12.8+
conda create -n crime python=3.11
conda activate crime
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate Pillow scikit-learn opencv-python
pip install rank-bm25 jieba chromadb sentence-transformers
pip install mediapipe==0.10.14 deepface tf-keras
pip install "tensorflow==2.16.*" "tf-keras==2.16.*" protobuf==4.25.9
pip install "jax==0.4.30" "jaxlib==0.4.30"
pip install google-genai matplotlib
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y  # for torch.compile

# IMPORTANT: conda activate sets SSL_CERT_FILE to a nonexistent path.
# Always run `unset SSL_CERT_FILE` before any HuggingFace download.
unset SSL_CERT_FILE

# Build RAG index (one-time, after law data is ready)
cd crime_analysis
python data/scripts/build_rag.py

# Train MIL Head (pre-screening, must run before inference)
# Params from config.py: bs=32, ep=60, lr=5e-4, λ_MIL=0.3, wd=1e-2
python -m scripts.train_mil --split Train

# Run analysis on a video
python pipeline.py

# Pilot experiment (small-sample calibration)
python -m scripts.pilot_experiment --n_samples 52 --split Test --seed 42

# Ablation experiments (5 variants, same test set)
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-env
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-rag
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-vlm
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-reflector
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-vlm-report
```

No formal test suite — testing is done via `pipeline.analyze(frames, metadata)` directly.

## Architecture

### Agent Workflow (3-Step, Rule-Based Planner)

```
Video Frames
  │
  ├─[Step 1] Environment Agent   — quality check, occlusion detection (conditional)
  ├─[Step 2] ActionEmotion Agent — R3D-18 + ViT feature fusion → MIL Head (13 classes, pre-screening)
  ├─[Step 2b] Qwen3-VL Classification — VLM zero-shot 分類覆核 MIL Head（4 keyframes）
  └─[Step 3] Planner integrates:
       3a. H-RAG query (BM25 + BGE-M3 + RRF)
       3b. Qwen3-VL report generation (same model as 2b, with keyframes + RAG context)
       3c. Rlegal computation (substring matching against LEGAL_ELEMENTS)
              │
          Reflector (CASAM, 3-layer audit)
              ├── NONE conflict  → final report
              ├── SOFT conflict  → refine() with target guidance
              └── HARD conflict  → re-analyze() with new context
```

**Qwen3-VL-32B-Instruct + QLoRA adapter** is loaded once (~18GB INT4 + 163MB LoRA) and used for both Step 2b classification and Step 3b report generation. Fine-tuned on UCF-Crime training set (59.6% accuracy, surpassing Gemini 2.5 Flash 51.9%).

**MIL Head** (`action_emotion_agent.py`) provides fast pre-screening. VLM overrides its classification in Step 2b.

**Planner** (`agents/planner.py`) is rule-based, not an LLM agent — by design to prevent hallucination in scheduling. It owns `_vlm_classify()`, `_call_qwen3_vl()`, and `build_report_prompt()`.

**Reflector** (`agents/reflector.py`) audits three layers: temporal consistency, causal antecedents, and spurious environmental correlations.

### Models & API Keys

| Model | Runs on | API Key? | Purpose |
|-------|---------|----------|---------|
| Qwen3-VL-32B-Instruct + QLoRA | Local GPU (~18GB INT4) | No | Classification + report generation (fine-tuned 59.6%) |
| R3D-18 / ViT-Base | Local GPU | No | Feature extraction for MIL Head |
| BGE-M3 | Local GPU (~2.2GB) | No | RAG dense embeddings |
| MediaPipe / DeepFace | Local CPU | No | Pose + emotion features |
| Gemini 2.0 Flash | Cloud API | Yes (`GEMINI_API_KEY`) | DPO Judge only (optional) |

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

- **MIL Head** (`scripts/train_mil.py`): trains FusionEncoder + crime_head + escalation_head using 4-modality features (R3D-18 + ViT + Pose + Emotion) with CE loss (class weights + label smoothing) + MIL ranking loss. Includes StandardScaler normalization. Weights + scaler saved to `outputs/mil_weights/` and auto-loaded by `ActionEmotionAgent`.
- **Ablation**: `pilot_experiment.py` supports `--no-env`, `--no-rag`, `--no-vlm`, `--no-reflector`, `--no-vlm-report` flags for controlled experiments. `NullReflector` in `reflector.py` always returns NONE with Rcons=1.0.
- **DPO** (`training/dpo_trainer.py`): primary alignment for report quality, using Gemini 2.0 Flash as pairwise judge (delegates to `evaluation/llm_judge.py`)
- **GRPO** (`training/grpo_trainer.py`): optional Planner policy optimization

### Key Files

| File | Role |
|------|------|
| `crime_analysis/pipeline.py` | Main entry point — `CrimeAnalysisPipeline` |
| `crime_analysis/config.py` | All configuration via dataclasses; global `cfg` instance |
| `crime_analysis/agents/planner.py` | 3-step workflow + `_vlm_classify()` + `_call_qwen3_vl()` (Qwen3-VL) |
| `crime_analysis/agents/reflector.py` | CASAM consistency auditing + Dempster-Shafer |
| `crime_analysis/agents/action_emotion_agent.py` | R3D-18 + ViT + FusionEncoder + crime/escalation heads + StandardScaler |
| `crime_analysis/rag/rag_module.py` | `LEGAL_ELEMENTS`, `GROUP_LEGAL_CONTEXT`, `compute_rlegal()` |
| `crime_analysis/rag/h_rag.py` | Dual-layer RAG backend |
| `crime_analysis/training/reward_functions.py` | Reward calculation + relative advantage normalization |
| `crime_analysis/benchmark/gemini_baseline.py` | Gemini baseline for comparison |
| `crime_analysis/evaluation/llm_judge.py` | Gemini 2.0 Flash LLM-as-Judge (rubric + pairwise, supports Claude/Gemini/OpenAI) |
| `crime_analysis/scripts/train_mil.py` | MIL Head training (4-modality features, class weights, StandardScaler) |
| `crime_analysis/scripts/pilot_experiment.py` | Pilot calibration + ablation experiments + confusion matrix + full report output |
| `crime_analysis/scripts/run_ablation.py` | Legacy ablation study (use pilot_experiment.py --no-* flags instead) |
| `system_flow.html` | Mermaid diagram of full system flow |

## Data Layout

```
UCA/
├── UCFCrime_{Train|Test|Val}.json  # UCA temporal sentence annotations
├── UCF_Crimes/UCF_Crimes/
│   ├── Videos/{Category}/*.mp4     # 13 crime categories (950 anomaly)
│   ├── Training_Normal_Videos_Anomaly/ # 800 normal mp4 (separate download)
│   ├── z_Normal_Videos_event/      # 50 normal mp4 (in UCF.zip)
│   ├── Action_Regnition_splits/    # ClassIDs.txt, train/test splits
│   └── Anomaly_Detection_splits/   # Anomaly_Train.txt, Anomaly_Test.txt

crime_analysis/data/rag/
├── laws/criminal_code.json         # 33 articles from 刑法 + 家暴法, mapped to 13 categories
└── mappings/visual_to_legal.json   # visual cue → legal element mapping (13 categories)

crime_analysis/outputs/
├── mil_weights/                    # Trained FusionEncoder + crime_head + escalation_head + feature_scaler.npz
├── feature_cache/v4/              # Pre-extracted 4-modality features (.npy per video, 1386D)
├── pilot_reports/                  # Full forensic reports per case (.txt)
├── pilot_stats.json                # Per-case pilot results
├── pilot_summary.{txt,json}        # Aggregated pilot statistics + threshold suggestions
└── confusion_matrix.png            # Classification confusion matrix (auto-generated)
```

UCA JSON format per video: `{duration, timestamps: [[start, end], ...], sentences: [...]}`
Video ID → category: `Abuse001_x264` → `Abuse`, `Normal_Videos_003_x264` → `Normal`

## RTX 5090 部署優化

目標 GPU：NVIDIA RTX 5090（Blackwell, SM 12.0, 32GB GDDR7）

| 優化項目 | 設定位置 | 說明 |
|----------|----------|------|
| CUDA 12.8 | `requirements_freeze.txt` | Blackwell 架構需要 cu128+ |
| BF16 推理 | `config.py → ModelConfig.torch_dtype` | Blackwell 原生 BF16 吞吐量 >> FP32，VRAM 減半 |
| torch.compile | `config.py → ModelConfig.compile_models` | Blackwell inductor 後端，vision models 加速 |
| cuDNN benchmark | `config.py → ModelConfig.cudnn_benchmark` | 自動調優 3D 卷積核（R3D-18 受益最大） |
| TF32 matmul | `pipeline.py → _init_cuda_backends()` | Blackwell 預設 TF32 精度 |
| 混合精度訓練 | `config.py → TrainingConfig.mixed_precision` | BF16 autocast（Blackwell 不需要 GradScaler） |
| DataLoader | `config.py → TrainingConfig.pin_memory/num_workers` | pin_memory + 4 workers 加速資料載入 |

降級到其他 GPU 時，修改 `config.py`：
```python
cfg.model.torch_dtype = "float32"       # 不支援 BF16 的 GPU
cfg.model.compile_models = False         # torch.compile 不穩定時關閉
cfg.training.batch_size = 16             # VRAM < 16GB
cfg.training.num_workers = 2             # CPU 核心少時降低
```

## Experimental Results (2026-04-07)

### Classification Accuracy (52 test videos, zero-shot)

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Gemini 2.5 Flash** | **51.9% (27/52)** | Commercial API baseline |
| 8B + 32B Ensemble | 40.4% (21/52) | Best local ensemble |
| Qwen3-VL-32B INT4 | 32.7% (17/52) | Robbery bias (21/52 predictions) |
| Qwen3-VL-8B standalone | 30.8% (16/52) | Best single local model |
| Pipeline (8B, fixed) | ≈30.8% | Matches standalone after fixes |
| Gemma-4-E4B | 21.2% (11/52) | Too small |
| MIL Head only | 15.4% (4/26) | Shooting/Explosion bias |
| InternVL3-8B | 0% (0/13) | Incompatible with task |

### MIL + VLM Ensemble Analysis (26 videos)

| Method | Accuracy |
|--------|----------|
| MIL Head only | 15.4% (4/26) |
| VLM (8B) only | 34.6% (9/26) |
| **MIL + VLM Ensemble** | **42.3% (11/26)** |

Complementarity: 2 both-right, 2 MIL-only, 7 VLM-only, 15 both-wrong.

### Pipeline Metrics (Pilot, 52 videos)

| Metric | Ours | Gemini |
|--------|------|--------|
| Racc | 25.0% | 51.9% |
| **Rlegal** | **0.956** | 0.000 |
| Rcons | 0.905 | 1.000 |
| Report quality | Qwen3-VL with frames | Gemini direct |
| Local deployment | ✅ | ❌ |

### Key Findings

1. **Pipeline no longer hurts classification** — fixed MIL rationale poisoning, Reflector override, prompt length issues
2. **VLM systematic bias**: Qwen3-VL-8B over-predicts Robbery (6/13) and Burglary (3/13)
3. **Frame count**: 8 frames > 16 frames (attention dilution confirmed)
4. **Prompt length**: Simple prompt > detailed/decision-tree prompt
5. **Rlegal fix effective**: Two-tier scoring (article numbers + elements) no longer always 1.0
6. **QLoRA fine-tune in progress**: Qwen3-VL-32B on UCF-Crime training set, expected 50-65%

### Systematic Confusion Pairs (15 both-wrong cases)

```
VLM wrong predictions: Robbery (6x), Burglary (5x), Vandalism (2x)
Top confusions:
  Assault/Fighting → Robbery (4x)  — violence misread as theft
  Arson/Arrest/Stealing → Burglary (3x) — scene misread as break-in
  8 cross-group errors, 7 within-group errors
```

## Design Decisions

- `action_agent.py`, `time_emotion_agent.py`, `semantic_agent.py` have been **removed**; use `action_emotion_agent.py` (early-fusion, 1386D → 512D)
- **Qwen3-VL-32B-Instruct (QLoRA INT4)** is the target unified VLM. Currently using 8B for pipeline, 32B for fine-tuning.
- Planner is intentionally **not** an LLM to avoid task-scheduling hallucinations
- MIL Head provides fast pre-screening; VLM overrides classification in Step 2b
- Feature extraction uses **StandardScaler** (mean/std saved in `feature_scaler.npz`) to normalize cross-modality scale differences
- R3D-18 features are **not** L2-normalized — amplitude information is preserved for classification
- Primary training path is **DPO**, not GRPO; GRPO is optional
- RAG uses **both** BM25 and dense retrieval — do not remove either layer; precision and recall are complementary
- `LEGAL_ELEMENTS` is defined once in `rag/rag_module.py` — all other modules import from there
- `compute_rlegal` uses two-tier scoring: 60% article number matching (e.g. "第277條") + 40% legal element substring matching (with negation detection). This ensures `no_rag` ablation shows meaningful Rlegal decrease.
- DS m_env no_crime is a derived value: `0.88 − 0.85 × ae_conf` (not a fixed constant)
- `evaluate()` supports optional `llm_judge` parameter for external semantic scoring alongside internal metrics
- `pipeline.py` uses the new 2-agent architecture (`agents_dict` with `environment` + `action_emotion`), not the old 4-agent list
- MIL Head weights are auto-loaded from `outputs/mil_weights/` if present; without them, classification is random
- `criminal_code.json` contains only substantive criminal law (刑法 + 家暴法), not procedural law (刑事訴訟法)
- Flash Attention 2: `flash-attn` does not yet support sm_120 (Blackwell); planner.py auto-fallbacks to default attention
