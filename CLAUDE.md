# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Codex will review your output once you are done.

## Project Overview

Multi-agent LLM system for forensic crime video analysis. Uses Qwen3-VL-32B-Instruct (QLoRA INT4) as unified VLM for crime classification and forensic report generation. Combines vision models (R3D-18, ViT) for MIL Head pre-screening, Qwen3-VL (local VLM, classification + report), Gemini 2.0 Flash (judge, legacy), and **Hybrid RAG** (BM25 + BGE-M3 dense + RRF fusion) over Taiwan Criminal Code (刑法). Targets the UCF-Crime dataset with UCA temporal annotations. Scope is limited to Taiwan criminal law. All models run locally on RTX 5090 — no API tokens required for inference.

**Category scope (14-way classification)**:
- **Crime (10)**: Assault, Robbery, Stealing, Shoplifting, Burglary, Fighting, Arson, Vandalism, Abuse, Shooting — enter full H-RAG + Reflector pipeline with Taiwan Criminal Code mapping
- **Non-crime anomaly (3)**: Arrest (law enforcement action), RoadAccidents (civil/administrative), Explosion (cannot determine criminal intent from frames alone) — short-circuited after classification; no H-RAG, no Rlegal, no Reflector; flagged as `is_non_crime_anomaly=True` with descriptive-only report
- **Normal (1)**: VLM can now explicitly output Normal; bypasses crime report generation

See [Non-Crime Category Handling](#non-crime-category-handling) below and `/home/yuting/.claude/plans/ucf-polished-nova.md` for rationale.

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

# Pilot experiment (small-sample calibration, 52 = 13 crime × 4 + Normal × 4)
python -m scripts.pilot_experiment \
    --n_samples 52 --n-normal 4 --split Test --seed 42 \
    --bias-correction data/bias_priors_qwen3vl32b_ft.json \
    --output_dir outputs/pilot_vN

# Resume interrupted run (re-uses already-produced per-case results)
python -m scripts.pilot_experiment ... --resume

# Ablation experiments (5 variants, same test set)
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-env
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-rag
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-vlm
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-reflector
python -m scripts.pilot_experiment --n_samples 154 --split Test --seed 42 --no-vlm-report

# Experiment 1 — VLM baseline (0-shot, symmetric, full Test data)
python -m scripts.experiment_1_baseline \
    --models qwen3vl_32b gemma4_26b_a4b internvl3_38b \
    --n_samples 1000 --n-normal 200 --split Test --seed 42 \
    --output-dir outputs/experiment_1

# LLM-as-a-Judge cross-evaluation (2 neutral judges × 2 models)
# Requires ANTHROPIC_API_KEY + OPENAI_API_KEY in .env
python -m scripts.run_cross_evaluation \
    --qwen-reports outputs/pilot_v5/pilot_reports \
    --gemini-reports outputs/experiments/gemini_baseline/pilot_reports \
    --output-dir outputs/cross_eval/pilot_v5
# (Optional) override judges:
#   --judges claude-sonnet-4-20250514,gpt-4o

# Human-vs-LLM correlation on sampled subset (after cross-eval)
python -m scripts.sample_for_human_eval \
    --qwen-reports outputs/pilot_v5/pilot_reports \
    --gemini-reports outputs/experiments/gemini_baseline/pilot_reports \
    --cross-eval-dir outputs/cross_eval/pilot_v5 \
    --per-category 2 --seed 42
# → 人工用 outputs/.../human_sample/questionnaire.html 評完後下載 CSV，再跑:
python -m scripts.compute_human_llm_correlation \
    --human outputs/cross_eval/pilot_v5/human_sample/human_scores.csv \
    --cross-eval outputs/cross_eval/pilot_v5/per_report.json \
    --output outputs/cross_eval/pilot_v5/human_sample/validation.json
```

No formal test suite — testing is done via `pipeline.analyze(frames, metadata)` directly.

## Architecture

### Agent Workflow (3-Step, Rule-Based Planner)

```
Video Frames
  │
  ├─[Step 1] Environment Agent   — quality check, occlusion detection (conditional)
  ├─[Step 2] ActionEmotion Agent — R3D-18 + ViT feature fusion → MIL Head (13 classes, pre-screening)
  ├─[Step 2a] Anomaly gate       — escalation_score < τ → force Normal (bypass VLM)  [optional]
  ├─[Step 2b] Qwen3-VL Classification — VLM 14-way (10 crime + 3 non-crime + Normal)
  ├─[Step 2c-2d] RAG verify      — element-match check; keeps VLM original (no force-swap)
  │
  ├─ crime_type ∈ NON_CRIME_CATEGORIES (Arrest/RoadAccidents/Explosion)
  │    → _synthesize_non_crime_report(): descriptive summary only, skip 3a/3c/Reflector
  │
  ├─ crime_type == "Normal"
  │    → bypass_vlm_report=True; minimal Normal report via _synthesize_final_report
  │
  └─[Step 3] Planner integrates (crime only):
       3a. H-RAG query (BM25 + BGE-M3 + RRF)
       3b. Qwen3-VL report generation (keyframes + RAG context)
       3c. Rlegal computation (substring matching against LEGAL_ELEMENTS)
              │
          Reflector (CASAM, 3-layer audit)
              ├── NONE conflict  → final report
              ├── SOFT conflict  → refine() with target guidance
              └── HARD conflict  → re-analyze() with new context
```

**Qwen3-VL-32B-Instruct + QLoRA adapter** is loaded once (~18GB INT4 + 163MB LoRA) and used for both Step 2b classification and Step 3b report generation. Fine-tuned on UCF-Crime training set (59.6% accuracy, surpassing Gemini 2.5 Flash 51.9%).

**MIL Head** (`action_emotion_agent.py`) provides fast pre-screening with 13-class `crime_head` + `escalation_head`. VLM overrides its classification in Step 2b. ⚠️ **Known issue** (pilot_v3): `escalation_head` weights are near Xavier init (std=0.04, mean≈0) because the MIL ranking loss in `train_mil.py` applies per-video instead of per-snippet and the `sparsity` term drives outputs to 0. 2-stage anomaly gate is effectively unusable until retrained — see Design Decisions → "escalation_head retraining deferred".

**Planner** (`agents/planner.py`) is rule-based, not an LLM agent — by design to prevent hallucination in scheduling. It owns `_vlm_classify()`, `_call_qwen3_vl()`, `build_report_prompt()`, `build_non_crime_report_prompt()`, and the two synthesis paths `_synthesize_final_report()` / `_synthesize_non_crime_report()`.

**Reflector** (`agents/reflector.py`) audits three layers: temporal consistency, causal antecedents, and spurious environmental correlations. Only invoked for crime cases; non-crime and VLM-predicted-Normal paths bypass it.

### Non-Crime Category Handling

Three UCF-Crime categories are **not criminal offenses** under Taiwan Criminal Code:

| UCF class | Why non-crime |
|---|---|
| **Arrest** | Law enforcement action; "逮捕" itself is not a 罪名 |
| **RoadAccidents** | Traffic law / civil liability, not criminal |
| **Explosion** | Cannot determine criminal intent from visual alone (may be public-safety offense OR accident) |

Defined in `config.py`:
```python
CRIME_CATEGORIES     = frozenset({Robbery, Stealing, ...})   # 10 classes
NON_CRIME_CATEGORIES = frozenset({Arrest, RoadAccidents, Explosion})
is_crime(cat)              # → enters full H-RAG + Reflector
is_non_crime_anomaly(cat)  # → short-circuit branch
```

**Two-stage decision flow** in `planner.run()` (after classification is finalized):
- `is_non_crime_anomaly(crime_type)` → `_synthesize_non_crime_report()` returns early with `is_non_crime_anomaly=True`, empty `legal_classification`, `rlegal=0`, `conflict_type="NONE"`
- `crime_type == "Normal"` (VLM-predicted or anomaly-gated) → `bypass_vlm_report=True`, goes through normal `_synthesize_final_report`
- `is_crime(crime_type)` → full crime flow

**Evaluation implications** (see `scripts/pilot_experiment.py` `_compute_summary`):
- `crime_only_accuracy`: 10-way accuracy on crime cases (main metric for thesis)
- `non_crime_to_crime_fp_rate`: FP control (% of non-crime cases predicted as crime; lower = better)
- `non_crime_3way_accuracy`: secondary (Arrest/RoadAccidents/Explosion correctly distinguished)
- `rlegal_gt` computed only on crime cases (`is_crime(gt)`); non-crime always 0

Per-case `pilot_stats.json` carries: `gt_is_crime`, `gt_is_non_crime`, `pred_is_crime`, `pred_is_non_crime`, `is_non_crime_anomaly`, `event_category`.

### Models & API Keys

| Model | Runs on | API Key? | Purpose |
|-------|---------|----------|---------|
| Qwen3-VL-32B-Instruct + QLoRA | Local GPU (~18GB INT4) | No | Classification + report generation (fine-tuned 59.6%) |
| R3D-18 / ViT-Base | Local GPU | No | Feature extraction for MIL Head |
| BGE-M3 | Local GPU (~2.2GB) | No | RAG dense embeddings |
| MediaPipe / DeepFace | Local CPU | No | Pose + emotion features |
| Gemini 2.0 Flash | Cloud API | Yes (`GEMINI_API_KEY`) | LLM-as-Judge / benchmark only (DPO dropped) |

### Reward Function

```
R = 0.4·Racc + 0.3·Rcons + 0.2·Rlegal − 0.1·Rcost
```

Weights vary by crime type (violent / property / public_safety) — see `training/reward_functions.py`.

### RAG System — Hybrid Retrieval

**Named `HierarchicalRAG` in code for historical reasons, but architecturally it is Hybrid RAG** (dual-track parallel retrieval + rank fusion, NOT multi-level hierarchy). For thesis/publication, refer to it as "Hybrid RAG" or "H-RAG (Hybrid)".

Hybrid retrieval in `rag/h_rag.py`:
- **BM25** with jieba Chinese tokenization — keyword precision for legal terms (e.g. `第277條`)
- **BGE-M3 dense embeddings** → ChromaDB — semantic recall for behavior descriptions
- **RRF (Reciprocal Rank Fusion)** — merge BM25 + dense rankings
- **HyDE query enhancement** — generate hypothetical legal description before retrieval
- Data: Taiwan Criminal Code (33 articles as per-article chunks)
- **Judgments collection is empty** — 判決書無法對齊 UCF-Crime 影像，本研究不納入

**Static category → article routing** (separate from hybrid retrieval):
- `GROUP_LEGAL_CONTEXT` in `rag/rag_module.py` maps each crime category to its expected articles (e.g. `Robbery → [第328條, 第330條]`)
- Used by `compute_rlegal` for two-tier scoring (60% article match + 40% element match)
- Provides a **fast direct lookup** when VLM classification is confident; Hybrid retrieval complements when lookup is insufficient or category is uncertain

### Training

- **MIL Head** (`scripts/train_mil.py`): trains FusionEncoder + crime_head + escalation_head using 4-modality features (R3D-18 + ViT + Pose + Emotion) with CE loss (class weights + label smoothing) + MIL ranking loss. Includes StandardScaler normalization. Weights + scaler saved to `outputs/mil_weights/` and auto-loaded by `ActionEmotionAgent`. ⚠️ Current MIL ranking loss operates per-video (not per-snippet) and has a dominating `sparsity` term — `escalation_head` barely learned. Retraining with proper per-snippet loss or BCE is deferred (Shooting 0% root cause is VLM vision, not MIL, so retraining priority is low).
- **Ablation**: `pilot_experiment.py` supports `--no-env`, `--no-rag`, `--no-vlm`, `--no-reflector`, `--no-vlm-report` flags for controlled experiments. `NullReflector` in `reflector.py` always returns NONE with Rcons=1.0.
- **Bias correction**: `--bias-correction <path>` loads per-category logit-space offsets from JSON (e.g. `data/bias_priors_qwen3vl32b_ft.json`). Subtracted from VLM logits before softmax; only flips top-1 when another class crosses. **Current setting (2026-04-23)**: non-crime categories (Arrest / RoadAccidents / Explosion) all set to **0.0** — they are supposed to trigger the non-crime short-circuit, not be suppressed. Previous v2 set RoadAccidents=1.5 which caused 4/4 RoadAccidents → Normal misclassification in pilot_v4.
- ~~**DPO**~~ — dropped (2026-04-22); `training/dpo_trainer.py` retained for reference but no longer used in primary experiments.
- **GRPO** (`training/grpo_trainer.py`): optional Planner policy optimization, not actively used.

### Key Files

| File | Role |
|------|------|
| `crime_analysis/pipeline.py` | Main entry point — `CrimeAnalysisPipeline` |
| `crime_analysis/config.py` | Dataclass configs + `CRIME_CATEGORIES` / `NON_CRIME_CATEGORIES` / `is_crime()` / `is_non_crime_anomaly()` |
| `crime_analysis/agents/planner.py` | 3-step workflow + `_vlm_classify()` (14-way with Normal) + `_rag_verify_classification()` (verify-only, no swap) + `_synthesize_non_crime_report()` + `build_non_crime_report_prompt()` |
| `crime_analysis/agents/reflector.py` | CASAM consistency auditing + Dempster-Shafer |
| `crime_analysis/agents/action_emotion_agent.py` | R3D-18 + ViT + FusionEncoder + crime/escalation heads + StandardScaler |
| `crime_analysis/rag/rag_module.py` | `LEGAL_ELEMENTS`, `GROUP_LEGAL_CONTEXT`, `compute_rlegal()` |
| `crime_analysis/rag/h_rag.py` | Dual-layer RAG backend |
| `crime_analysis/training/reward_functions.py` | Reward calculation + relative advantage normalization |
| `crime_analysis/benchmark/gemini_baseline.py` | Gemini baseline for comparison |
| `crime_analysis/evaluation/llm_judge.py` | LLM-as-Judge: **7-question rubric** (Q1 法條合理性 / Q2 要件覆蓋率 / Q3 幀號可追溯 / Q4 因果鏈 / Q5 不確定性 / Q6 司法語言 / Q7 情景事實吻合度). Supports Claude/Gemini/OpenAI backend; legacy 5-dim `RUBRIC_DIMENSIONS` kept for backward compat. |
| `crime_analysis/scripts/train_mil.py` | MIL Head training (4-modality features, class weights, StandardScaler) |
| `crime_analysis/scripts/pilot_experiment.py` | Pilot calibration + ablation experiments + confusion matrix + full report output |
| `crime_analysis/scripts/experiment_1_baseline.py` | Experiment 1: 0-shot VLM baseline (Qwen3-VL-32B + Gemma-4-26B-A4B + InternVL3-38B); binary P/R/F1/Acc video-level metrics, per-case resume |
| `crime_analysis/scripts/run_cross_evaluation.py` | LLM-as-Judge symmetric 2-judge cross-eval (Qwen reports + Gemini reports, both judged by {Claude, OpenAI}). Open-book γ: shows UCA scenario + cited 刑法 articles to judge. |
| `crime_analysis/scripts/sample_for_human_eval.py` | Stratified sample 20 crime videos × 2 models = 40 reports; renders self-contained `questionnaire.html` with open-book γ context |
| `crime_analysis/scripts/compute_human_llm_correlation.py` | Human vs LLM judge ICC(2,1) / Pearson / Spearman after questionnaire returns |
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
├── laws/criminal_code.json         # 33 articles from 刑法 + 家暴法, mapped to 10 crime categories
└── mappings/visual_to_legal.json   # visual cue → legal element mapping

crime_analysis/data/
└── bias_priors_qwen3vl32b_ft.json  # VLM per-category logit bias (subtracted in classification)

crime_analysis/outputs/
├── mil_weights/                    # Trained FusionEncoder + crime_head + escalation_head + feature_scaler.npz
├── feature_cache/v4/               # Pre-extracted 4-modality features (.npy per video, 1386D)
├── pilot_v{N}/                     # Per-run pilot output (v3 is the current baseline snapshot)
│   ├── pilot_reports/              # Full forensic reports per case (.txt)
│   ├── pilot_stats.json            # Per-case results (includes gt_is_crime, is_non_crime_anomaly, ...)
│   ├── pilot_summary.{txt,json}    # Aggregated stats + crime_only_accuracy + non_crime_to_crime_fp_rate
│   └── confusion_matrix.png
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

## Experimental Results

### Pilot v3 (2026-04-22, 52 videos, Qwen3-VL-32B QLoRA + bias correction)

Baseline snapshot **before** the three targeted fixes (non-crime branch, VLM Normal option, bias prior raise, RAG verify no-swap).

| Metric | Pilot v3 | Old baseline (CLAUDE.md 2026-04-07) | Δ |
|---|---|---|---|
| Overall accuracy (14-way) | **51.8%** | 25.0% | +26.8pp |
| Rlegal (predicted) | 0.918 | 0.956 | −3.8% |
| Rcons | 0.982 | 0.905 | +7.7% |
| Rlegal_GT gap (correct − wrong) | **0.824** | — | strong signal |

**Per-category accuracy (52 samples = 13 crime × 4 + Normal × 4, but only 4 × Normal)**:

| Tier | Categories |
|---|---|
| ≥ 75% | Burglary 100%, Robbery 100%, Arrest 75%, Explosion 75%, Shoplifting 75%, Stealing 75% |
| 50% | Arson, RoadAccidents, Vandalism |
| ≤ 25% | Abuse 25%, Assault 25%, Fighting 25%, **Shooting 0%**, **Normal 0%** |

### Known failure modes in Pilot v3 (fixed in subsequent commits)

1. **VLM over-predicts RoadAccidents** (confidence often > 0.9): caused 2/4 Shooting and 3/4 Normal failures. ✅ Mitigated by raising bias prior 0.636 → 1.5.
2. **VLM has no Normal option**: forced to pick a crime on normal scenes. 3/4 Normals had entropy > 1.5 (uncertain) but still picked a crime. ✅ Mitigated by adding Normal to VLM prompt / `VLM_CATEGORIES`.
3. **RAG verify force-swaps when elements don't match**: Shooting033 → Robbery because VLM saw "0/3 shooting elements" and was prompted to pick from same-group alternatives. ✅ Mitigated by making RAG verify *verify-only* (no swap); original VLM classification preserved.
4. **MIL `escalation_score` not discriminative**: all Normal/Shooting/Arrest/Explosion cases cluster near 0, making 2-stage anomaly gate unusable at any threshold. Root cause: per-video MIL ranking loss + dominating sparsity term. ⚠️ Deferred; retraining cost high, won't fix Shooting which is a VLM vision problem.

### Systematic Confusion Pairs (legacy 2026-04-07 data, pre-fixes)

```
VLM wrong predictions: Robbery (6x), Burglary (5x), Vandalism (2x)
Top confusions:
  Assault/Fighting → Robbery (4x)  — violence misread as theft
  Arson/Arrest/Stealing → Burglary (3x) — scene misread as break-in
```

## Design Decisions

- `action_agent.py`, `time_emotion_agent.py`, `semantic_agent.py` have been **removed**; use `action_emotion_agent.py` (early-fusion, 1386D → 512D)
- **Qwen3-VL-32B-Instruct (QLoRA INT4)** is the target unified VLM. Currently using 8B for pipeline, 32B for fine-tuning.
- Planner is intentionally **not** an LLM to avoid task-scheduling hallucinations
- MIL Head provides fast pre-screening; VLM overrides classification in Step 2b
- Feature extraction uses **StandardScaler** (mean/std saved in `feature_scaler.npz`) to normalize cross-modality scale differences
- R3D-18 features are **not** L2-normalized — amplitude information is preserved for classification
- ~~Primary training path is DPO~~ — DPO dropped 2026-04-22; GRPO also optional and unused.
- RAG uses **both** BM25 and dense retrieval — do not remove either layer; precision and recall are complementary
- `LEGAL_ELEMENTS` is defined once in `rag/rag_module.py` — all other modules import from there
- `compute_rlegal` uses two-tier scoring: 60% article number matching (e.g. "第277條") + 40% legal element substring matching (with negation detection). This ensures `no_rag` ablation shows meaningful Rlegal decrease.
- DS m_env no_crime is a derived value: `0.88 − 0.85 × ae_conf` (not a fixed constant)
- `evaluate()` supports optional `llm_judge` parameter for external semantic scoring alongside internal metrics
- `pipeline.py` uses the new 2-agent architecture (`agents_dict` with `environment` + `action_emotion`), not the old 4-agent list
- MIL Head weights are auto-loaded from `outputs/mil_weights/` if present; without them, classification is random
- `criminal_code.json` contains only substantive criminal law (刑法 + 家暴法), not procedural law (刑事訴訟法)
- Flash Attention 2: `flash-attn` does not yet support sm_120 (Blackwell); planner.py auto-fallbacks to default attention

### Recent fixes (2026-04-22, addressing pilot_v3 failure modes)

- **Non-crime two-stage decision** — Arrest / RoadAccidents / Explosion route to `_synthesize_non_crime_report()` and skip H-RAG + Reflector. Rationale in thesis §3.x "Research Scope and Classification Design"; plan file at `/home/yuting/.claude/plans/ucf-polished-nova.md`.
- **`VLM_CATEGORIES = UCF_CATEGORIES + ["Normal"]`** (planner.py module-level) — used **only** for VLM prompt/parsing. `UCF_CATEGORIES` in `action_emotion_agent.py` stays 13 because MIL head output dimension is tied to it. When VLM returns Normal, `bypass_vlm_report=True` short-circuits crime report generation.
- **RAG verify is verify-only** — `_rag_verify_classification` always returns `None` now (logs match count only). Caller block in `run()` simplified; previous force-swap logic caused Shooting033 → Robbery.
- **`escalation_head` retraining deferred** — pilot_v3 shows head outputs cluster near 0 for all classes including Shooting (making 2-stage gate useless). Root cause is broken MIL loss in `train_mil.py` (per-video instead of per-snippet, sparsity term dominates). Retraining cost is high (~2h) and won't fix Shooting 0% which is a VLM vision problem, so deferred.

### Pilot_v5 fixes (2026-04-23, addressing pilot_v4 regressions)

Pilot_v4 introduced Fix #1 (add Normal to VLM prompt) which caused Shoplifting 75%→0% (VLM picks Normal for store scenes) and RoadAccidents 50%→0% (bias=1.5 suppresses it). Three targeted fixes:

- **Fix A — Non-crime bias reset to 0.0** in `data/bias_priors_qwen3vl32b_ft.json`. Rationale: non-crime categories should trigger the short-circuit, not be suppressed. Previous v2 bias=1.5 on RoadAccidents caused all 4 RoadAccidents → Normal in pilot_v4.
- **Fix B — Stricter Normal definition** in planner.py VLM classify prompt. Explicitly lists "do NOT pick Normal if concealing items, physical contact between strangers, weapons, visible damage, loitering without purchase intent, or fleeing."
- **Fix C — Shoplifting "looks normal" hint** in planner.py. Prompt now notes "Shoplifting often LOOKS like ordinary shopping — look for pocketing items, hiding merchandise, leaving without checkout."

### LLM-as-a-Judge cross-evaluation framework (2026-04-23)

Replaces the 5-dim abstract rubric with a 7-question anchored rubric designed for Taiwan forensic report evaluation. Key design:

- **7 questions** (Q1–Q7) with specific anchors; normalized to 0–1 per question (max varies: Q2 dynamic by cited articles' element count, Q4=4, others=3).
- **Q7 scenario fidelity is the only GT-anchored question** — compares report facts to UCA scenario description. Other 6 questions evaluate "relevance/coverage", not "correctness", because legal GT doesn't exist.
- **Open-book γ** — judge prompt includes UCA scenario + full text of any 刑法 articles the report cites (regex extracts Arabic `185-3` / `277-A` / `320-SH` suffixes + Chinese numerals `第六條` / `第二十五條` via `_cn_to_arabic`).
- **Symmetric 2-judge cross-evaluation** — both Qwen and Gemini reports judged by {Claude, OpenAI} (both neutral third parties). `run_cross_evaluation.py --judges` overridable.
- **Inter-judge agreement** via Pearson / Spearman; **human vs LLM ICC(2,1)** via stratified-sampled questionnaire (`sample_for_human_eval.py` + `compute_human_llm_correlation.py`).
- **Blocked on API keys** — need `ANTHROPIC_API_KEY` + `OPENAI_API_KEY` in `.env` (only `GEMINI_API_KEY` currently set).

### Experiment 1 — Pre-trained VLM Baseline (2026-04-23)

Mirrors senior thesis Experiment 1 with max-spec variants fitting on 5090 INT4:

- **Qwen3-VL-32B-Instruct** (INT4 ~18GB) — current base
- ~~Qwen3.5-35B-A3B (MoE)~~ — **dropped**; too similar to Qwen3-VL-32B
- **Gemma-4-26B-A4B-it** (MoE, INT4 ~13GB) — cached
- **InternVL3-38B** (INT4 ~20GB) — downloaded

Metrics: video-level binary anomaly detection (Precision / Recall / F1 / Accuracy) for apples-to-apples comparison with senior's Table 4-3. Plus 14-way classification accuracy as reference.

Run: `python -m scripts.experiment_1_baseline --models qwen3vl_32b gemma4_26b_a4b internvl3_38b --n_samples 1000 --n-normal 200`

**Controlled variables audit**: see `outputs/experiment_1/controls.md` — documents what's held constant (sample set, prompt, frames, quantization, decoding) vs. intentional differences (architecture/params/pretraining). Use as thesis appendix and reviewer-response reference. Known cosmetic deviation: Gemma4's `classify_gemma4` is missing the explicit `temperature=0.1, do_sample=False` kwargs that Qwen/InternVL have, but is functionally equivalent (HF defaults to greedy; temperature ignored under greedy).

### Cleanup (2026-04-23)

- Removed 11 one-off test/legacy scripts and 73MB of stale outputs
- **Qwen3.5-35B-A3B fully purged** (2026-04-23 later): 67GB HF cache + 33MB FT adapter + all code refs (`load_qwen35_35b_a3b`, `classify_qwen35`, CLI/MODEL_REGISTRY entries). Model was dropped from Experiment 1 because it was too similar to Qwen3-VL-32B.
- Shell helpers added:
  - `cleanup.sh` — purges unused scripts/outputs (one-shot)
  - `exp1_queue.sh` / `stop_rerun_launch_v5.sh` — overnight orchestration

### RAG naming (2026-04-23)

- Class `HierarchicalRAG` in code is a legacy name. Architecture is **Hybrid Retrieval** (BM25 + dense in parallel + RRF fusion), not multi-level hierarchy.
- **Thesis writing convention**: refer to the system as **"Hybrid RAG"**.
  - Academically accurate (cf. Lewis et al. 2020 RAG; Karpukhin et al. 2020 DPR; Cormack et al. 2009 RRF)
  - No need to rename class/file — keep `HierarchicalRAG` for code stability
- If asked "is this hierarchical?" the honest answer is: "It's dual-track hybrid retrieval with static category → article routing as a fallback; not a textbook hierarchical RAG."

### Pilot_v7 fixes (2026-04-24, agent conflict resolution logic optimization)

Post-pilot_v6 code review uncovered three logical flaws in the conflict resolution mechanism. All fixed in architectural changes (no prompt tuning, no overfit risk). Plan at `/home/yuting/.claude/plans/ucf-polished-nova.md` (Agent 衝突解決邏輯優化 section).

**Fix E1 — MIL + VLM 獨立雙意見（L1 HARD ④ 復活）**
- Before: VLM prediction overwrote `ae_report.crime_category`; L1 HARD ③ category-mismatch check iterated over a single-element `non_normal` list → dead code.
- After: MIL prediction preserved in `ae_report.metadata["mil_crime_type"]` + `["mil_confidence"]` (already was; no planner change needed). Reflector `_layer1_temporal` adds HARD ④: if `vlm_used` AND MIL ≠ VLM AND both non-Normal AND `|conf_diff| > CONFIDENCE_GAP_HARD (0.4)` → HARD re-analyze.
- File: `reflector.py` HARD ④ block after HARD ③.
- Paper: "MIL Head 作為第二意見保留，僅在與 VLM 顯著衝突時啟動 re-analyze。"

**Fix E2 — RAG element match 注入 Rcons + SOFT**
- Before: `_rag_verify_classification` always returned `None`; element match count only logged.
- After:
  - `planner.py` `_rag_verify_classification` returns `Optional[Tuple[int, int]]` (matched, total).
  - Caller gate widened to `ae_confidence > 0.3` (was `0.3 < ae_confidence < 0.7`), stores result in `ae_report.metadata["rag_element_match_ratio"]`.
  - `reflector.py` L2 adds SOFT trigger: `ratio < 0.3 AND conf > 0.7 AND is_crime(cat)` → SOFT refine `element_mismatch`.
  - Dempster-Shafer `_mass` multiplies `m_crime` by calibration factor `(0.5 + 0.5 × ratio)`: ratio=1.0 → no change, ratio=0.0 → 50% reduction in crime belief.
- Unit test: Rcons with match=3/3 → 0.758; with match=0/3 → 0.495; gap 0.263.
- Paper: "法律要件匹配率作為 VLM 分類的獨立驗證信號，結合 Dempster-Shafer 信念融合調整一致性分數。"

**Fix E3 — escalation_head 守衛**
- Before: L2 HARD ① triggered when `escalation < 0.15 AND conf > 0.96 AND HIGH_SEVERITY`. Since escalation_head outputs ≈0 for all cases (broken MIL ranking loss; see pilot_v3 notes), any high-conf Shooting/Robbery/Assault triggered unnecessary re-analyze.
- After: `reflector.py` adds module-level `ESCALATION_HEAD_TRUSTED = False`. L2 HARD ① guarded by this flag. Set to `True` if escalation_head is retrained with per-snippet BCE loss.
- L2 HARD ② (n_antecedents=0 check) kept as-is — depends on `pre_crime_indicators` list, not directly on escalation_head.
- Paper: "L2 因果審查依賴 escalation_head 的情緒升溫指標；當該指標訓練不足時，機制降級為僅依賴前兆計數，避免誤觸 HARD 衝突。"

**Ancillary tuning (2026-04-24, applied before E1/E2/E3)**:
- Bias priors v4: Shooting +0.37 → −0.50 (flip to boost), Burglary +0.20 → −0.20 (flip), Arson/Fighting deepened to −0.80, Assault/RoadAccidents new −0.30 boost. Abuse at −1.10 already maxed.
- `CONFIDENCE_HARD_THRESHOLD`: 0.75 → **0.96** (pilot_v6 correct-prediction p75)
- `CONFIDENCE_SOFT_THRESHOLD`: 0.70 → **0.81** (pilot_v6 correct-prediction p50)
- `_rcost_threshold_high`: 6 → **4** (pilot_v6 turns p75)

**Fix F1 — Pre-classification RAG Priming (2026-04-24)**
- **Motivation**: Before F1, VLM classified "cold" — seeing all 14 category definitions but no hint about which were more likely. MIL Head's top-3 prediction was discarded. F1 feeds MIL top-3 + their `visual_to_legal.json` cues into the VLM prompt as a preliminary hint, letting VLM focus attention while retaining agency to override.
- **Flow**: `MIL crime_head (13-way) → top-3 with probs → visual_to_legal.json direct-evidence cues (top-2 per category) → prepend to VLM classify prompt → VLM decides`
- **Files**:
  - `action_emotion_agent.py`: `_classify_crime` returns `Tuple[str, float, List[Tuple[str, float]]]` (top-1 cat, conf, top-3 list). Stored in `ae_report.metadata["mil_top3"]`.
  - `planner.py`: new `format_priming_section(mil_top3)` helper; `_vlm_classify` accepts optional `ae_report` and prepends priming section when available. Caller passes `ae_report` at line 459.
- **Safety**: If MIL top-1 conf < 0.5 (raised from 0.2 in pilot_v8 RCA — see Pilot_v8/v9 fixes below), priming is skipped. Priming wording ("preliminary analysis... BUT rely on what you actually see") gives VLM license to override MIL.
- **Paper**: 第三章可寫「本研究將 MIL Head 的初步預測與 Hybrid RAG 的視覺-法律對應表結合，於 VLM 分類前注入 top-3 候選類別的視覺特徵提示（Pre-classification RAG Priming），改善細粒度分類。」

### Pilot_v8/v9 fixes (2026-04-25, addressing pilot_v7 14-way regression)

Pilot_v7 with E1/E2/E3 + F1 + bias_priors v4 showed: **binary F1 0.884 → 0.905 ✅** but **14-way 48.2% → 33.9% ❌** and **crime-only 42.5% → 20.0% ❌**. Three-pronged root cause:

1. **bias_priors v4 over-correction** — fixed by **v5 conservative rollback** (Shooting −0.50→−0.20, Burglary −0.20→0, Fighting −0.80→−0.50, Arson back to −0.59, Assault −0.30→−0.10, RoadAccidents −0.30→−0.20, Robbery +0.37→+0.10, Stealing +0.20→0). v5 keeps directional fixes at half intensity to avoid inter-category cannibalization.
2. **F1 priming traps MIL errors** — RCA: when MIL crime_head conf=0.32–0.69 (uncertain), priming injected wrong-category visual cues into VLM prompt, VLM followed MIL's bad signal. **Fix**: raise priming threshold from 0.2 → **0.5** in `format_priming_section`. Below 0.5 = MIL "no opinion" zone, skip priming entirely.
3. **Fix E1 over-triggering** — `CONFIDENCE_GAP_HARD` 0.4 was too sensitive; HARD count went 9 (v6) → 16 (v7). **Fix**: raise to **0.5** AND add `MIL_CONF_MIN = 0.4` guard so MIL's low-confidence noise (<0.4) doesn't trigger HARD even when VLM disagrees.

**MIL retrained with BCE loss (2026-04-25)** — per-video BCE replaces snippet-level MIL ranking loss:
- `_mil_ranking_loss` rewritten in `train_mil.py` (kept name for backward compat). Per-video BCE: anomaly→1, normal→0.
- Old MIL ranking had `sparsity = scores_a.sum()` driving all anomaly scores to ~0 (escalation_head broken since pilot_v3); BCE removes this corruption.
- Trained with `--lambda_mil 1.0` (was 0.3) so BCE weight matches CE in total loss.
- Final epoch: CE=0.751, MIL=0.0000 (escalation_head perfectly separates anomaly vs normal embeddings).
- Old weights backed up to `outputs/mil_weights_v3_milranking_<timestamp>/`.
- Note: must wrap BCE in `torch.amp.autocast("cuda", enabled=False)` because `F.binary_cross_entropy` is hard-blocked under any autocast context (bf16 incompatible). Cast inputs to float32 inside the function.
- ⚠ `ESCALATION_HEAD_TRUSTED = False` in `reflector.py` should be flipped to `True` after pilot_v9 confirms BCE-trained head behaves correctly. Currently still gated for safety; revisit after v9 validation.

### File cleanup (2026-04-25)

- Deleted obsolete shell scripts (11 one-shot queue scripts: `cleanup.sh`, `exp1_queue.sh`, `overnight.sh`, `post_pilot.sh`, `post_v7_chain.sh`, `smart_queue.sh`, `stop_rerun_launch_v5.sh`, `v6_then_exp1.sh`, `v7_after_exp1.sh`, `v9_after_train.sh`, `gemma4_int8_smoke_test.sh`).
- Active: `train_and_v9.sh` (currently running v9 chain).
- Deleted prompt-bug Experiment 1 outputs: `qwen3vl_32b_prompt_bug/`, `qwen35_35b_a3b_prompt_bug/`, empty `qwen35_35b_a3b/`, failed `experiment_1_smoke/`.
- Deleted superseded pilots: `pilot_v2/`, `pilot_v3/`, `pilot_v4/`, `pilot_v5/`. Kept v6/v7/v8/v9 (current research baselines).

### Two-stage prompt support (Experiment 1)

`scripts/test_vlm_classify.py` now supports two-stage classification:
- **Stage 1** (`BINARY_PROMPT`): yes/no anomaly detection biased toward positive (counters single-stage "default to Normal" Recall loss).
- **Stage 2** (`STAGE2_PROMPT`): 13-way crime classification only fires when Stage 1 = yes; no Normal escape.
- `STAGE2_FALLBACK_CATEGORY = "Vandalism"` if Stage 2 parse fails.
- Per-model `classify_<model>_two_stage` functions in `test_vlm_classify.py`; `experiment_1_baseline.py` `--two-stage` flag dispatches via `TWO_STAGE_MAP`. Output goes to `<model>_2stage` subdir.

### InternVL3 API fix (2026-04-25)

`load_internvl3*` was using `AutoModel` which returns the bare `InternVLModel` class (no `.generate()`). Fixed to use `AutoModelForImageTextToText`, matching the pattern used by Gemma4. The earlier Experiment 1 InternVL3-38B run produced ERROR rows for all 406 cases (now archived in `experiment_1/internvl3_38b_apibroken_<timestamp>/`); needs re-run when GPU free.

### Gemma-4-26B-A4B (MoE) — declared too slow for Experiment 1 (2026-04-25)

- INT4 NF4 + double_quant: 324s/case (BnB×MoE dequantization pathology — each token activates different expert weights, breaking weight cache).
- INT8 attempted: smoke test ran for 25+ min on a single case before being killed (NOT faster than INT4).
- Outcome: Gemma4-26B-A4B is **not usable** for two-stage Experiment 1 on this hardware/library stack. Single-stage result kept as a "safety alignment collapse" finding (predicts Normal for 100% of crime cases, Acc=0.493).
- Paper: write as model limitation — "Gemma-4-26B-A4B with BnB quantization on RTX 5090 (32 GB) cannot meet inference latency requirements; result reported as best-effort baseline".

### Pilot snapshot table (as of 2026-04-25)

| Pilot | 14-way Acc | Crime-only Acc | Binary F1 | Notes |
|---|---|---|---|---|
| v6 | **48.2%** | **42.5%** | 0.884 | Best baseline; bias_priors v3 + thresholds 0.75/0.70 |
| v7 | 33.9% | 20.0% | 0.905 | bias_priors v4 over-correct + F1 priming traps MIL errors |
| v8 | 33.9% | 20.0% | 0.905 | bias_priors v5 + GAP_HARD 0.5 + MIL_CONF_MIN — same headline as v7; per-category Δ negligible |
| v9 | (running) | — | — | + retrained MIL (BCE) + priming threshold 0.5 |
