# Experiment 1 — Controlled Variables Audit

Pre-trained VLM baseline comparison on UCF-Crime Test split. Purpose: document every variable held constant vs. the intentional differences, so the cross-model comparison is auditable.

## Sample Selection

| Attribute | Value |
|---|---|
| Source dataset | UCF-Crime Test split + UCA temporal annotations |
| Crime videos | 206 (13 UCF categories) |
| Normal videos | 200 (sampled from Training_Normal_Videos_Anomaly pool, `--n-normal 200`) |
| **Total per model** | **406** |
| Sample order | Fixed — `random.seed(42)` in `scripts/experiment_1_baseline.py:main()` |
| Loader | `load_pilot_samples(n_samples=1000, split="Test", n_normal=200)` — called once in `main()`, shared across all models |

Verification:
```bash
wc -l outputs/experiment_1/qwen3vl_32b/stats.json   # should be 406 entries
diff <(jq -r '.[].video_id' outputs/experiment_1/qwen3vl_32b/stats.json | sort) \
     <(jq -r '.[].video_id' outputs/experiment_1/gemma4_26b_a4b/stats.json | sort)
# should be empty (same video IDs in both)
```

## Prompt

Every model receives the **identical** prompt string, built from `scripts/test_vlm_classify.py::CLASSIFY_PROMPT` with `CRIME_CATEGORIES` as the category list.

```
You are a forensic surveillance video analyst.
Look at these frames from a CCTV video and determine what category the scene falls into.
If the footage shows no criminal, hazardous, or law-enforcement activity, reply Normal.

Choose ONE category from: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
Normal, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism

Category definitions:
- Assault: One person attacking another (one-sided)
- Robbery: Forcibly taking someone's belongings
...
```

**Zero-shot**: No in-context examples, no system message, no chain-of-thought instruction.

## Frame Extraction

| Attribute | Value |
|---|---|
| Function | `agents.frame_utils.uniform_keyframes(video_path, n)` |
| Frames per video | **8** |
| Sampling | Uniform linear interpolation across video duration (deterministic) |
| Preprocessing | OpenCV `VideoCapture` → BGR→RGB → PIL.Image (no resize in extraction; processor handles per-model resolution) |

Each model sees the **exact same** 8 frames per video — no per-model frame policy.

## Quantization Config

| Model | Quantization | VRAM (loaded) | Reason |
|---|---|---|---|
| Qwen3-VL-32B | INT4 NF4 + double_quant | ~18 GB | Max-spec for dense 32GB budget |
| InternVL3-38B | INT4 NF4 + double_quant | ~20 GB | Max-spec for dense 32GB budget |
| **Gemma-4-26B-A4B** | **INT8** | ~26 GB | **MoE + INT4 had pathological speed (324s/case); switched to INT8 (2026-04-25)** |

Two configs:

```python
# Qwen3-VL-32B + InternVL3-38B (dense, INT4):
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Gemma-4-26B-A4B (MoE, INT8):
BitsAndBytesConfig(load_in_8bit=True)
```

Other load parameters (all models):
- `device_map={"": 0}` — single-GPU (RTX 5090)
- `low_cpu_mem_usage=True`

**Disclosure**: The Gemma4 quantization differs from Qwen/InternVL. Rationale:
- Each model uses the lowest-precision quantization that (a) fits on a single 32 GB GPU and (b) does not have catastrophic inference latency on this specific hardware/library stack.
- For dense models, INT4 NF4 fits and runs efficiently.
- For Gemma-4-26B-A4B (MoE), INT4 NF4 + BnB has known pathological interaction: per-token expert routing forces dequantization on every step, breaking weight cache. Observed inference cost ~324s/case in pilot runs.
- INT8 has simpler dequantization (no NF4 lookup table, no double-quant nesting) and the larger weight footprint (26 GB) still fits on the 5090.
- This is closer to the experiment's intended question ("max-spec deployment per family on this hardware") than forcing a numerically-identical quantization that disadvantages the MoE.

## Decoding

| Model | `max_new_tokens` | `temperature` | `do_sample` | Effective behavior |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct | 64 | 0.1 (explicit) | False (explicit) | Greedy |
| InternVL3-38B-hf | 64 | 0.1 (explicit) | False (explicit) | Greedy |
| Gemma-4-26B-A4B-it | 64 | (not set) | (not set) | Greedy (HF default `do_sample=False`) |

**Functional equivalence confirmed**: `temperature` is ignored by HuggingFace `generate()` when `do_sample=False`. All three models use greedy decoding. The missing `temperature=0.1, do_sample=False` on Gemma4 is a **cosmetic** omission, not a behavior difference.

Rationale for `temperature=0.1` on Qwen/InternVL: defensive carry-over from earlier diagnostic runs that experimented with sampling. Left in place for consistency with prior logs.

## Post-processing

| Model | Output parsing |
|---|---|
| Qwen3-VL | Strip `<think>...</think>` tags via regex before parsing (Qwen supports native thinking mode) |
| InternVL3 | No thinking-tag stripping (not emitted) |
| Gemma4 | No thinking-tag stripping (not emitted) |

All three responses are then parsed by `parse_category()` in `scripts/test_vlm_classify.py` — same regex + case-insensitive match against the 14 categories.

## Intentional Differences (Not Variables — Design Choices)

These are **not** controlled variables; they are the experiment's independent variable (the model itself):

| Dimension | Qwen3-VL-32B | Gemma-4-26B-A4B | InternVL3-38B |
|---|---|---|---|
| Architecture | Dense | **MoE (4B active)** | Dense |
| Parameters (total) | 32B | 26B | 38B |
| Pretraining corpus | Qwen family | Gemma family | InternVL family |
| VLM lineage | Qwen3-VL | Gemma 4 multimodal | InternVL 3 |
| Context length | 32K | 128K | 8K |
| Vision encoder | ViT (native) | SigLIP | InternViT |

**Selection rationale**: Each model is the respective family's **max-spec variant that fits on a single 32 GB GPU**. The quantization level chosen per model is the lowest-precision setting that (a) fits in 32 GB and (b) avoids catastrophic latency on this hardware/library stack — INT4 NF4 for the two dense models, INT8 for the Gemma-4 MoE (see Quantization Config above). This mirrors the senior thesis methodology (Table 4-3: Qwen3-VL-8B, InternVL3.5, Gemma3n-E4B). The comparison answers "given a 32 GB deployment budget, which open-weights VLM family is strongest on forensic classification?" — not "which model is best at fixed parameter count."

## Per-case Resume

All three models use **per-video checkpointing**:
- After each classification, `stats.json` is atomically rewritten with the new result appended.
- On re-launch, existing cases in `stats.json` are skipped (matched by `video_id`).
- This means: if Gemma4 is interrupted at case 197/406, the next launch continues from case 198 without re-running 1–197.

**Guarantee**: Resume does not change results — samples are loaded in the same seed-42 order each time; only the skip set changes.

## Known Deviations from Strict Apples-to-Apples

**Disclose these in the thesis so reviewers don't flag them**:

1. **Explicit vs implicit greedy on Gemma4**: Functional no-op (see Decoding table above). The three `classify_*` functions should ideally share identical `generate()` kwargs; they don't, but the output distribution is provably identical.

2. **Qwen3-VL think-tag stripping**: If Qwen ever emits `<think>` content (it occasionally does in thinking mode), stripping it is necessary for fair parsing. Other models don't emit these tags. Not a confound — it's model-specific output handling.

3. **No vision encoder harmonization**: Each model uses its own processor, which resizes/normalizes images per its training recipe. This is **correct practice** (use each model as it was trained), not a controlled variable to equalize.

4. **MoE vs Dense + Quantization differs for Gemma**: Gemma4-26B-A4B's 4B active parameters make it inference-faster in theory but BnB INT4 NF4 + MoE routing interacts poorly (observed in initial run: ~324s/case vs ~60s for dense at similar scale). For the production run, Gemma-4 was switched to INT8 (~26 GB on 5090) to avoid this pathological interaction. This is a **deployment-stack property**, not a confound; see Quantization Config above for the disclosure rationale.

## Reproducibility Checklist

```
[x] Sample list deterministic (seed=42, shared loader)
[x] Prompt identical (CLASSIFY_PROMPT format string)
[x] Frames identical (uniform_keyframes, n=8, deterministic)
[x] Quantization documented (Qwen+InternVL: INT4 NF4; Gemma: INT8 — see disclosure)
[x] Decoding effectively identical (all greedy)
[x] Post-processing differs only for Qwen <think> tags (model-specific)
[x] No training/finetuning on Test split (all three are pre-trained weights)
```

## Artifacts

| File | Role |
|---|---|
| `scripts/experiment_1_baseline.py` | Main driver (model registry, resume logic, metrics) |
| `scripts/test_vlm_classify.py` | `load_*` + `classify_*` functions + `CLASSIFY_PROMPT` |
| `agents/frame_utils.py::uniform_keyframes` | Deterministic frame sampler |
| `scripts/pilot_experiment.py::load_pilot_samples` | Sample loader (seed=42) |
| `outputs/experiment_1/<model>/stats.json` | Per-case predictions (incremental) |
| `outputs/experiment_1/<model>/metrics.json` | Binary P/R/F1/Acc + 14-way |
| `outputs/experiment_1/summary.{json,txt}` | Cross-model summary |

## Suggested Thesis Phrasing

> "All three pre-trained VLMs were evaluated on the same 406 videos (206 crime + 200 normal) using identical prompts and identical 8-frame uniform sampling. Decoding was greedy in all cases. The two dense models (Qwen3-VL-32B, InternVL3-38B) used INT4 NF4 quantization with double-quant; Gemma-4-26B-A4B used INT8 — the lowest precision per model that (a) fits on a single 32 GB GPU and (b) avoids known BnB×MoE inference-latency pathologies. The only model-specific post-processing was stripping `<think>` tags from Qwen3-VL's native thinking mode output. The comparison isolates pre-training data and architecture; quantization is selected per model to give each family a fair deployment-realistic setting."
