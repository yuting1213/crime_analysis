# Findings & Decisions — UCF-Crime Pipeline

## Architecture Summary

- **Qwen3-VL-32B-Instruct + QLoRA adapter** as unified VLM (classification + report generation)
- **MIL Head** (R3D-18 + ViT + Pose + Emotion) 4-modality pre-screen → 13 crime classes + escalation score
- **Hybrid RAG** (renamed from H-RAG): BM25 + BGE-M3 + RRF over Taiwan Criminal Code (33 articles)
- **Reflector CASAM** 3-layer audit (NONE / SOFT / HARD) with Dempster-Shafer Rcons
- **Non-crime two-stage decision**: Arrest / RoadAccidents / Explosion short-circuit (no H-RAG, no Reflector)
- Local RTX 5090 (32 GB), all VLMs INT4 NF4 except Gemma-4-26B-A4B (INT8)

## Pilot Snapshot (2026-04-25)

| Pilot | 14-way | Crime-only | Binary F1 | Highlights |
|---|---|---|---|---|
| v6 | **48.2%** | **42.5%** | 0.884 | Best baseline; bias_priors v3 |
| v7 | 33.9% | 20.0% | 0.905 | bias_priors v4 over-correct + F1 priming traps MIL errors |
| v8 | 33.9% | 20.0% | 0.905 | bias_priors v5 conservative rollback |
| v9 | (running) | — | — | Retrained MIL (BCE) + priming threshold 0.5 |

## Key Findings

### 1. Pipeline beats raw VLM on anomaly classification (not just headline 14-way)
- Exp1 Qwen3-VL-32B raw 14-way = 64.8% on 406 videos — but most credit goes to "easy Normal"
- Pipeline pilot_v6 14-way = 48.2% on 56 videos with only 4 Normal — apples-to-apples (anomaly-only) gives **+14.6pp pipeline gain (46.2% vs 31.6%)**
- Caveat: 14-way overall depends on Normal proportion; thesis must report **anomaly accuracy** (= excluding Normal class) as the fair pipeline-vs-VLM comparison

### 2. F1 RAG priming traps MIL errors when MIL conf is mid-range
- Pilot_v7 RCA: MIL crime_head conf 0.32–0.69 produces wrong predictions; priming injected wrong-category visual cues into VLM, VLM followed
- Cured by: priming guard `mil_top1_conf >= 0.5` (was 0.2) — below 0.5 = MIL "no opinion", skip priming entirely

### 3. MIL ranking loss broke escalation_head (per-video architecture mismatch)
- Original Sultani 2018 / Elmetwally 2025 MIL ranking loss assumes snippet-level features (32 snippets / video)
- Our setup: ONE 1386D feature per video → `sparsity = scores_a.sum()` drove all anomaly scores to ~0
- Fixed: `_mil_ranking_loss` now per-video BCE; lambda_mil bumped 0.3→1.0; `escalation_head` learns properly (final epoch BCE=0.0000)

### 4. Bias priors over-correction is a sharp edge
- v4 (2026-04-24) flipped Shooting/Burglary from suppress→boost, deepened Fighting/Arson — Robbery/Shoplifting/Stealing/Vandalism collapsed because boosted neighbors stole their cases
- v5 (2026-04-25) conservative half-magnitude rollback + lowered Robbery/Stealing suppress for relative balance
- Lesson: bias priors interact non-linearly across categories; single-class tuning can cascade

### 5. Gemma-4-26B-A4B is unusable on this stack for forensic CCTV
- INT4 NF4: 324s/case (BnB×MoE dequantization pathology)
- INT8: 25+ min/case in smoke (NOT faster, possibly slower)
- Single-stage classification result: 100% predict Normal for all 206 anomaly videos (safety alignment collapse)
- Limitation paragraph in thesis, not a failure of methodology

### 6. InternVL3-38B requires `AutoModelForImageTextToText`, not `AutoModel`
- Earlier Exp1 InternVL3 result was all ERROR rows (`InternVLModel` lacks `.generate`)
- Fixed; needs re-run when GPU free
- Archived broken stats at `outputs/experiment_1/internvl3_38b_apibroken_<timestamp>/`

## Technical Decisions

| Decision | Rationale |
|---|---|
| FT Qwen3-VL-32B over 8B | +27 pts accuracy on 52-video pilot (30.8→59.6%) |
| Drop DPO | Decided 2026-04-22; cross-eval (LLM-as-Judge) replaces it for thesis evaluation |
| Hybrid RAG naming (was "Hierarchical") | Architecturally accurate per Lewis 2020 / Karpukhin 2020 / Cormack 2009 |
| Non-crime short-circuit (Arrest/RoadAccidents/Explosion) | Taiwan Criminal Code mismatch; descriptive-only output, no H-RAG/Rlegal |
| BCE replaces MIL ranking loss | Per-video architecture incompatible with snippet-level MIL semantics |
| Priming guard 0.2→0.5 | pilot_v8 RCA showed MIL conf < 0.5 contains noise, not signal |
| `CONFIDENCE_GAP_HARD` 0.4→0.5 + `MIL_CONF_MIN=0.4` | Suppresses pilot_v7's false-HARD spike (9→16 conflicts) |
| Gemma4 INT8 disclosure in `controls.md` | Quantization differs from Qwen/InternVL; honest framing of "max-spec per family on 32GB" |

## LLM-as-a-Judge Cross-Evaluation Design (replaces DPO)
- 7-question rubric (Q1 法條合理性 / Q2 要件覆蓋率 / Q3 幀號可追溯 / Q4 因果鏈 / Q5 不確定性 / Q6 司法語言 / Q7 情景吻合度)
- Open-book γ: judge sees UCA scenario + cited 刑法 articles (extracted from report via regex)
- Cross-judge symmetric: Qwen and Gemini reports both judged by Claude + OpenAI
- Human ICC validation via stratified sample (20 crime × 2 model = 40 reports)
- Blocked on `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`

## Resources
- Repo: https://github.com/yuting1213/crime_analysis (branch master)
- FT adapter: `outputs/vlm_finetune/adapter_model/` (163 MB)
- New BCE-trained MIL: `outputs/mil_weights/`; backup of MIL ranking weights at `outputs/mil_weights_v3_milranking_<ts>/`
- Bias priors v5: `data/bias_priors_qwen3vl32b_ft.json`
- Cross-eval scripts: `scripts/run_cross_evaluation.py`, `sample_for_human_eval.py`, `compute_human_llm_correlation.py`
- API keys in `crime_analysis/.env` (600 perm, gitignored)
- Plan file: `/home/yuting/.claude/plans/ucf-polished-nova.md`
