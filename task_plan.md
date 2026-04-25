# Task Plan: UCF-Crime Forensic Report Pipeline — Thesis Experiments

## Goal
Validate and benchmark the multi-agent **MIL → Qwen3-VL-32B → Hybrid RAG → Reflector** pipeline on UCF-Crime: produce thesis-grade accuracy, anomaly detection (binary F1), and per-category breakdown; then evaluate report quality via LLM-as-Judge cross-eval.

## Current Phase
**Pilot iteration validation** — pilot_v9 running (PID 1679349, started 2026-04-25 22:09).

## Phases

### Phase 1: Stage 2A Baseline ✅ DONE
- All baseline infra: Rlegal 3-tier, real VLM confidence, alphabetical category order, resume + atomic checkpoint, anomaly gate wiring, .env loader
- 6 unit-test suites green
- pilot_v3 baseline = 51.8% 14-way

### Phase 2: Pilot Iterations v4 → v9 ✅/🟢
- v6 = best baseline (48.2% / 42.5% / 0.884)
- v7/v8 regression study (33.9%) → identified F1 priming MIL-error trap
- v9 (running) = retrained MIL BCE + priming threshold 0.5

### Phase 2.5: Experiment 1 — Pre-trained VLM Baseline ⚠ PARTIAL
- ✅ Qwen3-VL-32B single-stage (P=0.983 R=0.548 F1=0.704 Acc=0.766)
- ❌ Gemma4 (model limitation — kept in paper)
- ❌ InternVL3-38B (API bug fixed; needs re-run)
- 🟡 Two-stage prompt support (code ready, not yet run)

### Phase 3: Pipeline on Full 406 Test (NEXT) 🟡 PENDING
- Run pipeline (best config from v6/v9) on the same 406 videos used for Exp1 (206 anomaly + 200 Normal)
- Apples-to-apples comparison: pipeline vs raw VLM 14-way
- Estimated: ~20-30h GPU
- Decision rule: If pipeline 14-way > Exp1 64.8% → headline thesis result. If not → report anomaly-only accuracy as the fair comparison

### Phase 4: Ablation 🟡 PENDING (infrastructure ready)
- 4 ablation variants: `--no-env`, `--no-rag`, `--no-reflector`, `--no-vlm-report`
- 56 videos × 4 = ~10h GPU
- For thesis Chapter 4 — quantify each module's contribution

### Phase 5: LLM-as-Judge Cross-Evaluation 🟡 BLOCKED on API keys
- Symmetric Qwen×Gemini reports judged by Claude + OpenAI
- 7-question rubric + open-book γ
- 20 crime × 2 model = 40 reports for human ICC validation
- Estimated: ~3h API + 1 week human evaluation

### Phase 6: Formal Test Set ⏳ AFTER PROMPTS STABILIZE
- Held-out subset (not used for prompt tuning) for final reported metrics
- Counters "prompt overfit" reviewer concern
- Same pipeline config as full-406 run

### Phase 7: Thesis Write-up ⏳ FINAL
- Chapter 4 tables: pilot evolution / Exp1 baseline / pipeline vs raw VLM / ablation / cross-eval / human ICC
- Chapter 4 figures: confusion matrix, calibration, DET / NDCF
- Chapter 5 limitations: Gemma4 collapse, Shooting/Abuse VLM ceiling, prompt-engineering bounds

## Key Open Questions
1. **v9 vs v6** — does retrained MIL + priming threshold 0.5 reverse the 14-way drop?
2. **Pipeline on full 406** — does the +14.6pp anomaly accuracy (vs raw VLM) translate to a 14-way win after Normal proportions are matched?
3. **Two-stage prompt** — does Stage1/Stage2 split help Recall on Exp1 Qwen?
4. **Cross-eval** — does Qwen3-VL-32B + pipeline produce reports that Claude/OpenAI judge favor over Gemini baseline?
5. **Human ICC** — does ICC ≥ 0.75 hold for the 7-question rubric?

## Decisions Made
| Decision | Rationale |
|---|---|
| Drop DPO | Cross-eval gives more rigorous signal without preference-pair cost |
| Hybrid RAG (was H-RAG) | Architecturally accurate (BM25 + dense + RRF, not multi-level hierarchy) |
| Non-crime two-stage decision | Taiwan Criminal Code mismatch for Arrest/RoadAccidents/Explosion |
| BCE replaces MIL ranking loss | Per-video architecture incompatible with snippet-level MIL semantics |
| Gemma4 declared unusable | INT4 NF4 = 324s/case, INT8 not faster; documented as model limitation |
| Cross-judge symmetric (no Qwen-as-judge of self) | Avoid self-preference bias (Zheng 2023) |
| Priming guard 0.5 (was 0.2) | pilot_v8 RCA: MIL conf < 0.5 = noise traps VLM |
| `CONFIDENCE_GAP_HARD` 0.5 + `MIL_CONF_MIN` 0.4 | Suppress pilot_v7 false-HARD spike |

## Active Background Processes
- **PID 1679349** — `train_and_v9.sh` chain (training done; pilot_v9 running)

## Logs
- `/tmp/train_and_v9.log` — chain
- `/tmp/train_mil_v4.log` — MIL retrain (BCE)
- `/tmp/pilot_v9.log` — pilot_v9
- `/tmp/pilot_v9_review.txt` — final v6/v8/v9 comparison (generated when chain finishes)

## Notes
- GPU: RTX 5090 32 GB; Qwen3-VL-32B INT4 ~21 GB; BGE-M3 ~2 GB
- API keys in `crime_analysis/.env` (600 perm, gitignored)
- Pipeline rate: ~6 min/video for full pipeline; raw VLM ~60s/case
- Exp1 raw VLM Qwen3-VL-32B baseline = 64.8% 14-way (mostly Normal recognition); fair comparison must use anomaly-only accuracy (~31.6% raw vs ~46.2% pipeline = +14.6pp gain)
