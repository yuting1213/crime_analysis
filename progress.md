# Progress Log

## Sessions Summary (2026-04-20 ~ 2026-04-25)

### Phase 1: Stage 2 A Baseline Pilot ✅ DONE
- Built Rlegal 3-tier scoring, real VLM confidence extraction, alphabetical category prompt order
- Implemented pilot resume + atomic checkpoint
- 6 unit-test suites all green
- Final result: pilot_v3 baseline 51.8% 14-way

### Phase 2: Pilot iterations v4 → v9 (2026-04-22 ~ 2026-04-25) ✅ MOSTLY DONE
- **v4 → v5**: Fix A/B/C addressed v4 regressions (Shoplifting 75→0%, RoadAccidents 50→0%) by adjusting non-crime bias + Normal definition + Shoplifting hint
- **v6**: 48.2% 14-way / 42.5% crime-only / Binary F1 0.884 — best baseline
- **v7**: Added Fix E1/E2/E3 + F1 priming + bias_priors v4 → binary F1 0.905 ✅ but 14-way crash to 33.9% ❌
- **v8**: bias_priors v5 + GAP_HARD 0.5 + MIL_CONF_MIN 0.4 — same 33.9% headline; identified F1 priming as root cause
- **v9** (running): Retrained MIL with BCE loss + priming threshold 0.5 + bias_priors v5

### Phase 2.5: Experiment 1 — Pre-trained VLM Baseline ⚠ PARTIAL
- ✅ Qwen3-VL-32B single-stage on 406 videos: P=0.983 R=0.548 F1=0.704 Acc=0.766 14-way=0.648
- ❌ Gemma-4-26B-A4B: 100% predicts Normal (safety alignment collapse, kept as limitation)
- ❌ InternVL3-38B: API bug (used `AutoModel` instead of `AutoModelForImageTextToText`); fixed in code; needs re-run
- 🟡 Two-stage prompt support added in code (`BINARY_PROMPT` + `STAGE2_PROMPT`); not yet run
- Documentation: `outputs/experiment_1/controls.md` (variables audit + disclosure)

### Phase 3: Ablation 🟡 INFRASTRUCTURE READY, NOT RUN
- `pilot_experiment.py` supports `--no-env`, `--no-rag`, `--no-vlm`, `--no-reflector`, `--no-vlm-report`
- Will run after pilot_v9 confirms best config

### Phase 4: ~~DPO Alignment~~ ❌ DROPPED (2026-04-22)
- Reason: cross-eval (LLM-as-Judge) gives more rigorous report-quality signal without preference-pair collection cost

### Phase 5: LLM-as-Judge Cross-Evaluation 🟡 CODE READY, BLOCKED ON API KEYS
- 7-question rubric implemented in `evaluation/llm_judge.py`
- Open-book γ context builder
- Symmetric cross-eval (Qwen+Gemini reports × Claude+OpenAI judges)
- `scripts/run_cross_evaluation.py` ready; needs `ANTHROPIC_API_KEY` + `OPENAI_API_KEY`

### Phase 6: Thesis Write-up Data 🟡 PENDING
- Will populate after pilot_v9 result + (optional) Exp1 two-stage re-run + cross-eval

## Test Results
| Test | Status |
|---|---|
| Confidence extraction (10 cases, geometric mean) | ✓ |
| Ablation flags (10 combinations) | ✓ |
| Detection metrics (13 AUROC/NDCF edge cases) | ✓ |
| Anomaly gate wiring | ✓ |
| Env loader (8 cases) | ✓ |
| Pilot resume + atomic write | ✓ |
| Reflector E1 (MIL vs VLM HARD) | ✓ (5 unit cases pass) |
| Reflector E2 (RAG element match SOFT + DS calibration) | ✓ (Rcons gap 0.263 confirmed) |
| Reflector E3 (escalation guard) | ✓ |
| Priming format helper (low-conf skip) | ✓ |
| MIL BCE under bf16 autocast | ✓ (after `with autocast(enabled=False)` wrapper) |

## Error Log (active project lifetime)
| Timestamp | Error | Resolution |
|---|---|---|
| 2026-04-20 23:50 | Pilot silently killed after 10 min | Bash tool timeout; switched to `nohup + disown` |
| 2026-04-21 00:30 | Bias priors from 8B over-corrected Shooting on FT 32B | Disabled; recomputed v3 from FT diagnostic |
| 2026-04-22 | pilot_v3 escalation_head ≈ 0 for all classes | Diagnosed: per-video MIL ranking loss + sparsity term; fixed via BCE retrain (2026-04-25) |
| 2026-04-23 | Experiment 1 first run all Normal (prompt missing Normal option) | Backed up as `_prompt_bug/`; rewrote prompt with Normal |
| 2026-04-24 | pilot_v7 14-way regression | Three-pronged: bias_priors v4 over-correct, F1 priming traps MIL errors, false-HARD from low-conf MIL |
| 2026-04-25 | InternVL3-38B all ERROR rows | `AutoModel` returns bare class without `.generate`; switched to `AutoModelForImageTextToText` |
| 2026-04-25 | Gemma-4-26B-A4B INT4/INT8 too slow | BnB×MoE pathology; declared model limitation in thesis |
| 2026-04-25 | BCE crashes under bf16 autocast | Wrapped `F.binary_cross_entropy` in `torch.amp.autocast("cuda", enabled=False)` |

## Current State (2026-04-25 22:15)

- **GPU**: pilot_v9 running (PID 1679349) — ~3h to complete
- **Next blocker**: pilot_v9 result will tell us if v9 > v6 on 14-way
- **Active queue script**: `train_and_v9.sh` (training done, pilot in progress)
- **Pending decisions**:
  1. If v9 > v6: use v9 config; run pipeline on full 406 (~25h) for direct Exp1 baseline comparison
  2. If v9 ≤ v6: use v6 config; same full 406 run
  3. After full 406: launch Exp1 two-stage re-run for prompt engineering ablation
  4. Cross-eval (need API keys)

## 5-Question Reboot Check
| Question | Answer |
|---|---|
| Where am I? | pilot_v9 running; 4 pilot iterations from v6 trying to improve 14-way |
| Where am I going? | v9 done → pick best config → pipeline on full 406 → Exp1 two-stage → cross-eval → thesis |
| What's the goal? | Beat Exp1 Qwen3-VL-32B raw 14-way (64.8%) with pipeline; report binary F1 0.9+ as anomaly-detection win |
| What have I learned? | (see findings.md) MIL BCE > ranking; priming needs guard; bias priors interact non-linearly; Gemma4 unusable |
| What have I done? | 4 pilot iterations + Reflector Fix E1/E2/E3 + RAG Priming F1 + MIL BCE retrain + Exp1 controls audit + cleanup |
