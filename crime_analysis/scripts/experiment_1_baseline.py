"""
Experiment 1 — Pre-trained VLM Baseline Evaluation (0-shot).

Mirrors the senior thesis's "實驗一：預訓練VLM基準模型評估" using max-spec
variants that fit on a single RTX 5090 (32 GB, INT4 quantization).

Reports video-level binary anomaly detection metrics (Precision / Recall /
F1 / Accuracy) — directly comparable to the senior's Table 4-3.

Key feature: **per-case resume**. `stats.json` is written after every
single video, so an interrupted run can be continued with the same command.

Usage (full UCA Test data, 310 videos = 206 crime + 104 Normal):
    cd crime_analysis
    python -m scripts.experiment_1_baseline \\
        --models qwen3vl_32b gemma4_26b_a4b internvl3_38b \\
        --n_samples 1000 --n-normal 200 \\
        --split Test --seed 42 \\
        --output-dir outputs/experiment_1

Outputs:
    outputs/experiment_1/<model>/stats.json      per-case predictions (incremental)
    outputs/experiment_1/<model>/metrics.json    binary P/R/F1/Acc
    outputs/experiment_1/summary.{json,txt}      cross-model table
"""
import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pilot_experiment import load_pilot_samples
from scripts.test_vlm_classify import (
    extract_frames,
    parse_category,
    load_qwen3vl, classify_qwen3vl, classify_qwen3vl_two_stage, classify_qwen3vl_no_normal,
    load_qwen3vl_32b,
    load_internvl3, classify_internvl3, classify_internvl3_two_stage, classify_internvl3_no_normal,
    load_internvl3_38b,
    load_gemma4, classify_gemma4, classify_gemma4_two_stage, classify_gemma4_no_normal,
    load_gemma4_26b_a4b,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODELS = ["qwen3vl_32b", "gemma4_26b_a4b", "internvl3_38b"]

# (loader_fn, classify_fn, display_name) per model key
MODEL_REGISTRY = {
    "qwen3vl":          (load_qwen3vl,        classify_qwen3vl,  "Qwen3-VL-8B-Instruct"),
    "qwen3vl_32b":      (load_qwen3vl_32b,    classify_qwen3vl,  "Qwen3-VL-32B-Instruct (INT4)"),
    "internvl3":        (load_internvl3,      classify_internvl3,"InternVL3-8B-hf"),
    "internvl3_38b":    (load_internvl3_38b,  classify_internvl3,"InternVL3-38B-hf (INT4)"),
    "gemma4":           (load_gemma4,         classify_gemma4,   "Gemma-4-E4B-it"),
    "gemma4_26b_a4b":   (load_gemma4_26b_a4b, classify_gemma4,   "Gemma-4-26B-A4B-it (INT8)"),
}

# 1-stage classify_fn → 對應的 2-stage 版本
TWO_STAGE_MAP = {
    classify_qwen3vl:   classify_qwen3vl_two_stage,
    classify_internvl3: classify_internvl3_two_stage,
    classify_gemma4:    classify_gemma4_two_stage,
}

# 1-stage classify_fn → 對應的 no-Normal 版本（forced crime 13-way, no Normal escape）
NO_NORMAL_MAP = {
    classify_qwen3vl:   classify_qwen3vl_no_normal,
    classify_internvl3: classify_internvl3_no_normal,
    classify_gemma4:    classify_gemma4_no_normal,
}


def compute_binary_metrics(results: List[Dict]) -> Dict:
    """Video-level binary anomaly detection (gt != 'Normal' vs pred != 'Normal')."""
    tp = fp = fn = tn = 0
    for r in results:
        is_anom_gt = r["ground_truth"] != "Normal"
        is_anom_pred = r["predicted"] != "Normal"
        if is_anom_gt and is_anom_pred:       tp += 1
        elif not is_anom_gt and is_anom_pred: fp += 1
        elif is_anom_gt and not is_anom_pred: fn += 1
        else:                                 tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = (tp + tn) / n if n else 0.0
    cls_acc = sum(1 for r in results if r["correct"]) / n if n else 0.0
    return {
        "n": n,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(acc, 4),
        "class_accuracy_14way": round(cls_acc, 4),
    }


def run_model(model_name: str, samples: List[Dict], out_dir: Path,
              two_stage: bool = False, no_normal: bool = False) -> Dict:
    """Run a single model with per-case resume. Incremental stats.json writes."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats.json"
    metrics_path = out_dir / "metrics.json"

    # ── Load existing results for resume ─────────────────────
    existing: List[Dict] = []
    if stats_path.exists():
        try:
            existing = json.loads(stats_path.read_text())
            logger.info(f"[{model_name}] resume: {len(existing)} cases already done")
        except Exception as e:
            logger.warning(f"[{model_name}] stats.json broken ({e}); starting fresh")
            existing = []

    done_ids = {r["video_id"] for r in existing}
    remaining = [s for s in samples if s["video_id"] not in done_ids]

    if not remaining:
        logger.info(f"[{model_name}] all {len(existing)} samples already done, skipping model load")
        metrics = compute_binary_metrics(existing)
        metrics["model"] = model_name
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    logger.info(f"[{model_name}] {len(remaining)} samples to run (out of {len(samples)})")

    # ── Load model ────────────────────────────────────────────
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    loader_fn, classify_fn, display_name = MODEL_REGISTRY[model_name]
    if two_stage and no_normal:
        raise ValueError("--two-stage and --no-normal are mutually exclusive")
    if two_stage:
        if classify_fn not in TWO_STAGE_MAP:
            raise ValueError(f"No 2-stage variant for {model_name}")
        classify_fn = TWO_STAGE_MAP[classify_fn]
        display_name = f"{display_name} [2-stage]"
        logger.info(f"[{model_name}] using 2-stage classification")
    elif no_normal:
        if classify_fn not in NO_NORMAL_MAP:
            raise ValueError(f"No no-Normal variant for {model_name}")
        classify_fn = NO_NORMAL_MAP[classify_fn]
        display_name = f"{display_name} [no-Normal]"
        logger.info(f"[{model_name}] using no-Normal prompt (force crime 13-way)")

    t_load = time.time()
    logger.info(f"[{model_name}] loading {display_name}...")
    model, processor = loader_fn()
    logger.info(f"[{model_name}] loaded in {time.time()-t_load:.1f}s")

    # ── Process remaining samples with incremental save ───────
    results = list(existing)
    t_start = time.time()
    try:
        for i, sample in enumerate(remaining):
            vid = sample["video_id"]
            gt = sample["ground_truth"]
            t_case = time.time()

            frames = extract_frames(sample["video_path"], n=8)
            if not frames:
                logger.warning(f"[{model_name} {len(results)+1}] {vid} — no frames")
                continue

            try:
                response = classify_fn(model, processor, frames)
                predicted = parse_category(response)
            except Exception as e:
                logger.error(f"[{model_name}] {vid} FAILED: {e}")
                predicted = "Normal"
                response = f"ERROR: {e}"
                gc.collect()
                try: torch.cuda.empty_cache()
                except Exception: pass

            correct = predicted == gt
            results.append({
                "video_id": vid,
                "ground_truth": gt,
                "predicted": predicted,
                "correct": correct,
                "response": response[:500],
            })

            # Incremental checkpoint after every case
            stats_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

            elapsed = time.time() - t_case
            logger.info(
                f"[{model_name} {len(results)}/{len(samples)}] "
                f"{vid} ({gt}) → {predicted} "
                f"{'V' if correct else 'X'} ({elapsed:.1f}s)"
            )
    finally:
        # Always clean up the model
        logger.info(f"[{model_name}] cleaning up...")
        del model, processor
        gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass

    total_elapsed = time.time() - t_start
    metrics = compute_binary_metrics(results)
    metrics["model"] = model_name
    metrics["display_name"] = display_name
    metrics["elapsed_seconds"] = round(total_elapsed, 1)
    metrics["samples_run_this_session"] = len(remaining)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    logger.info(
        f"[{model_name}] DONE {len(results)} cases in {total_elapsed/60:.1f} min total\n"
        f"  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  Acc={metrics['accuracy']:.3f}  "
        f"(14-way={metrics['class_accuracy_14way']:.3f})"
    )
    return metrics


def write_summary(per_model: Dict[str, Dict], out_dir: Path):
    summary_json = out_dir / "summary.json"
    summary_txt  = out_dir / "summary.txt"
    summary_json.write_text(json.dumps(per_model, ensure_ascii=False, indent=2))

    lines = []
    lines.append("=" * 90)
    lines.append(" Experiment 1 — Pre-trained VLM Baseline (video-level binary anomaly detection)")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Model':<28} {'n':>4} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Acc':>8} {'14-way':>8}")
    lines.append("-" * 90)
    for name, m in per_model.items():
        if "precision" not in m:
            lines.append(f"{name:<28} {'ERROR':>6}  {str(m.get('error', ''))[:40]}")
            continue
        lines.append(
            f"{name:<28} {m['n']:>4} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>8.3f} {m['accuracy']:>8.3f} {m['class_accuracy_14way']:>8.3f}"
        )
    lines.append("")
    lines.append("Confusion matrices:")
    for name, m in per_model.items():
        if "confusion" in m:
            c = m["confusion"]
            lines.append(
                f"  {name:<28}  TP={c['TP']}  FP={c['FP']}  FN={c['FN']}  TN={c['TN']}"
            )
    lines.append("")
    lines.append("Reference — senior's Table 4-3 (pretrained VLM baseline):")
    lines.append("  InternVL3.5           P=0.837  R=0.869  F1=0.852  Acc=0.850")
    lines.append("  Qwen3-VL (8B)         P=0.978  R=0.715  F1=0.826  Acc=0.850")
    lines.append("  Gemma3n-E4B           P=0.500  R=1.000  F1=0.666  Acc=0.500")

    summary_txt.write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 — VLM baseline (0-shot)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model keys to run (default: {DEFAULT_MODELS})")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Upper bound on crime samples (per_cat = ceil(n/13))")
    parser.add_argument("--n-normal", dest="n_normal", type=int, default=200,
                        help="Upper bound on Normal samples")
    parser.add_argument("--split", default="Test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/experiment_1")
    parser.add_argument("--two-stage", action="store_true",
                        help="Use 2-stage classification: binary anomaly detection then 13-way category")
    parser.add_argument("--no-normal", action="store_true",
                        help="Use no-Normal prompt: force model to pick from 13 crime categories (no Normal escape). "
                             "Tests whether Normal escape hatch triggers safety-alignment collapse on Gemma/InternVL.")
    args = parser.parse_args()

    # seed just for reproducible sample order (loader uses seed internally)
    import random
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading samples (n_crime≤{args.n_samples}, n_normal≤{args.n_normal})...")
    samples = load_pilot_samples(
        n_samples=args.n_samples,
        split=args.split,
        n_normal=args.n_normal,
    )
    logger.info(f"  loaded {len(samples)} samples")

    per_model: Dict[str, Dict] = {}
    suffix = "  [2-stage]" if args.two_stage else ("  [no-Normal]" if args.no_normal else "")
    for model_name in args.models:
        logger.info(f"\n{'=' * 70}")
        logger.info(f" Running {model_name}{suffix}")
        logger.info(f"{'=' * 70}")
        # 變體結果獨立目錄，避免覆蓋既有 stats.json
        if args.two_stage:
            sub_dir = out_dir / f"{model_name}_2stage"
        elif args.no_normal:
            sub_dir = out_dir / f"{model_name}_no_normal"
        else:
            sub_dir = out_dir / model_name
        try:
            per_model[model_name] = run_model(
                model_name, samples, sub_dir,
                two_stage=args.two_stage,
                no_normal=args.no_normal,
            )
        except Exception as e:
            logger.error(f"[{model_name}] fatal error: {e}")
            per_model[model_name] = {"error": str(e)}
        # Write summary after each model in case we crash
        write_summary(per_model, out_dir)

    logger.info(f"\nAll done. Summary → {out_dir}/summary.txt")


if __name__ == "__main__":
    main()
