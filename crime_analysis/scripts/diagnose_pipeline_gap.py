"""
Diagnose whether pipeline wrapping causes prediction drift vs the underlying
VLM standalone diagnostic on the same videos.

Uses existing output JSONs (no GPU required):
  outputs/pilot_stats.json                      — pipeline run (13 videos)
  outputs/vlm_diagnostic_qwen3vl.json           — Qwen3-VL-8B standalone (52)
  outputs/vlm_diagnostic_qwen3vl32b_finetuned.json — Qwen3-VL-32B FT (52)
  outputs/vlm_diagnostic_qwen3vl32b_int4.json   — Qwen3-VL-32B zero-shot (52)
  outputs/experiments/gemini_baseline/pilot_stats.json — Gemini (52)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def index_by_video(entries: list) -> dict:
    return {e["video_id"]: e for e in entries}


def main() -> int:
    pipeline = index_by_video(load(OUT / "pilot_stats.json"))
    vlm_8b = index_by_video(load(OUT / "vlm_diagnostic_qwen3vl.json"))
    vlm_32b_ft = index_by_video(load(OUT / "vlm_diagnostic_qwen3vl32b_finetuned.json"))
    vlm_32b_zs = index_by_video(load(OUT / "vlm_diagnostic_qwen3vl32b_int4.json"))
    gemini = index_by_video(load(OUT / "experiments" / "gemini_baseline" / "pilot_stats.json"))

    shared = sorted(set(pipeline) & set(vlm_8b) & set(vlm_32b_ft))
    print(f"Pipeline videos: {len(pipeline)}   overlap with standalone diagnostics: {len(shared)}\n")

    # ── Per-video prediction comparison ──
    header = (
        f"{'video_id':<28} {'GT':<14} "
        f"{'PIPELINE':<14} {'8B_SA':<14} {'32B_ZS':<14} {'32B_FT':<14} {'GEMINI':<14} "
        f"{'drift':>6}"
    )
    print(header)
    print("-" * len(header))

    drift_count = 0
    drift_videos = []
    gain_vs_pipeline = {"8B_SA": 0, "32B_ZS": 0, "32B_FT": 0, "GEMINI": 0}
    loss_vs_pipeline = {"8B_SA": 0, "32B_ZS": 0, "32B_FT": 0, "GEMINI": 0}

    for vid in shared:
        p = pipeline[vid]
        s8 = vlm_8b[vid]
        zs = vlm_32b_zs[vid]
        ft = vlm_32b_ft[vid]
        gm = gemini.get(vid, {})
        gt = p["ground_truth"]
        pipe_pred = p["predicted"]
        drift = "YES" if pipe_pred != s8["predicted"] else "."
        if drift == "YES":
            drift_count += 1
            drift_videos.append((vid, pipe_pred, s8["predicted"]))

        print(
            f"{vid:<28} {gt:<14} "
            f"{pipe_pred + (' ✓' if p['correct'] else ' ✗'):<14} "
            f"{s8['predicted'] + (' ✓' if s8['correct'] else ' ✗'):<14} "
            f"{zs['predicted'] + (' ✓' if zs['correct'] else ' ✗'):<14} "
            f"{ft['predicted'] + (' ✓' if ft['correct'] else ' ✗'):<14} "
            f"{gm.get('predicted', '?') + (' ✓' if gm.get('correct') else ' ✗'):<14} "
            f"{drift:>6}"
        )

        for key, src in (("8B_SA", s8), ("32B_ZS", zs), ("32B_FT", ft), ("GEMINI", gm)):
            if not src:
                continue
            if src.get("correct") and not p["correct"]:
                gain_vs_pipeline[key] += 1
            elif not src.get("correct") and p["correct"]:
                loss_vs_pipeline[key] += 1

    print("-" * len(header))

    # ── Accuracy summary on the shared 13 videos ──
    def acc(idx: dict, keys: list) -> float:
        correct = sum(1 for k in keys if idx[k]["correct"])
        return correct / len(keys) if keys else 0.0

    print("\n=== Accuracy on the same 13 videos ===")
    print(f"Pipeline (8B wrapper)       : {acc(pipeline, shared):>6.1%}  "
          f"({sum(1 for k in shared if pipeline[k]['correct'])}/{len(shared)})")
    print(f"Qwen3-VL-8B standalone      : {acc(vlm_8b, shared):>6.1%}  "
          f"({sum(1 for k in shared if vlm_8b[k]['correct'])}/{len(shared)})")
    print(f"Qwen3-VL-32B INT4 zero-shot : {acc(vlm_32b_zs, shared):>6.1%}  "
          f"({sum(1 for k in shared if vlm_32b_zs[k]['correct'])}/{len(shared)})")
    print(f"Qwen3-VL-32B FT             : {acc(vlm_32b_ft, shared):>6.1%}  "
          f"({sum(1 for k in shared if vlm_32b_ft[k]['correct'])}/{len(shared)})")
    gm_shared = [k for k in shared if k in gemini]
    print(f"Gemini 2.0 Flash            : {acc(gemini, gm_shared):>6.1%}  "
          f"({sum(1 for k in gm_shared if gemini[k]['correct'])}/{len(gm_shared)})")

    # ── Drift analysis ──
    print(f"\n=== Pipeline vs 8B standalone drift ===")
    print(f"Videos where pipeline ≠ 8B standalone: {drift_count}/{len(shared)}")
    for vid, pp, sp in drift_videos:
        gt = pipeline[vid]["ground_truth"]
        print(f"  {vid:<25} GT={gt:<14} pipe={pp:<14} 8B_SA={sp}")

    # ── Upside if pipeline swapped to 32B FT ──
    print("\n=== Counterfactual: if pipeline used X as its VLM ===")
    for key in ("8B_SA", "32B_ZS", "32B_FT", "GEMINI"):
        g = gain_vs_pipeline[key]
        l = loss_vs_pipeline[key]
        print(f"  {key:<10} would GAIN {g:>2}   LOSE {l:>2}   net {g - l:+d}")

    # ── Per-category breakdown on 13 ──
    print("\n=== Per-category (n=13) ===")
    per_cat = defaultdict(lambda: defaultdict(list))
    for vid in shared:
        cat = pipeline[vid]["ground_truth"]
        per_cat[cat]["pipe"].append(pipeline[vid]["correct"])
        per_cat[cat]["8B"].append(vlm_8b[vid]["correct"])
        per_cat[cat]["32B_FT"].append(vlm_32b_ft[vid]["correct"])
        if vid in gemini:
            per_cat[cat]["gem"].append(gemini[vid]["correct"])
    print(f"{'category':<16} {'pipe':>5} {'8B_SA':>6} {'32B_FT':>7} {'gem':>5}")
    for cat in sorted(per_cat):
        d = per_cat[cat]
        def _r(v):
            return f"{sum(v)}/{len(v)}" if v else "-"
        print(f"{cat:<16} {_r(d['pipe']):>5} {_r(d['8B']):>6} "
              f"{_r(d['32B_FT']):>7} {_r(d['gem']):>5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
