"""
Unit test for pilot_experiment resume helpers — no GPU, no model loading.

Covers:
  1. _load_resume_state on missing file → ([], set())
  2. _load_resume_state on valid JSON → stats + done_ids
  3. _load_resume_state on corrupt JSON → ([], set()), no raise
  4. _load_resume_state on JSON that's not a list → ([], set())
  5. _atomic_write_stats writes + no .tmp residue
  6. _atomic_write_stats round-trips exactly same data
  7. CLI has --resume flag wired into argparse + ablation_flags-agnostic
  8. run_pilot signature accepts resume kwarg
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from scripts.pilot_experiment import (  # noqa: E402
    _load_resume_state, _atomic_write_stats, run_pilot,
)


def run():
    cases = []

    # ── 1. missing file → empty state ──
    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "pilot_stats.json"
        s, d = _load_resume_state(stats_path)
        assert s == [] and d == set(), f"missing should return empty: {s}, {d}"
    cases.append("missing file → empty")

    # ── 2. valid JSON → correct stats + done_ids ──
    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "pilot_stats.json"
        stats_path.write_text(json.dumps([
            {"video_id": "A001", "ground_truth": "Robbery", "correct": True},
            {"video_id": "B002", "ground_truth": "Burglary", "correct": False},
            {"video_id": "C003", "ground_truth": "Assault", "correct": True},
        ], ensure_ascii=False, indent=2), encoding="utf-8")
        s, d = _load_resume_state(stats_path)
        assert len(s) == 3
        assert d == {"A001", "B002", "C003"}
    cases.append("valid JSON → 3 stats + 3 done_ids")

    # ── 3. corrupt JSON → fallback ──
    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "pilot_stats.json"
        stats_path.write_text("{definitely not json", encoding="utf-8")
        s, d = _load_resume_state(stats_path)
        assert s == [] and d == set(), "corrupt should fallback to empty"
    cases.append("corrupt JSON → empty fallback (no raise)")

    # ── 4. non-list JSON → fallback ──
    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "pilot_stats.json"
        stats_path.write_text(json.dumps({"oops": "dict not list"}), encoding="utf-8")
        s, d = _load_resume_state(stats_path)
        assert s == [] and d == set()
    cases.append("non-list JSON → empty fallback")

    # ── 5. atomic write + no .tmp residue ──
    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "pilot_stats.json"
        tmp_path = stats_path.with_suffix(".json.tmp")
        _atomic_write_stats(stats_path, [{"video_id": "A001", "x": 1}])
        assert stats_path.exists(), "stats file should exist after write"
        assert not tmp_path.exists(), "tmp file should be renamed away"
    cases.append("atomic write leaves no .tmp residue")

    # ── 6. round-trip fidelity ──
    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "pilot_stats.json"
        original = [
            {"video_id": "A001", "ground_truth": "Robbery",
             "metadata": {"nested": [1, 2, 3], "unicode": "竊盜罪"}},
            {"video_id": "B002", "ground_truth": "Normal"},
        ]
        _atomic_write_stats(stats_path, original)
        loaded, ids = _load_resume_state(stats_path)
        assert loaded == original, f"round-trip mismatch: {loaded}"
        assert ids == {"A001", "B002"}
    cases.append("round-trip: Chinese + nested data preserved")

    # ── 7. CLI flag exists ──
    src = (ROOT / "scripts" / "pilot_experiment.py").read_text(encoding="utf-8")
    assert '"--resume"' in src, "--resume CLI flag missing"
    assert "resume=args.resume" in src, "args.resume not passed to run_pilot"
    cases.append("CLI --resume flag wired")

    # ── 8. run_pilot signature accepts resume ──
    sig = inspect.signature(run_pilot)
    assert "resume" in sig.parameters, "run_pilot missing resume kwarg"
    assert sig.parameters["resume"].default is False, \
        "resume should default to False (non-intrusive)"
    cases.append("run_pilot(resume=False) default is safe")

    print(f"Ran {len(cases)} resume tests:")
    for c in cases:
        print(f"  ✓ {c}")
    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    run()
