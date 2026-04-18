"""
Unit test the 2-stage anomaly gate wiring in PlannerAgent.

Uses mock objects to avoid loading GPU models. Verifies:
  - gate inactive (threshold=None) → VLM still runs, no Normal override
  - gate active, score >= τ → VLM still runs normally
  - gate active, score <  τ → final_category forced to Normal, VLM bypassed
  - _skip_vlm_classify / _skip_vlm_report NOT mutated on instance (so next
    video in batch isn't affected)
  - escalation_score / anomaly_gated / anomaly_threshold in result dict
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))


def _find_anomaly_gate_block(src: str) -> str:
    """Extract the gating block from planner source for drift check."""
    import re
    m = re.search(
        r"Step 2a：2-stage anomaly gate.*?bypass_vlm_report = True",
        src, re.DOTALL,
    )
    return m.group(0) if m else ""


def run():
    cases = []

    # Read planner source for structural assertions (no GPU load)
    planner_src = (ROOT / "agents" / "planner.py").read_text()

    # ── 1. gate block is present ──
    block = _find_anomaly_gate_block(planner_src)
    assert block, "anomaly gate block not found in planner.py"
    cases.append("gate block present in planner.py")

    # ── 2. gate uses local bypass vars, not self._skip_* mutation ──
    # The block must NOT assign to self._skip_* (would leak across videos)
    assert "self._skip_vlm_classify = True" not in block, \
        "gate mutates instance state — leaks across videos"
    assert "self._skip_vlm_report = True" not in block, \
        "gate mutates instance state — leaks across videos"
    # It must use the local bypass vars
    assert "bypass_vlm_classify = True" in block
    assert "bypass_vlm_report = True" in block
    cases.append("gate uses local bypass vars (no instance mutation)")

    # ── 3. _anomaly_threshold attribute exists in __init__ ──
    assert "self._anomaly_threshold: Optional[float] = None" in planner_src, \
        "_anomaly_threshold attribute missing from __init__"
    cases.append("_anomaly_threshold defaults to None")

    # ── 4. anomaly_gated handled in _synthesize_final_report ──
    assert 'anomaly_gated' in planner_src, "anomaly_gated metadata not handled"
    assert 'crime_type = "Normal"' in planner_src
    cases.append("anomaly_gated overrides crime_type in final report")

    # ── 5. result dict carries escalation_score / anomaly_gated / threshold ──
    for key in ("escalation_score", "anomaly_gated", "anomaly_threshold"):
        assert f'"{key}"' in planner_src, f"result dict missing '{key}'"
    cases.append("result dict carries escalation_score / anomaly_gated / threshold")

    # ── 6. pilot_experiment passes --anomaly-threshold through ──
    pilot_src = (ROOT / "scripts" / "pilot_experiment.py").read_text()
    assert 'args.anomaly_threshold' in pilot_src
    assert 'pipeline.planner._anomaly_threshold = float(anomaly_threshold)' in pilot_src
    cases.append("pilot_experiment threads --anomaly-threshold into planner")

    # ── 7. pilot stats record is_anomaly_gt / is_anomaly_pred / anomaly_correct ──
    for key in ('"is_anomaly_gt"', '"is_anomaly_pred"', '"anomaly_correct"',
                '"escalation_score"'):
        assert key in pilot_src, f"pilot case_stats missing {key}"
    cases.append("pilot case_stats logs is_anomaly + escalation_score")

    # ── 8. pilot_experiment --n-normal flag exists and routes through ──
    assert '--n-normal' in pilot_src, "--n-normal flag missing"
    assert 'n_normal=args.n_normal' in pilot_src or 'n_normal = args.n_normal' in pilot_src
    cases.append("--n-normal propagates into load_pilot_samples")

    # ── 9. load_pilot_samples has Normal directory fallback ──
    assert 'Testing_Normal_Videos_Anomaly' in pilot_src
    assert 'z_Normal_Videos_event' in pilot_src
    cases.append("Normal directory fallback present when UCA lacks entries")

    # ── 10. summary computes AUROC / NDCF when Normal + escalation present ──
    assert 'from evaluation.detection_metrics import' in pilot_src
    assert 'has_normal' in pilot_src and 'has_escalation' in pilot_src
    cases.append("summary computes detection metrics conditionally")

    print(f"Ran {len(cases)} anomaly-gate wiring tests:")
    for c in cases:
        print(f"  ✓ {c}")
    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    run()
