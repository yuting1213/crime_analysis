"""
Unit-test the 5 ablation flags' wiring + --bias-correction.

Doesn't require GPU. Instead of instantiating a full CrimeAnalysisPipeline
(which would load BGE-M3, R3D-18, ViT, ChromaDB…), builds a minimal mock
pipeline with just the attributes the ablation branches touch and then runs
the same flag-handling code path.

Verifies:
  --no-env          → planner.agents has no "environment" key
  --no-rag          → planner.rag is None
  --no-vlm          → planner._skip_vlm_classify is True
  --no-reflector    → pipeline.reflector and planner.reflector are NullReflector
  --no-vlm-report   → planner._skip_vlm_report is True
  --bias-correction → planner._bias_corrections loaded from JSON

Also tests flag independence: setting one does not leak into others.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]  # crime_analysis/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logging.basicConfig(level=logging.WARNING)


def build_mock_pipeline():
    """Minimal pipeline with the attributes ablation inspects."""
    planner = SimpleNamespace(
        agents={"environment": object(), "action_emotion": object()},
        rag=object(),
        reflector=object(),
        _skip_vlm_classify=False,
        _skip_vlm_report=False,
        _bias_corrections=None,
    )
    return SimpleNamespace(planner=planner, reflector=planner.reflector)


def apply_flags(pipeline, flags: dict) -> None:
    """
    Replicates the ablation block inside scripts.pilot_experiment.run_pilot.
    Keep in sync if run_pilot changes!
    """
    from crime_analysis.agents.reflector import NullReflector

    if flags.get("no_reflector"):
        pipeline.reflector = NullReflector()
        pipeline.planner.reflector = NullReflector()
    if flags.get("no_rag"):
        pipeline.planner.rag = None
    if flags.get("no_env"):
        pipeline.planner.agents.pop("environment", None)
    if flags.get("no_vlm"):
        pipeline.planner._skip_vlm_classify = True
    if flags.get("no_vlm_report"):
        pipeline.planner._skip_vlm_report = True

    bias_file = flags.get("bias_correction")
    if bias_file and Path(bias_file).exists():
        pipeline.planner._bias_corrections = json.loads(
            Path(bias_file).read_text(encoding="utf-8")
        )


def assert_pristine(p) -> None:
    assert "environment" in p.planner.agents
    assert p.planner.rag is not None
    assert p.planner._skip_vlm_classify is False
    assert p.planner._skip_vlm_report is False
    assert p.planner._bias_corrections is None
    from crime_analysis.agents.reflector import NullReflector
    assert not isinstance(p.planner.reflector, NullReflector)


def run():
    from crime_analysis.agents.reflector import NullReflector

    cases = []

    # ── 1. no flags → pristine ──
    p = build_mock_pipeline()
    apply_flags(p, {})
    assert_pristine(p)
    cases.append("pristine (no flags)")

    # ── 2. --no-env ──
    p = build_mock_pipeline()
    apply_flags(p, {"no_env": True})
    assert "environment" not in p.planner.agents, "no-env failed"
    assert "action_emotion" in p.planner.agents, "no-env leaked into others"
    assert p.planner.rag is not None
    assert p.planner._skip_vlm_classify is False
    cases.append("--no-env")

    # ── 3. --no-rag ──
    p = build_mock_pipeline()
    apply_flags(p, {"no_rag": True})
    assert p.planner.rag is None, "no-rag failed"
    assert "environment" in p.planner.agents, "no-rag leaked"
    cases.append("--no-rag")

    # ── 4. --no-vlm ──
    p = build_mock_pipeline()
    apply_flags(p, {"no_vlm": True})
    assert p.planner._skip_vlm_classify is True, "no-vlm failed"
    assert p.planner._skip_vlm_report is False, "no-vlm leaked into no-vlm-report"
    cases.append("--no-vlm")

    # ── 5. --no-reflector ──
    p = build_mock_pipeline()
    apply_flags(p, {"no_reflector": True})
    assert isinstance(p.planner.reflector, NullReflector), "no-reflector planner"
    assert isinstance(p.reflector, NullReflector), "no-reflector pipeline"
    assert "environment" in p.planner.agents
    cases.append("--no-reflector")

    # ── 6. --no-vlm-report ──
    p = build_mock_pipeline()
    apply_flags(p, {"no_vlm_report": True})
    assert p.planner._skip_vlm_report is True, "no-vlm-report failed"
    assert p.planner._skip_vlm_classify is False, "no-vlm-report leaked"
    cases.append("--no-vlm-report")

    # ── 7. multiple flags combined ──
    p = build_mock_pipeline()
    apply_flags(p, {"no_env": True, "no_rag": True, "no_reflector": True})
    assert "environment" not in p.planner.agents
    assert p.planner.rag is None
    assert isinstance(p.planner.reflector, NullReflector)
    assert p.planner._skip_vlm_classify is False
    assert p.planner._skip_vlm_report is False
    cases.append("combined no-env + no-rag + no-reflector")

    # ── 8. --bias-correction loads JSON into planner ──
    p = build_mock_pipeline()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"Robbery": 1.1, "Burglary": 1.0, "Abuse": -1.0}, f)
        tmp_path = f.name
    try:
        apply_flags(p, {"bias_correction": tmp_path})
        assert p.planner._bias_corrections is not None
        assert p.planner._bias_corrections["Robbery"] == 1.1
        assert p.planner._bias_corrections["Burglary"] == 1.0
        assert p.planner._bias_corrections["Abuse"] == -1.0
    finally:
        Path(tmp_path).unlink()
    cases.append("--bias-correction load JSON")

    # ── 9. --bias-correction with missing file is a no-op ──
    p = build_mock_pipeline()
    apply_flags(p, {"bias_correction": "/nonexistent/path.json"})
    assert p.planner._bias_corrections is None
    cases.append("--bias-correction missing file → no-op")

    # ── 10. Drift-check: real run_pilot block stays in sync ──
    # Read the actual code path in pilot_experiment and confirm the ablation
    # branches we replicated still exist verbatim (fail loud if someone
    # silently adds/renames a flag without updating this test).
    pilot_src = (ROOT / "scripts" / "pilot_experiment.py").read_text()
    for snippet in (
        'flags.get("no_reflector")' if False else 'ablation_flags.get("no_reflector")',
        'ablation_flags.get("no_rag")',
        'ablation_flags.get("no_env")',
        'ablation_flags.get("no_vlm"):',
        'ablation_flags.get("no_vlm_report")',
        'ablation_flags.get("bias_correction")',
    ):
        assert snippet in pilot_src, (
            f"run_pilot no longer contains `{snippet}` — "
            "test_ablation_flags.apply_flags is now out of sync, update both."
        )
    cases.append("run_pilot drift check")

    print(f"Ran {len(cases)} ablation wiring tests:")
    for name in cases:
        print(f"  ✓ {name}")
    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    run()
