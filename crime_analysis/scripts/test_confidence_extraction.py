"""Unit test for PlannerAgent._confidence_from_scores — no GPU required."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]  # crime_analysis/
sys.path.insert(0, str(ROOT))  # for `from config import cfg`
sys.path.insert(0, str(ROOT.parent))  # for `from crime_analysis...`

from crime_analysis.agents.planner import PlannerAgent  # noqa: E402


class FakeTokenizer:
    """Minimal tokenizer: maps category → fixed token id sequence."""

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._mapping.get(text, [])


def _mock_scores(vocab_size: int, steps: int, target_probs: dict):
    """
    Build a list of per-step logits tensors.

    target_probs: {step_index: {token_id: probability}} — we construct logits
    so softmax yields the requested probability for that token; other tokens
    share the remaining mass uniformly.
    """
    import math

    scores = []
    for step in range(steps):
        logits = torch.full((1, vocab_size), -10.0)
        spec = target_probs.get(step, {})
        # Set one "target" high logit so softmax gives the desired prob
        if spec:
            remaining = 1.0 - sum(spec.values())
            # Uniformly spread remaining over non-target vocab
            base = remaining / (vocab_size - len(spec)) if remaining > 0 else 1e-9
            base_logit = math.log(max(base, 1e-12))
            logits[0, :] = base_logit
            for tid, p in spec.items():
                logits[0, tid] = math.log(max(p, 1e-12))
        scores.append(logits)
    return tuple(scores)


def run():
    cases = []

    # ── Case 1: High-confidence prediction ──
    # Generated sequence: [100, 10, 11] where 10,11 are "Robbery" tokens with
    # probabilities 0.9 each → geometric mean ≈ 0.9
    tok = FakeTokenizer({" Robbery": [10, 11], "Robbery": [10, 11]})
    scores = _mock_scores(
        vocab_size=50,
        steps=3,
        target_probs={1: {10: 0.9}, 2: {11: 0.9}},
    )
    generated = torch.tensor([100, 10, 11])
    conf = PlannerAgent._confidence_from_scores(scores, generated, tok, "Robbery")
    assert 0.88 < conf < 0.92, f"case1 expected ~0.9, got {conf}"
    cases.append(("high conf", conf))

    # ── Case 2: Low-confidence prediction ──
    # Target token prob = 0.3 each → geometric mean ≈ 0.3
    scores = _mock_scores(
        vocab_size=50,
        steps=3,
        target_probs={1: {10: 0.3}, 2: {11: 0.3}},
    )
    conf = PlannerAgent._confidence_from_scores(scores, generated, tok, "Robbery")
    assert 0.28 < conf < 0.32, f"case2 expected ~0.3, got {conf}"
    cases.append(("low conf", conf))

    # ── Case 3: Asymmetric probs (0.9 then 0.1) → geometric mean = 0.3 ──
    scores = _mock_scores(
        vocab_size=50,
        steps=3,
        target_probs={1: {10: 0.9}, 2: {11: 0.1}},
    )
    conf = PlannerAgent._confidence_from_scores(scores, generated, tok, "Robbery")
    assert 0.28 < conf < 0.32, f"case3 expected ~0.3, got {conf}"
    cases.append(("asymmetric", conf))

    # ── Case 4: Category not in generated sequence → fallback 0.5 ──
    tok_missing = FakeTokenizer({" Assault": [77, 88], "Assault": [77, 88]})
    conf = PlannerAgent._confidence_from_scores(scores, generated, tok_missing, "Assault")
    assert abs(conf - 0.5) < 1e-6, f"case4 expected 0.5 fallback, got {conf}"
    cases.append(("fallback missing", conf))

    # ── Case 5: Empty scores → fallback 0.5 ──
    conf = PlannerAgent._confidence_from_scores((), generated, tok, "Robbery")
    assert abs(conf - 0.5) < 1e-6, f"case5 expected 0.5, got {conf}"
    cases.append(("fallback empty scores", conf))

    # ── Case 6: Single-token category ──
    tok_single = FakeTokenizer({" Arson": [42], "Arson": [42]})
    scores = _mock_scores(vocab_size=50, steps=2, target_probs={1: {42: 0.75}})
    generated = torch.tensor([100, 42])
    conf = PlannerAgent._confidence_from_scores(scores, generated, tok_single, "Arson")
    assert 0.73 < conf < 0.77, f"case6 expected ~0.75, got {conf}"
    cases.append(("single token", conf))

    print(f"{'case':<25} {'conf':>8}")
    for name, c in cases:
        print(f"{name:<25} {c:>8.4f}")

    # ── Case 7: Diagnostics top-k margin ──
    # 3 類別共 3 個 first-token ids；在 decision step logits 上放不同機率
    tok_multi = FakeTokenizer({
        " Arson": [30], "Arson": [30],
        " Robbery": [40], "Robbery": [40],
        " Burglary": [50], "Burglary": [50],
    })
    # decision pos=1：Arson=0.6, Robbery=0.3, Burglary=0.1
    scores = _mock_scores(
        vocab_size=100, steps=2,
        target_probs={1: {30: 0.6, 40: 0.3, 50: 0.1}},
    )
    generated = torch.tensor([100, 30])  # 選到 Arson
    diag = PlannerAgent._classify_diagnostics(
        scores, generated, tok_multi, "Arson", ["Arson", "Robbery", "Burglary"]
    )
    # softmax 重新 normalise 在 3 類上時，Arson 比例提高
    assert diag["top1_prob"] > diag["top2_prob"] > diag["top2_prob"] - 1, \
        f"case7 ranking broken: {diag}"
    assert diag["margin"] > 0, f"case7 margin non-positive: {diag}"
    assert diag["top3"][0][0] == "Arson", f"case7 top1 should be Arson: {diag}"
    print(f"\n-- top-k diagnostics --")
    print(f"  confidence = {diag['confidence']:.3f}")
    print(f"  top1_prob  = {diag['top1_prob']:.3f}  ({diag['top3'][0][0]})")
    print(f"  top2_prob  = {diag['top2_prob']:.3f}  ({diag['top3'][1][0]})")
    print(f"  margin     = {diag['margin']:.3f}")
    print(f"  entropy    = {diag['entropy']:.3f}")
    print(f"  top3       = {diag['top3']}")

    # ── Case 8: Diagnostics fallback when category not in generated ──
    diag_fb = PlannerAgent._classify_diagnostics(
        scores, generated, tok_multi, "Burglary", ["Arson", "Robbery", "Burglary"]
    )
    assert diag_fb["top1_prob"] == 0.0 and diag_fb["margin"] == 0.0, \
        f"case8 should fallback: {diag_fb}"

    # ── Case 9: Bias correction flips over-predicted class ──
    # Raw top-1 = Robbery (0.6); bias adds log-prior ~+1.5 on Robbery and ~-1.0
    # on Arson, so after correction the ordering flips to Arson.
    scores = _mock_scores(
        vocab_size=100, steps=2,
        target_probs={1: {30: 0.3, 40: 0.6, 50: 0.1}},  # Arson=0.3, Rob=0.6, Burg=0.1
    )
    generated_rob = torch.tensor([100, 40])  # greedy picks Robbery
    bias = {"Arson": -1.0, "Robbery": 1.5, "Burglary": 0.0}
    diag_bias = PlannerAgent._classify_diagnostics(
        scores, generated_rob, tok_multi, "Robbery",
        ["Arson", "Robbery", "Burglary"], bias_corrections=bias,
    )
    # Raw top-1 should still be Robbery
    assert diag_bias["top3"][0][0] == "Robbery", \
        f"case9 raw top1 should stay Robbery: {diag_bias['top3']}"
    # But corrected top-1 should flip to Arson (bias subtracted penalises Robbery)
    assert diag_bias["corrected_top1"][0] == "Arson", \
        f"case9 corrected top1 should flip to Arson: {diag_bias['corrected_top1']}"
    print(f"\n-- bias correction flip --")
    print(f"  raw top3       = {diag_bias['top3']}")
    print(f"  corrected top1 = {diag_bias['corrected_top1']}")
    print(f"  corrected all  = {diag_bias['corrected_all']}")

    # ── Case 10: Bias correction with zero vector is a no-op ──
    zero_bias = {"Arson": 0.0, "Robbery": 0.0, "Burglary": 0.0}
    diag_zero = PlannerAgent._classify_diagnostics(
        scores, generated_rob, tok_multi, "Robbery",
        ["Arson", "Robbery", "Burglary"], bias_corrections=zero_bias,
    )
    # corrected top-1 should match raw top-1
    assert diag_zero["corrected_top1"][0] == diag_zero["top3"][0][0], \
        f"case10 zero-bias should be identity: {diag_zero}"

    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    run()
