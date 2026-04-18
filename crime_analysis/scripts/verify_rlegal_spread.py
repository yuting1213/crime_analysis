"""
驗證新版 compute_rlegal 在既有 pilot reports 上的分布。

不吃 GPU，只讀 outputs/pilot_reports/*.txt 與 outputs/pilot_stats.json，
重新計算 Rlegal 並輸出新舊對比表。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from crime_analysis.rag.rag_module import RAGModule  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "pilot_reports"
STATS_PATH = ROOT / "outputs" / "pilot_stats.json"


def extract_report_body(path: Path) -> str:
    """Strip the header block before the generated report."""
    text = path.read_text(encoding="utf-8")
    # 報告本文從「一、鑑定報告」開始；之前是 meta header
    idx = text.find("一、鑑定報告")
    return text[idx:] if idx >= 0 else text


def main() -> int:
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    rag = RAGModule(rag_system=None)  # Rlegal 不需要真的 RAG backend

    header = (
        f"{'video_id':<28} {'GT':<14} {'PRED':<14} "
        f"{'old':>6} {'new_p':>6} {'new_g':>6} "
        f"{'t1p':>5} {'t2p':>5} {'t3p':>5} {'correct':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for case in stats:
        vid = case["video_id"]
        gt = case["ground_truth"]
        pred = case["predicted"]
        old_rlegal = case["rlegal"]
        correct = case["correct"]

        report_path = REPORT_DIR / f"{vid}.txt"
        if not report_path.exists():
            print(f"[SKIP] {vid}: report file missing")
            continue

        body = extract_report_body(report_path)
        new_rlegal_pred = rag.compute_rlegal(pred, body)
        new_rlegal_gt = rag.compute_rlegal(gt, body)

        tiers_p = _compute_tiers(rag, pred, body)

        print(
            f"{vid:<28} {gt:<14} {pred:<14} "
            f"{old_rlegal:>6.3f} {new_rlegal_pred:>6.3f} {new_rlegal_gt:>6.3f} "
            f"{tiers_p[0]:>5.2f} {tiers_p[1]:>5.2f} {tiers_p[2]:>5.2f} "
            f"{'✓' if correct else '✗':>8}"
        )
        rows.append((old_rlegal, new_rlegal_pred, new_rlegal_gt, correct))

    if rows:
        olds = [r[0] for r in rows]
        news_p = [r[1] for r in rows]
        news_g = [r[2] for r in rows]
        correct_mask = [r[3] for r in rows]
        correct_g = [g for g, c in zip(news_g, correct_mask) if c]
        wrong_g = [g for g, c in zip(news_g, correct_mask) if not c]
        print("-" * len(header))
        print(
            f"old (pred)   mean={mean(olds):.3f}  stdev={stdev(olds):.3f}  "
            f"min={min(olds):.3f}  max={max(olds):.3f}"
        )
        print(
            f"new (pred)   mean={mean(news_p):.3f}  stdev={stdev(news_p):.3f}  "
            f"min={min(news_p):.3f}  max={max(news_p):.3f}"
        )
        print(
            f"new (GT)     mean={mean(news_g):.3f}  stdev={stdev(news_g):.3f}  "
            f"min={min(news_g):.3f}  max={max(news_g):.3f}"
        )
        if correct_g and wrong_g:
            print(
                f"  └── correct (n={len(correct_g)}): mean={mean(correct_g):.3f}   "
                f"wrong (n={len(wrong_g)}): mean={mean(wrong_g):.3f}   "
                f"gap={mean(correct_g) - mean(wrong_g):.3f}"
            )
    return 0


def _compute_tiers(rag: RAGModule, crime_type: str, report_text: str):
    """Re-derive (tier1, tier2, tier3) for inspection; mirrors compute_rlegal."""
    from crime_analysis.rag.rag_module import GROUP_LEGAL_CONTEXT, LEGAL_ELEMENTS

    expected_refs = GROUP_LEGAL_CONTEXT.get(crime_type, [])
    elements = LEGAL_ELEMENTS.get(crime_type, [])

    cited = set(re.findall(r'第\s*(\d+(?:-\d+)?)\s*條', report_text))
    expected = set()
    for ref in expected_refs:
        expected.update(re.findall(r'第(\d+(?:-\d+)?)條', ref))

    if expected and cited:
        tp = len(cited & expected)
        p = tp / len(cited)
        r = tp / len(expected)
        t1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        t1 = 0.0

    if elements:
        neg = ("不該當", "不構成", "不成立", "未該當", "未構成",
               "無法滿足", "無法成立", "未具備", "欠缺",
               "不具有", "不具備")
        window = 160
        covered = 0
        for el in elements:
            pos = 0
            found = False
            while True:
                idx = report_text.find(el, pos)
                if idx < 0:
                    break
                ctx = report_text[idx: idx + len(el) + window]
                if not any(n in ctx for n in neg):
                    found = True
                    break
                pos = idx + len(el)
            if found:
                covered += 1
        t2 = covered / len(elements)
        gneg = any(
            f"{n}本{s}" in report_text or f"本{s}{n}" in report_text
            for n in ("不構成", "不成立", "不該當", "無法適用")
            for s in ("罪", "案", "件")
        )
        if gneg:
            t2 *= 0.3
    else:
        t2 = 0.0

    other = set()
    for ctype, refs in GROUP_LEGAL_CONTEXT.items():
        if ctype == crime_type:
            continue
        for ref in refs:
            other.update(re.findall(r'第(\d+(?:-\d+)?)條', ref))
    other -= expected
    fc = len(cited & other)
    t3 = max(0.0, 1.0 - fc / 3.0)
    return t1, t2, t3


if __name__ == "__main__":
    raise SystemExit(main())
