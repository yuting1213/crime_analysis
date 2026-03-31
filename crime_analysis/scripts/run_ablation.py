"""
消融實驗自動化腳本

五個變體：
  1. full_system       — 完整系統
  2. no_environment    — 移除 Environment Agent
  3. no_rag            — 移除 H-RAG（Semantic Agent 無法條檢索）
  4. no_ae_backbone    — 只用 MIL Head（移除 R3D-18 + ViT backbone）
  5. no_reflector      — NullReflector（跳過 CASAM 審查）

使用方式：
  python -m scripts.run_ablation --ucf_dir ./data/ucf_crime --n_samples 100

產出：
  results/ablation/
    ├── full_system.json
    ├── no_environment.json
    ├── no_rag.json
    ├── no_ae_backbone.json
    ├── no_reflector.json
    └── comparison_table.json   ← 所有變體的指標彙整
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── NullReflector ────────────────────────────────────────────

class NullReflector:
    """
    跳過所有 CASAM 審查的 Reflector。
    永遠回傳 NONE conflict + Rcons = 1.0。
    """

    def reset(self):
        pass

    def audit(self, reports, retry_count=0):
        from agents.reflector import ReflectorOutput
        consensus = "Normal"
        if reports:
            from collections import Counter
            cats = [r.crime_category for r in reports
                    if r.crime_category != "ENVIRONMENTAL_ASSESSMENT"]
            if cats:
                consensus = Counter(cats).most_common(1)[0][0]

        return ReflectorOutput(
            conflict_type="NONE",
            conflict_layer="NONE",
            target_agent="",
            refinement_instruction="",
            recommended_focus_frames=[],
            rcons_score=1.0,
            consensus_category=consensus,
            is_convergent=True,
            audit_log=["NullReflector: 跳過所有審查"],
        )

    def get_debate_log(self):
        return ["NullReflector: no debate"]


# ── 樣本載入（委派 pilot_experiment）──────────────────────────

def load_ablation_samples(n_samples: int, split: str = "Test") -> List[Dict]:
    """載入消融實驗樣本（使用 UCA 標註 + UCF-Crime 影片）。"""
    from scripts.pilot_experiment import load_pilot_samples
    return load_pilot_samples(n_samples=n_samples, split=split)


def extract_frames(video_path: str, n_frames: int = 32) -> List:
    """從影片抽取幀（委派 pilot_experiment）。"""
    from scripts.pilot_experiment import extract_frames as _extract
    return _extract(video_path, n_frames)


# ── 變體建構 ─────────────────────────────────────────────────

def build_variant(variant_name: str) -> Dict[str, Any]:
    """
    根據變體名稱建構 pipeline 組件。

    Returns:
        {
            "agents_dict": {...},
            "reflector": ReflectorAgent or NullReflector,
            "rag_module": RAGModule or None,
            "ae_config": dict,  # ActionEmotion 的額外配置
        }
    """
    from agents import EnvironmentAgent, ReflectorAgent
    from agents.action_emotion_agent import ActionEmotionAgent
    from rag import HierarchicalRAG
    from rag.rag_module import RAGModule
    from config import cfg

    agents_dict = {}
    reflector = ReflectorAgent()
    rag_module = None
    ae_config = {}

    if variant_name == "full_system":
        agents_dict["environment"] = EnvironmentAgent()
        agents_dict["action_emotion"] = ActionEmotionAgent()
        rag = HierarchicalRAG(cfg.rag)
        rag_module = RAGModule(rag)

    elif variant_name == "no_environment":
        agents_dict["action_emotion"] = ActionEmotionAgent()
        rag = HierarchicalRAG(cfg.rag)
        rag_module = RAGModule(rag)

    elif variant_name == "no_rag":
        agents_dict["environment"] = EnvironmentAgent()
        agents_dict["action_emotion"] = ActionEmotionAgent()
        rag_module = None  # Planner 的 self.rag = None → 跳過 Step 3a/3c

    elif variant_name == "no_ae_backbone":
        agents_dict["environment"] = EnvironmentAgent()
        agents_dict["action_emotion"] = ActionEmotionAgent()
        ae_config["disable_backbone"] = True  # ActionEmotion 只用 MIL Head
        rag = HierarchicalRAG(cfg.rag)
        rag_module = RAGModule(rag)

    elif variant_name == "no_reflector":
        agents_dict["environment"] = EnvironmentAgent()
        agents_dict["action_emotion"] = ActionEmotionAgent()
        reflector = NullReflector()
        rag = HierarchicalRAG(cfg.rag)
        rag_module = RAGModule(rag)

    else:
        raise ValueError(f"未知的消融變體：{variant_name}")

    return {
        "agents_dict": agents_dict,
        "reflector": reflector,
        "rag_module": rag_module,
        "ae_config": ae_config,
    }


# ── 單變體實驗 ───────────────────────────────────────────────

def run_variant(
    variant_name: str,
    samples: List[Dict],
    output_dir: Path,
) -> List[Dict]:
    """
    執行單一消融變體。
    """
    from agents.planner import PlannerAgent

    logger.info(f"═══ 消融變體：{variant_name} ═══")
    variant = build_variant(variant_name)

    planner = PlannerAgent(
        agents_dict=variant["agents_dict"],
        reflector=variant["reflector"],
        rag_module=variant["rag_module"],
    )

    results = []
    for i, sample in enumerate(samples):
        logger.info(f"  [{i+1}/{len(samples)}] {sample['video_id']}")
        frames = extract_frames(sample["video_path"])
        metadata = sample.get("metadata", {})

        # no_ae_backbone 配置
        if variant["ae_config"].get("disable_backbone"):
            metadata["disable_backbone"] = True

        try:
            result = planner.run(frames, metadata)
            results.append({
                "video_id": sample["video_id"],
                "ground_truth": sample["ground_truth"],
                "predicted": result.get("final_category", "Normal"),
                "correct": result.get("final_category") == sample["ground_truth"],
                "rcons": result.get("rcons", 0.0),
                "rlegal": result.get("rlegal", 0.0),
                "rcost": result.get("rcost", 0.0),
                "total_turns": result.get("total_turns", 0),
                "conflict_type": result.get("conflict_type", "NONE"),
                "report": result.get("fact_finding", {}).get("description", ""),
            })
        except Exception as e:
            logger.error(f"  樣本 {sample['video_id']} 失敗：{e}")

    # 存結果
    variant_path = output_dir / f"{variant_name}.json"
    with open(variant_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"  → {variant_path}")

    return results


# ── 彙整比較表 ───────────────────────────────────────────────

def build_comparison_table(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
):
    """
    對應論文第四章的消融實驗表格。

    Columns: Variant | Accuracy | Rcons_avg | Rlegal_avg | Rcost_avg | Turns_avg
    """
    import statistics

    table = {}
    for variant_name, results in all_results.items():
        if not results:
            continue
        n = len(results)
        acc = sum(1 for r in results if r["correct"]) / n
        rcons_avg = statistics.mean([r["rcons"] for r in results])
        rlegal_avg = statistics.mean([r["rlegal"] for r in results])
        rcost_avg = statistics.mean([r["rcost"] for r in results])
        turns_avg = statistics.mean([r["total_turns"] for r in results])

        table[variant_name] = {
            "n_samples": n,
            "accuracy": round(acc, 4),
            "rcons_avg": round(rcons_avg, 4),
            "rlegal_avg": round(rlegal_avg, 4),
            "rcost_avg": round(rcost_avg, 4),
            "turns_avg": round(turns_avg, 2),
        }

    # 存表格
    table_path = output_dir / "comparison_table.json"
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

    # 文字版表格
    txt_path = output_dir / "comparison_table.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        header = f"{'Variant':<20} {'Acc':>6} {'Rcons':>6} {'Rlegal':>7} {'Rcost':>6} {'Turns':>6}"
        f.write(header + "\n")
        f.write("─" * len(header) + "\n")
        for name, m in table.items():
            f.write(
                f"{name:<20} {m['accuracy']:>6.1%} {m['rcons_avg']:>6.3f} "
                f"{m['rlegal_avg']:>7.3f} {m['rcost_avg']:>6.3f} {m['turns_avg']:>6.1f}\n"
            )

    logger.info(f"比較表格 → {table_path}")
    logger.info(f"文字版   → {txt_path}")

    return table


# ── Main ─────────────────────────────────────────────────────

VARIANTS = [
    "full_system",
    "no_environment",
    "no_rag",
    "no_ae_backbone",
    "no_reflector",
]


def main():
    parser = argparse.ArgumentParser(description="消融實驗")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--split", default="Test", choices=["Train", "Test", "Val"])
    parser.add_argument("--output_dir", default="./results/ablation")
    parser.add_argument(
        "--variants", nargs="*", default=VARIANTS,
        help="要跑的變體（預設全部）"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_ablation_samples(args.n_samples, args.split)
    if not samples:
        logger.error("無法載入樣本")
        return

    all_results = {}
    for variant in args.variants:
        if variant not in VARIANTS:
            logger.warning(f"未知變體 '{variant}'，跳過")
            continue
        results = run_variant(variant, samples, output_dir)
        all_results[variant] = results

    if all_results:
        build_comparison_table(all_results, output_dir)


if __name__ == "__main__":
    main()
