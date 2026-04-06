"""
全實驗自動執行腳本

用法：
    cd crime_analysis
    python -m scripts.run_all_experiments              # 跑全部
    python -m scripts.run_all_experiments --only ours no_rag  # 只跑指定變體
    python -m scripts.run_all_experiments --skip pilot  # 跳過 pilot

實驗組：
    pilot          Pilot 52 部（門檻校準）
    ours           完整系統 154 部
    no_env         消融① 無 EnvironmentAgent
    no_rag         消融② 無 RAG
    no_vlm         消融③ 無 VLM 分類
    no_reflector   消融④ NullReflector
    no_vlm_report  消融⑤ 無 VLM 報告

每個實驗的結果存到 outputs/experiments/{experiment_name}/ 下。
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable
SCRIPT = "-m scripts.pilot_experiment"
SEED = 42
SPLIT = "Test"

EXPERIMENTS = {
    "pilot": {
        "n_samples": 52,
        "flags": [],
        "description": "Pilot（門檻校準，52 部）",
    },
    "ours": {
        "n_samples": 200,
        "flags": ["--exclude-pilot"],
        "description": "完整系統（Ours，154 部，排除 Pilot）",
    },
    "no_env": {
        "n_samples": 200,
        "flags": ["--exclude-pilot", "--no-env"],
        "description": "消融① 無 EnvironmentAgent",
    },
    "no_rag": {
        "n_samples": 200,
        "flags": ["--exclude-pilot", "--no-rag"],
        "description": "消融② 無 RAG",
    },
    "no_vlm": {
        "n_samples": 200,
        "flags": ["--exclude-pilot", "--no-vlm"],
        "description": "消融③ 無 VLM 分類（MIL Head only）",
    },
    "no_reflector": {
        "n_samples": 200,
        "flags": ["--exclude-pilot", "--no-reflector"],
        "description": "消融④ NullReflector（= Baseline B）",
    },
    "no_vlm_report": {
        "n_samples": 200,
        "flags": ["--exclude-pilot", "--no-vlm-report"],
        "description": "消融⑤ 無 VLM 報告（fallback 模板）",
    },
}


def run_experiment(name: str, config: dict, base_output: Path):
    """執行單一實驗並保存結果。"""
    output_dir = base_output / name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, *SCRIPT.split(),
        "--n_samples", str(config["n_samples"]),
        "--split", SPLIT,
        "--seed", str(SEED),
        "--output_dir", str(output_dir),
        *config["flags"],
    ]

    logger.info(f"{'='*60}")
    logger.info(f"開始實驗：{name} — {config['description']}")
    logger.info(f"指令：{' '.join(cmd)}")
    logger.info(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    logger.info(f"實驗 {name} 完成：{status}（{elapsed/60:.1f} 分鐘）")

    # 記錄 meta
    meta = {
        "experiment": name,
        "description": config["description"],
        "n_samples": config["n_samples"],
        "flags": config["flags"],
        "seed": SEED,
        "split": SPLIT,
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": result.returncode,
    }
    with open(output_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="全實驗自動執行")
    parser.add_argument(
        "--only", nargs="+", choices=list(EXPERIMENTS.keys()),
        help="只跑指定的實驗（預設全部）",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=list(EXPERIMENTS.keys()),
        help="跳過指定的實驗",
    )
    parser.add_argument(
        "--output_base", default="./outputs/experiments",
        help="實驗結果根目錄",
    )
    args = parser.parse_args()

    base_output = Path(args.output_base)
    base_output.mkdir(parents=True, exist_ok=True)

    # 決定要跑哪些
    to_run = list(EXPERIMENTS.keys())
    if args.only:
        to_run = args.only
    if args.skip:
        to_run = [e for e in to_run if e not in args.skip]

    logger.info(f"實驗計畫：{to_run}")
    logger.info(f"輸出目錄：{base_output.resolve()}")

    results = {}
    total_start = time.time()

    for name in to_run:
        config = EXPERIMENTS[name]
        ok = run_experiment(name, config, base_output)
        results[name] = "OK" if ok else "FAIL"

    total_elapsed = time.time() - total_start

    # 總結
    logger.info(f"\n{'='*60}")
    logger.info(f"全部實驗完成（{total_elapsed/60:.1f} 分鐘）")
    logger.info(f"{'='*60}")
    for name, status in results.items():
        logger.info(f"  {name:20s} {status}")

    # 存總結
    with open(base_output / "run_summary.json", "w") as f:
        json.dump({
            "experiments": results,
            "total_elapsed_minutes": round(total_elapsed / 60, 1),
        }, f, indent=2)


if __name__ == "__main__":
    main()
