#!/usr/bin/env bash
# Ablation orchestrator — 序列跑 5 個 ablation variants + baseline（選配）
#
# 啟動方式：
#   nohup /home/yuting/crime_analysis/crime_analysis/scripts/run_ablations.sh \
#       > /tmp/ablations.log 2>&1 & disown
#
# 或者在 tmux 裡跑：
#   tmux new -s ablations 'bash scripts/run_ablations.sh'
#
# 特性：
#   - 每個 variant 自己的 output dir，不會互相覆蓋
#   - 一律 --resume，crash / 重啟只損失最多 1 題
#   - 預設跑 52 anomaly + 4 Normal（Option D 評估需要 Normal）
#   - 若已有 pilot_v2 結果，可用 BASELINE_DIR 指定當 no-flag baseline
#
# 環境變數（可覆寫）：
#   N_SAMPLES   (default 52)
#   N_NORMAL    (default 4)
#   SPLIT       (default Test)
#   SEED        (default 42)
#   VARIANTS    (default "no-env no-rag no-vlm no-reflector no-vlm-report")
#   OUTPUT_ROOT (default outputs/ablations)
#   SKIP_BASELINE (default 1；若 pilot_v2 已跑完，直接拿它當 baseline)
#
# 產出：
#   outputs/ablations/<variant>/pilot_stats.json + pilot_reports/ + pilot_summary.txt

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

# ── 可覆寫參數 ──
N_SAMPLES="${N_SAMPLES:-52}"
N_NORMAL="${N_NORMAL:-4}"
SPLIT="${SPLIT:-Test}"
SEED="${SEED:-42}"
VARIANTS="${VARIANTS:-no-env no-rag no-vlm no-reflector no-vlm-report}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"

PY="$HOME/miniconda3/envs/crime/bin/python"
unset SSL_CERT_FILE
mkdir -p "$OUTPUT_ROOT"

run_variant() {
    local variant="$1"         # 格式: no-env / no-rag / ...
    local flag=""
    local name="$variant"
    if [ "$variant" != "baseline" ]; then
        flag="--$variant"
        name="${variant//-/_}"  # no-env → no_env 當資料夾名
    fi
    local outdir="$OUTPUT_ROOT/$name"

    echo ""
    echo "========================================================"
    echo "[$(date -Iseconds)] Variant: $variant  (flag: ${flag:-<none>})"
    echo "  output_dir: $outdir"
    echo "========================================================"

    $PY -m scripts.pilot_experiment \
        --n_samples "$N_SAMPLES" --n-normal "$N_NORMAL" \
        --split "$SPLIT" --seed "$SEED" \
        --output_dir "$outdir" \
        --resume \
        $flag
    local exit_code=$?
    echo "[$(date -Iseconds)] $variant exit=$exit_code"
    return $exit_code
}

echo "=== ABLATION ORCHESTRATOR START $(date -Iseconds) ==="
echo "  N_SAMPLES=$N_SAMPLES  N_NORMAL=$N_NORMAL  SPLIT=$SPLIT  SEED=$SEED"
echo "  VARIANTS: $VARIANTS"
echo "  OUTPUT_ROOT: $OUTPUT_ROOT"
echo "  SKIP_BASELINE: $SKIP_BASELINE"

# Optional baseline run（若 pilot_v2 還沒跑完就用這個補）
if [ "$SKIP_BASELINE" != "1" ]; then
    run_variant baseline || echo "baseline failed，繼續 ablation"
fi

for variant in $VARIANTS; do
    run_variant "$variant" || echo "$variant failed，繼續下一個"
done

echo ""
echo "=== ORCHESTRATOR DONE $(date -Iseconds) ==="
echo "用 scripts/consolidate_ablation.py 合併結果："
echo "  $PY -m scripts.consolidate_ablation --root $OUTPUT_ROOT"
