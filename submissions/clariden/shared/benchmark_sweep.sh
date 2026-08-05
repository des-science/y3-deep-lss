#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=benchmark_sweep
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Single-GPU synthetic (config x batch) step-time/memory sweep -- generic form of the harness
# that used to be ~15 near-identical scripts under benchmark/. Each (config, batch) is its own
# `srun` step (fresh process -> memory released; OOM/kernel errors classified, not fatal to the
# sweep), appended to a JSONL, then aggregated into CSV/markdown.
#
# Usage:
#   BENCH_SCRIPT=benchmark_transformer.py CONFIGS_GLOB="configs/transformer/lensing/bench_t7/*.yaml" \
#       OUT_DIR=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/<name> \
#       BATCH_SIZES="16 32" sbatch benchmark_sweep.sh
#
# BENCH_CONFIGS="a.yaml b.yaml" times only those (paths relative to CONFIGS_GLOB's dir, or
# repo-relative, or absolute) and APPENDS to JSONL instead of truncating -- use to add a config
# without re-timing ones already recorded.

BENCH_SCRIPT="${BENCH_SCRIPT:-benchmark_transformer.py}"
R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/$BENCH_SCRIPT"

: "${CONFIGS_GLOB:?set CONFIGS_GLOB, e.g. configs/transformer/lensing/bench_t7/*.yaml}"
: "${OUT_DIR:?set OUT_DIR, e.g. /iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/<name>}"
JSONL="$OUT_DIR/benchmark_results.jsonl"

BATCH_SIZES="${BATCH_SIZES:-16}"
PROBE="${PROBE:-combined}"
MSFM="${MSFM:-$R/multiprobe-simulation-forward-model/configs/v17/baseline.yaml}"
SCALES="${SCALES:-$R/y3-deep-lss/configs/scales/8wl,32gc.yaml}"
LOSS="${LOSS:-$R/y3-deep-lss/configs/loss/vmim.yaml}"
DATA="${DATA:-$R/y3-deep-lss/configs/data/default.yaml}"
PROBES_CONFIG="${PROBES_CONFIG:-$R/y3-deep-lss/configs/probes/${PROBE}.yaml}"

GLOB_ABS="$CONFIGS_GLOB"
case "$CONFIGS_GLOB" in /*) ;; *) GLOB_ABS="$R/y3-deep-lss/$CONFIGS_GLOB" ;; esac

shopt -s nullglob
if [ -n "${BENCH_CONFIGS:-}" ]; then
    CONFIGS=()
    for c in $BENCH_CONFIGS; do
        case "$c" in
            /*) CONFIGS+=("$c") ;;
            */*) CONFIGS+=("$R/$c") ;;
            *) CONFIGS+=("$(dirname "$GLOB_ABS")/$c") ;;
        esac
    done
    APPEND_JSONL=1
else
    CONFIGS=($GLOB_ABS)
    APPEND_JSONL=0
fi
shopt -u nullglob
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs matched $CONFIGS_GLOB" >&2
    exit 1
fi
for cfg in "${CONFIGS[@]}"; do
    [ -f "$cfg" ] || { echo "Config not found: $cfg" >&2; exit 1; }
done

mkdir -p "$OUT_DIR/slurm"
if [ "$APPEND_JSONL" -eq 1 ] && [ -f "$JSONL" ]; then
    for cfg in "${CONFIGS[@]}"; do
        base="$(basename "$cfg")"
        grep -vF "$base\"" "$JSONL" > "$JSONL.tmp" && mv "$JSONL.tmp" "$JSONL"
    done
else
    : > "$JSONL"
fi

for cfg in "${CONFIGS[@]}"; do
    cfg_name="$(basename "$(dirname "$cfg")")/$(basename "$cfg")"
    for bs in $BATCH_SIZES; do
        echo ">>> ${cfg_name}  batch=$bs ..."
        log="$(mktemp)"
        srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
            bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --single \
                --net_config '$cfg' --batch_size $bs \
                --msfm_config '$MSFM' --probes_config '$PROBES_CONFIG' --scales_config '$SCALES' \
                --loss_config '$LOSS' --data_config '$DATA'" \
            > "$log" 2>&1 || true

        if grep -q '^BENCH_JSON ' "$log"; then
            row="$(grep '^BENCH_JSON ' "$log" | head -1 | sed 's/^BENCH_JSON //')"
            echo "${row/\{/\{\"probe\": \"$PROBE\", }" >> "$JSONL"
            echo "    OK"
        else
            low="$(tr 'A-Z' 'a-z' < "$log")"
            if echo "$low" | grep -q 'resourceexhausted\|out of memory'; then
                st="OOM"
            elif echo "$low" | grep -q 'invalid configuration argument\|non-ok-status'; then
                st="KERNEL"
            else
                st="ERROR"
                echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -20 | sed 's/^/      /'
            fi
            printf '{"probe": "%s", "config": "%s", "batch_size": %s, "status": "%s"}\n' \
                "$PROBE" "$cfg_name" "$bs" "$st" >> "$JSONL"
            echo "    $st"
        fi
        rm -f "$log"
    done
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
