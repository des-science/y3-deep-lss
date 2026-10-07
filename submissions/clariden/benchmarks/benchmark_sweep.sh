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

# Single-GPU synthetic step-time/memory sweep over (config x batch), one srun step per point, into
# $OUT_DIR/benchmark_results.jsonl, then aggregated. BENCH_SCRIPT is benchmark_{transformer,resnet}.py.
#   BENCH_SCRIPT=benchmark_resnet.py CONFIGS_GLOB="configs/maps/dev/deepsphere/combined/bench_v12/*.yaml" \
#       OUT_DIR=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/<name> BATCH_SIZES="16 32" \
#       sbatch benchmark_sweep.sh
# BENCH_CONFIGS="a.yaml b.yaml" times only those (relative to the glob's dir, to y3-deep-lss, or
# absolute) and appends to the JSONL instead of starting it over.

source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

: "${CONFIGS_GLOB:?set CONFIGS_GLOB, e.g. configs/maps/dev/transformer/lensing/bench_t7/*.yaml}"
: "${OUT_DIR:?set OUT_DIR, e.g. /iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/<name>}"
BENCH_SCRIPT="${BENCH_SCRIPT:-benchmark_transformer.py}"
BATCH_SIZES="${BATCH_SIZES:-16}"
PROBE="${PROBE:-combined}"  # configs/probes/<PROBE>.yaml

SCRIPT="$DEEP_LSS/deep_lss/apps/benchmark/$BENCH_SCRIPT"
TF_PY="$HOME/dlss/tf_env/bin/python"  # the benchmark apps run from the tf_env venv
JSONL="$OUT_DIR/benchmark_results.jsonl"

if [ -n "${BENCH_CONFIGS:-}" ]; then
    glob_dir="$(dirname "$CONFIGS_GLOB")"
    case "$glob_dir" in /*) ;; *) glob_dir="$DEEP_LSS/$glob_dir" ;; esac
    CONFIGS=()
    for c in $BENCH_CONFIGS; do
        case "$c" in
            /*) CONFIGS+=("$c") ;;
            */*) CONFIGS+=("$DEEP_LSS/$c") ;;
            *) CONFIGS+=("$glob_dir/$c") ;;
        esac
    done
    for cfg in "${CONFIGS[@]}"; do
        [ -f "$cfg" ] || { echo "Config not found: $cfg" >&2; exit 1; }
    done
    # Drop earlier rows of the re-timed configs so they are not duplicated.
    for cfg in "${CONFIGS[@]}"; do
        [ -f "$JSONL" ] && grep -vF "$(basename "$cfg")\"" "$JSONL" > "$JSONL.tmp" && mv "$JSONL.tmp" "$JSONL"
    done
else
    resolve_glob "$CONFIGS_GLOB"
    [ "$DRYRUN" = "1" ] || { mkdir -p "$OUT_DIR"; : > "$JSONL"; }
fi

# --- Time every (config, batch) point ----------------------------------------------------------

for cfg in "${CONFIGS[@]}"; do
    cfg_name="$(basename "$(dirname "$cfg")")/$(basename "$cfg")"
    for bs in $BATCH_SIZES; do
        echo ">>> $cfg_name  batch=$bs"
        log="$(mktemp)"
        $SRUN --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
            "$TF_PY" "$SCRIPT" --single \
                --net_config "$cfg" --batch_size "$bs" \
                --msfm_config "$MSFM_CONFIG" --probes_config "$DEEP_LSS/configs/probes/$PROBE.yaml" \
                --scales_config "$SCALES_CONFIG" --loss_config "$LOSS_CONFIG" --data_config "$DATA_CONFIG" \
            > "$log" 2>&1
        [ "$DRYRUN" = "1" ] && { cat "$log"; rm -f "$log"; continue; }

        # A failed point is recorded, not fatal: OOM and kernel limits are what the sweep looks for.
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
                grep -viE 'ptx85|gpu_timer' "$log" | tail -20 | sed 's/^/      /'
            fi
            printf '{"probe": "%s", "config": "%s", "batch_size": %s, "status": "%s"}\n' \
                "$PROBE" "$cfg_name" "$bs" "$st" >> "$JSONL"
            echo "    $st"
        fi
        rm -f "$log"
    done
done

# --- Aggregate into CSV/markdown ---------------------------------------------------------------

$SRUN --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    "$TF_PY" "$SCRIPT" --aggregate --jsonl "$JSONL" --out_dir "$OUT_DIR"
