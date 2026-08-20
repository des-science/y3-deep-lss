#!/bin/bash
# Moved to deprecated/ 2026-08-04: superseded by submissions/clariden/benchmark_sweep.sh, the
# generalized form of this harness (BENCH_SCRIPT/CONFIGS_GLOB/OUT_DIR/BATCH_SIZES env vars).
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_v4
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/resnet/v4/slurm/slurm-%j.out

# Benchmark the configs/deepsphere/combined/bench_v4/ architecture sweep (combined maps+cls).
# Goal for THIS round: at the fixed comparison batch (20), record peak_gb / step_ms / throughput
# per variant so we can (a) confirm batch 20 fits and clears the cuSPARSE SPARSE_LIMIT, and
# (b) set each variant's ONE-12h-job n_steps from its measured it/s (so the cosine schedule
# completes within 12 h -- see the header of bench_v4/default.yaml). No training happens here.
#
# Model of submissions/clariden/benchmark/benchmark_resnet.sh: each (config, batch) is its own
# `srun --environment=tensorflow` step (fresh process -> memory released, OOM/kernel-limit contained
# and classified, not fatal to the sweep). GCNN memory is tiny; the binding constraint is the
# cuSPARSE nnz(a)*output.shape[1] > 2^31 kernel-launch ceiling (classified as SPARSE_LIMIT) -- the
# wide (w64) / hi-res-trunk (hires_trunk) / single-res (singleres) variants are the ones to watch.
#
# Submit:  sbatch submissions/clariden/benchmark/benchmark_v4.sh
# Headroom check (does batch fit above 20?):
#          BATCH_SIZES="20 32 48" sbatch submissions/clariden/benchmark/benchmark_v4.sh

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark/benchmark_resnet.py"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/resnet/v4"
JSONL="$OUT_DIR/benchmark_results.jsonl"

# Fixed comparison batch for this round (unified recipe). Override to probe headroom.
BATCH_SIZES="${BATCH_SIZES:-20}"

# Same forward-model / scales / loss / data as benchmark_resnet.sh (synthetic batches, so these
# only fix the map geometry + channel count + loss head used to build the model).
MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"
PROBE="combined"
PROBES_CONFIG="$R/y3-deep-lss/configs/probes/${PROBE}.yaml"

# All bench_v4 variants (default + one-knob siblings).
BENCH_DIR="$R/y3-deep-lss/configs/deepsphere/combined/bench_v4"
# BENCH_CONFIGS override: space-separated basenames (or paths) to time ONLY those configs and
# APPEND to the existing JSONL, instead of re-timing the whole dir. Use it to add a newly-created
# variant without re-running the siblings whose step time is already recorded (their geometry is
# unchanged). Example: BENCH_CONFIGS="bernstein.yaml global_attn.yaml" sbatch benchmark_v4.sh
# extglob must be enabled at PARSE time of the !(*_2x) glob below; set it here at top level so it
# is active before the if-compound is parsed at runtime (a shopt inside the branch would be too late).
shopt -s nullglob extglob
if [ -n "${BENCH_CONFIGS:-}" ]; then
    CONFIGS=()
    for c in $BENCH_CONFIGS; do
        case "$c" in
            /*) CONFIGS+=("$c") ;;                 # absolute path
            */*) CONFIGS+=("$R/$c") ;;             # repo-relative path
            *) CONFIGS+=("$BENCH_DIR/$c") ;;       # bare basename in the bench dir
        esac
    done
    APPEND_JSONL=1
else
    # Skip *_2x.yaml: the chained (2x12 h) long-run variants share default/w64 geometry, so their
    # step time is identical -- no separate timing needed (their n_steps just follow ~2x the parent).
    CONFIGS=("$BENCH_DIR"/!(*_2x).yaml)
    APPEND_JSONL=0
fi
shopt -u nullglob extglob
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs found in $BENCH_DIR" >&2
    exit 1
fi
for cfg in "${CONFIGS[@]}"; do
    [ -f "$cfg" ] || { echo "Config not found: $cfg" >&2; exit 1; }
done

mkdir -p "$OUT_DIR/slurm"
# Full-dir sweep truncates and rewrites; a targeted BENCH_CONFIGS run appends to keep the
# already-recorded siblings (drop any prior rows for the configs being re-timed to avoid dupes).
if [ "$APPEND_JSONL" -eq 1 ] && [ -f "$JSONL" ]; then
    for cfg in "${CONFIGS[@]}"; do
        # benchmark_resnet.py writes "config" as the bare basename (e.g. "bernstein.yaml"); the
        # shell failure-path writes "bench_v4/<basename>". Match the basename + closing quote to
        # drop any prior row for this config in either form before re-appending.
        base="$(basename "$cfg")"
        grep -vF "$base\"" "$JSONL" > "$JSONL.tmp" && mv "$JSONL.tmp" "$JSONL"
    done
else
    : > "$JSONL"
fi

for cfg in "${CONFIGS[@]}"; do
    cfg_name="$(basename "$(dirname "$cfg")")/$(basename "$cfg")"   # e.g. bench_v4/w64.yaml
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
            elif echo "$low" | grep -q 'cannot use gpu when output.shape'; then
                st="SPARSE_LIMIT"  # cuSPARSE nnz(a)*output.shape[1] > 2^31 kernel-launch ceiling
            elif echo "$low" | grep -q 'invalid configuration argument\|non-ok-status'; then
                st="KERNEL"
            else
                st="ERROR"
                echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -20 | sed 's/^/      /'
            fi
            printf '{"probe": "%s", "config": "%s", "batch_size": %s, "multires": "-", "return_cls": "-", "n_pix": "-", "graph_build_s": "-", "params_M": "-", "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
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
