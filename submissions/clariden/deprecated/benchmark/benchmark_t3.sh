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
#SBATCH --job-name=bench_t3
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t3/slurm/slurm-%j.out

# Benchmark the bench_t3 nested-transformer configs for GPU-memory fit and step time, to pick
# per-config training batch sizes (fill the 120 GB GH200 / stay under the softmax kernel-launch
# ceiling) and a step count for ~10 h of training.
#
# Sweeps each bench_t3/*.yaml over a batch ladder. Also benchmarks a few bench_t2 configs at
# batch 16 as synthetic->real calibration anchors (we have their real 4-GPU it/s from the t2
# training logs), so the single-GPU synthetic step_ms can be mapped to real 4-GPU wall time.
#
# Each (config, batch) runs as its own `srun --environment=tensorflow` step (only form that
# reliably has TF in the CSCS container); a fresh process releases GPU memory between runs and
# contains OOM / kernel-launch crashes, which the loop classifies (OK / OOM / KERNEL / ERROR).

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark/benchmark_transformer.py"
T3_DIR="$R/y3-deep-lss/configs/transformer/lensing/bench_t3"
T2_DIR="$R/y3-deep-lss/configs/transformer/lensing/bench_t2"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t3"
JSONL="$OUT_DIR/benchmark_results.jsonl"
mkdir -p "$OUT_DIR/slurm"

T3_BATCHES="${T3_BATCHES:-16 24 32 48 64}"
T2_BATCHES="${T2_BATCHES:-16}"
T2_CAL="default deep wide global"   # bench_t2 configs with real 4-GPU logs -> calibration

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
PROBES="$R/y3-deep-lss/configs/probes/lensing.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

run_one () {  # $1=config path  $2=batch
    local cfg="$1" bs="$2" name log low st
    name="$(basename "$cfg")"
    echo ">>> $name  batch=$bs ..."
    log="$(mktemp)"
    srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
        bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --single \
            --net_config '$cfg' --batch_size $bs \
            --msfm_config '$MSFM' --probes_config '$PROBES' --scales_config '$SCALES' \
            --loss_config '$LOSS' --data_config '$DATA'" \
        > "$log" 2>&1 || true

    if grep -q '^BENCH_JSON ' "$log"; then
        grep '^BENCH_JSON ' "$log" | head -1 | sed 's/^BENCH_JSON //' >> "$JSONL"
        echo "    OK"
    else
        low="$(tr 'A-Z' 'a-z' < "$log")"
        if echo "$low" | grep -q 'resourceexhausted\|out of memory'; then
            st="OOM"
        elif echo "$low" | grep -q 'invalid configuration argument\|non-ok-status'; then
            st="KERNEL"
        else
            st="ERROR"
            echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -12 | sed 's/^/      /'
        fi
        printf '{"config": "%s", "batch_size": %s, "n_tokens": "-", "pix_per_token": "-", "params_M": "-", "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
            "$name" "$bs" "$st" >> "$JSONL"
        echo "    $st"
    fi
    rm -f "$log"
}

: > "$JSONL"

echo "===== bench_t3 sweep (batches: $T3_BATCHES) ====="
for cfg in "$T3_DIR"/*.yaml; do
    for bs in $T3_BATCHES; do
        run_one "$cfg" "$bs"
    done
done

echo ""
echo "===== bench_t2 calibration anchors (batches: $T2_BATCHES) ====="
for c in $T2_CAL; do
    for bs in $T2_BATCHES; do
        run_one "$T2_DIR/$c.yaml" "$bs"
    done
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
