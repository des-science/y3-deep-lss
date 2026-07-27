#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_combined
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/combined/slurm/slurm-%j.out

# Size local_batch_size + n_steps for the MULTI-RESOLUTION combined-probe transformer
# (configs/transformer/combined/maps+cls.yaml). The map branch is now HealpixMultiResMapEncoder:
# lensing @512 is the main input, clustering @256 is INJECTED at body level 1 (not upsampled), so
# the combined cost differs from both lensing-only (b20 = 81.6 GB / 181.6 ms) and the old
# upsampled-512 combined. The benchmark builds the exact map branch through
# HealpixTransformerNetwork (maps-only path; ignores the cls head — its overhead is folded into the
# synthetic->real K, see the sizing recipe), fed synthetic batches on a single GPU.
#
# Method (project_transformer_bench_sizing_recipe):
#   1. sweep the batch ladder, record peak_gb / step_ms of the OK cells.
#   2. pick the largest batch whose synthetic peak_gb stays < ~85 GB (the NCCL-safe band; the
#      single-GPU benchmark cannot see the multi-GPU all-reduce deadlock that ~89 GB triggers).
#   3. n_steps = floor_to_10k(11h * 3600 / (K * step_ms/1000)), K = 1.4528.
# The lensing bench_t6/default.yaml @ b20 anchor is re-run here (probes=lensing) to confirm the
# allocation reproduces the recorded 81.6 GB / 181.6 ms before trusting the combined numbers.

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
T6_DIR="$R/y3-deep-lss/configs/transformer/lensing/bench_t6"
COMBINED="$R/y3-deep-lss/configs/transformer/combined/maps+cls.yaml"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/combined"
JSONL="$OUT_DIR/benchmark_results.jsonl"
mkdir -p "$OUT_DIR/slurm"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
PROBES_LENSING="$R/y3-deep-lss/configs/probes/lensing.yaml"
PROBES_COMBINED="$R/y3-deep-lss/configs/probes/combined.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

run_one () {  # $1=config path  $2=batch  $3=probes config
    local cfg="$1" bs="$2" probes="$3" name log low st
    name="$(basename "$cfg")"
    echo ">>> $name  batch=$bs  probes=$(basename "$probes") ..."
    log="$(mktemp)"
    srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
        bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --single \
            --net_config '$cfg' --batch_size $bs \
            --msfm_config '$MSFM' --probes_config '$probes' --scales_config '$SCALES' \
            --loss_config '$LOSS' --data_config '$DATA'" \
        > "$log" 2>&1 || true

    if grep -q '^BENCH_JSON ' "$log"; then
        grep '^BENCH_JSON ' "$log" | head -1 | sed 's/^BENCH_JSON //' | tee -a "$JSONL"
        echo "    OK"
    else
        low="$(tr 'A-Z' 'a-z' < "$log")"
        if echo "$low" | grep -q 'resourceexhausted\|out of memory'; then
            st="OOM"
        elif echo "$low" | grep -q 'invalid configuration argument\|non-ok-status'; then
            st="KERNEL"
        else
            st="ERROR"
            echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -15 | sed 's/^/      /'
        fi
        printf '{"config": "%s", "batch_size": %s, "n_tokens": "-", "pix_per_token": "-", "params_M": "-", "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
            "$name" "$bs" "$st" >> "$JSONL"
        echo "    $st"
    fi
    rm -f "$log"
}

: > "$JSONL"

echo "===== anchor: lensing bench_t6/default.yaml @ b20 (expect ~81.6 GB / 181.6 ms) ====="
run_one "$T6_DIR/default.yaml" 20 "$PROBES_LENSING"

echo ""
echo "===== combined/maps+cls.yaml (multi-res) batch ladder ====="
for bs in 16 18 20 24 28; do
    run_one "$COMBINED" "$bs" "$PROBES_COMBINED"
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
