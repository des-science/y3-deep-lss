#!/bin/bash
# Moved to deprecated/ 2026-08-04: superseded by submissions/clariden/benchmark_sweep.sh, the
# generalized form of this harness (BENCH_SCRIPT/CONFIGS_GLOB/OUT_DIR/BATCH_SIZES env vars).
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_t7
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t7/slurm/slurm-%j.out

# Quick double-check of the bench_t7 configs: each is bench_t6/default.yaml (the winning base
# architecture) + one feature toggle. Confirm that at the base training batch (local_batch_size 20)
# every variant stays inside the ~85 GB NCCL-safe memory band and that its step time is near the
# base (default.yaml @ b20: 81.6 GB, 181.6 ms synthetic single-GPU bf16), so the inherited
# n_steps=150000 / ~11 h sizing holds. real 4-GPU wall/step ~= K x synthetic, K=1.4528.
#
# default.yaml is re-benchmarked here at b20 as an in-run anchor (apples-to-apples with the
# variants in this same allocation). Each (config, batch) runs as its own
# `srun --environment=tensorflow` step; precision (bf16) and jit_compile_body (true) come from
# each config, as in real training.

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
T6_DIR="$R/y3-deep-lss/configs/transformer/lensing/bench_t6"
T7_DIR="$R/y3-deep-lss/configs/transformer/lensing/bench_t7"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t7"
JSONL="$OUT_DIR/benchmark_results.jsonl"
mkdir -p "$OUT_DIR/slurm"

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
            echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -12 | sed 's/^/      /'
        fi
        printf '{"config": "%s", "batch_size": %s, "n_tokens": "-", "pix_per_token": "-", "params_M": "-", "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
            "$name" "$bs" "$st" >> "$JSONL"
        echo "    $st"
    fi
    rm -f "$log"
}

: > "$JSONL"

echo "===== bench_t6/default.yaml (base anchor) ====="
run_one "$T6_DIR/default.yaml" 20

echo ""
echo "===== bench_t7 variants (base + one feature) at the training batch (b20) ====="
for cfg in dropout masked multiscale pool symmetric; do
    run_one "$T7_DIR/$cfg.yaml" 20
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
