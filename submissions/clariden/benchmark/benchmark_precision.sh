#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_precision
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/transformer/slurm/slurm-%j.out

# Benchmark the lensing/bench candidate configs at batch 16, each using its OWN precision and
# jit_compile_body (set in the config). Validates the float32-smoothing carve-out: bf16 configs
# (bfloat16/deep/wide) should now run as fast as their fp32 counterparts while using ~half the
# memory, instead of the ~10x slowdown from running the sparse smoothing in bf16.
#
# Same harness as benchmark.sh: one `srun --environment=tensorflow` step per config so each child
# runs in a fresh process (GPU memory released between runs, a fatal OOM/kernel crash is contained).
#
# Submit:  sbatch submissions/clariden/benchmark_precision.sh

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
CONFIGS_DIR="$R/y3-deep-lss/configs/transformer/lensing/bench"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/transformer"
JSONL="$OUT_DIR/bench_results.jsonl"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
PROBES="$R/y3-deep-lss/configs/probes/lensing.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

BATCH=16

mkdir -p "$OUT_DIR/slurm"
: > "$JSONL"

for cfg in "$CONFIGS_DIR"/*.yaml; do
    name="$(basename "$cfg")"
    # config-driven: precision and jit_compile_body are read from the config (not overridden)
    echo ">>> $name  batch=$BATCH  (config-driven precision + XLA) ..."
    log="$(mktemp)"
    srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
        bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --single \
            --net_config '$cfg' --batch_size $BATCH \
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
        printf '{"config": "%s", "batch_size": %s, "precision": "config", "n_tokens": "-", "pix_per_token": "-", "params_M": "-", "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
            "$name" "$BATCH" "$st" >> "$JSONL"
        echo "    $st"
    fi
    rm -f "$log"
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
