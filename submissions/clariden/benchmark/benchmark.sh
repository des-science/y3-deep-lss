#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_transformer
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/bench_transformer/slurm/slurm-%j.out

# Benchmark the nested-transformer hyperparameter configs for GPU memory fit and step time.
#
# Each (config, batch) is run as its own `srun --environment=tensorflow` step — the only form
# that reliably has TensorFlow inside the CSCS container (a plain Python subprocess spawned
# within the container under sbatch does NOT inherit TF). Every child runs in a fresh process
# so GPU memory is released between runs and a fatal crash (OOM or a CUDA kernel-launch limit)
# is contained: the loop classifies it and moves on.
#
# Submit:  sbatch submissions/clariden/benchmark.sh
# Tune:    BATCH_SIZES="16 32 64" sbatch submissions/clariden/benchmark.sh   (space-separated)

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
CONFIGS_DIR="$R/y3-deep-lss/configs/transformer/lensing/hyperparameters"
# outputs go to scratch, not home (the home per-user quota is easily exceeded and silently
# drops the JSONL/log appends, which corrupts the results table)
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/bench_transformer"
JSONL="$OUT_DIR/benchmark_results.jsonl"
BATCH_SIZES="${BATCH_SIZES:-16 32 64}"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
PROBES="$R/y3-deep-lss/configs/probes/lensing.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

: > "$JSONL"

for cfg in "$CONFIGS_DIR"/*.yaml; do
    name="$(basename "$cfg")"
    for bs in $BATCH_SIZES; do
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
                st="KERNEL"   # CUDA grid-dim limit in the attention softmax; fits in memory, can't launch
            else
                st="ERROR"
                echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -12 | sed 's/^/      /'
            fi
            printf '{"config": "%s", "batch_size": %s, "n_tokens": "-", "pix_per_token": "-", "params_M": "-", "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
                "$name" "$bs" "$st" >> "$JSONL"
            echo "    $st"
        fi
        rm -f "$log"
    done
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
