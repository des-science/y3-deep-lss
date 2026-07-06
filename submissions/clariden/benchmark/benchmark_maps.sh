#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_maps
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/bench_transformer/maps/slurm/slurm-%j.out

# Benchmark the new per-probe transformer maps.yaml configs (lensing reference + clustering +
# combined) for single-GPU memory fit and step time, so we can pick the max comfortable per-GPU
# batch and set n_steps to fill ~10h. Mirrors benchmark.sh: each (config, batch) is its own
# `srun --environment=tensorflow` step (the only form that reliably has TF in the CSCS container),
# in a fresh process so GPU memory is released and an OOM / CUDA kernel-launch cap is contained.
#
# Submit: sbatch submissions/clariden/benchmark/benchmark_maps.sh
# Tune:   BATCH_SIZES="16 24 32 48 64" sbatch submissions/clariden/benchmark/benchmark_maps.sh

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/bench_transformer/maps"
JSONL="$OUT_DIR/benchmark_results.jsonl"
BATCH_SIZES="${BATCH_SIZES:-16 24 32 48 64}"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

# (probe, net_config) pairs to sweep
PROBES=(lensing clustering combined)
CONFIGS=(
    "$R/y3-deep-lss/configs/transformer/lensing/maps.yaml"
    "$R/y3-deep-lss/configs/transformer/clustering/maps.yaml"
    "$R/y3-deep-lss/configs/transformer/combined/maps.yaml"
)

mkdir -p "$OUT_DIR/slurm"
: > "$JSONL"

for i in "${!PROBES[@]}"; do
    probe="${PROBES[$i]}"
    cfg="${CONFIGS[$i]}"
    probes_config="$R/y3-deep-lss/configs/probes/${probe}.yaml"
    for bs in $BATCH_SIZES; do
        echo ">>> ${probe}/maps.yaml  batch=$bs ..."
        log="$(mktemp)"
        srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
            bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --single \
                --net_config '$cfg' --batch_size $bs \
                --msfm_config '$MSFM' --probes_config '$probes_config' --scales_config '$SCALES' \
                --loss_config '$LOSS' --data_config '$DATA'" \
            > "$log" 2>&1 || true

        if grep -q '^BENCH_JSON ' "$log"; then
            # tag the row with the probe so the aggregate table is unambiguous
            row="$(grep '^BENCH_JSON ' "$log" | head -1 | sed 's/^BENCH_JSON //')"
            echo "${row/\{/\{\"probe\": \"$probe\", }" >> "$JSONL"
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
            printf '{"probe": "%s", "config": "%s/maps.yaml", "batch_size": %s, "n_tokens": "-", "pix_per_token": "-", "params_M": "-", "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
                "$probe" "$probe" "$bs" "$st" >> "$JSONL"
            echo "    $st"
        fi
        rm -f "$log"
    done
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
