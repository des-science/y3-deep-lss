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
#SBATCH --job-name=bench_resnet
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/resnet/slurm/slurm-%j.out

# Size local_batch_size (+ later n_steps) for the DeepSphere/ResNet GCNN maps.yaml/maps+cls.yaml
# configs (lensing, clustering, combined), applying the same recipe as the transformer sizing
# (project_transformer_bench_sizing_recipe): sweep the batch ladder per config, record peak_gb /
# step_ms, then pick the largest batch whose synthetic peak_gb stays inside the NCCL-safe band
# (empirically ~85 GB for the transformer; re-derive for the GCNN from the ladder rather than
# assuming it transfers) before computing n_steps for ~11h with the K synthetic->real factor.
#
# Each (config, batch) is its own `srun --environment=tensorflow` step (fresh process -> GPU memory
# released, an OOM/kernel-limit is contained and classified, not fatal to the sweep).
#
# Submit:  sbatch submissions/clariden/benchmark/benchmark_resnet.sh
# Tune:    BATCH_SIZES="16 24 32 48 64 96 128" sbatch submissions/clariden/benchmark/benchmark_resnet.sh

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_resnet.py"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/resnet"
JSONL="$OUT_DIR/benchmark_results.jsonl"
# GCNN memory footprint is tiny (batch 24 -> ~2-5 GB, vs the transformer's ~80 GB), so the binding
# constraint is NOT GPU memory but a cuSPARSE kernel-launch limit in the Chebyshev sparse matmul
# (deepsphere-cosmo-tf2/utils.py:split_sparse_dense_matmul): INVALID_ARGUMENT "Cannot use GPU when
# output.shape[1] * nnz(a) > 2^31" once batch_size * channels * nnz(adjacency) crosses 2^31 -- a
# hard ceiling on batch, confirmed at b=512 for lensing/maps.yaml (b=24 OK). Ladder finely to find it.
BATCH_SIZES="${BATCH_SIZES:-32 64 96 128 192 256 384 512}"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

# (probe, net_config) pairs to sweep -- maps-only and maps+cls per probe
PROBES=(lensing lensing clustering clustering combined combined)
CONFIGS=(
    "$R/y3-deep-lss/configs/deepsphere/lensing/maps.yaml"
    "$R/y3-deep-lss/configs/deepsphere/lensing/maps+cls.yaml"
    "$R/y3-deep-lss/configs/deepsphere/clustering/maps.yaml"
    "$R/y3-deep-lss/configs/deepsphere/clustering/maps+cls.yaml"
    "$R/y3-deep-lss/configs/deepsphere/combined/maps.yaml"
    "$R/y3-deep-lss/configs/deepsphere/combined/maps+cls.yaml"
)

mkdir -p "$OUT_DIR/slurm"
: > "$JSONL"

for i in "${!PROBES[@]}"; do
    probe="${PROBES[$i]}"
    cfg="${CONFIGS[$i]}"
    cfg_name="$(basename "$(dirname "$cfg")")/$(basename "$cfg")"
    probes_config="$R/y3-deep-lss/configs/probes/${probe}.yaml"
    for bs in $BATCH_SIZES; do
        echo ">>> ${cfg_name}  batch=$bs ..."
        log="$(mktemp)"
        srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
            bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --single \
                --net_config '$cfg' --batch_size $bs \
                --msfm_config '$MSFM' --probes_config '$probes_config' --scales_config '$SCALES' \
                --loss_config '$LOSS' --data_config '$DATA'" \
            > "$log" 2>&1 || true

        if grep -q '^BENCH_JSON ' "$log"; then
            row="$(grep '^BENCH_JSON ' "$log" | head -1 | sed 's/^BENCH_JSON //')"
            echo "${row/\{/\{\"probe\": \"$probe\", }" >> "$JSONL"
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
                "$probe" "$cfg_name" "$bs" "$st" >> "$JSONL"
            echo "    $st"
        fi
        rm -f "$log"
    done
done

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
