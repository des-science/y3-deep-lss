#!/bin/bash
# Moved to deprecated/ 2026-08-04: superseded by submissions/clariden/benchmark_sweep.sh, the
# generalized form of this harness (BENCH_SCRIPT/CONFIGS_GLOB/OUT_DIR/BATCH_SIZES env vars).
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_resnet_ref
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/resnet/slurm/slurm-%j.out

# Refine benchmark_resnet.sh's coarse ladder. That sweep found the GCNN networks are nowhere near
# memory-bound (peak_gb << 120 GB even at batch 192) -- the real ceiling is a cuSPARSE kernel-launch
# limit (SPARSE_LIMIT, confirmed hard at batch>=256 for every config) PLUS a brittle, non-monotonic
# bug in the auto-split workaround (split_sparse_dense_matmul) that ValueErrors for some batch sizes
# (96, 192, 384) but not neighboring ones (64, 128) -- it depends on exact divisibility, not batch
# size per se. maps.yaml (lensing/clustering) confirmed clean up to batch 192; the four maps+cls /
# combined configs capped at 128 (192 hit the divisibility bug). Probe the gaps to find the true
# largest reliably-working batch per config before picking one for production.

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark/benchmark_resnet.py"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/resnet"
JSONL="$OUT_DIR/benchmark_results_refine.jsonl"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

PROBES=(lensing lensing clustering clustering combined combined)
CONFIGS=(
    "$R/y3-deep-lss/configs/deepsphere/lensing/maps.yaml"
    "$R/y3-deep-lss/configs/deepsphere/lensing/maps+cls.yaml"
    "$R/y3-deep-lss/configs/deepsphere/clustering/maps.yaml"
    "$R/y3-deep-lss/configs/deepsphere/clustering/maps+cls.yaml"
    "$R/y3-deep-lss/configs/deepsphere/combined/maps.yaml"
    "$R/y3-deep-lss/configs/deepsphere/combined/maps+cls.yaml"
)
# per-config refined batch lists: push the 128-capped configs toward 256 (the confirmed real
# SPARSE_LIMIT ceiling), push the 192-capped maps.yaml configs toward the same ceiling
BATCHES=(
    "208 224 240"
    "144 160 176 208 224 240"
    "208 224 240"
    "144 160 176 208 224 240"
    "144 160 176 208 224 240"
    "144 160 176 208 224 240"
)

mkdir -p "$OUT_DIR/slurm"
: > "$JSONL"

for i in "${!PROBES[@]}"; do
    probe="${PROBES[$i]}"
    cfg="${CONFIGS[$i]}"
    cfg_name="$(basename "$(dirname "$cfg")")/$(basename "$cfg")"
    probes_config="$R/y3-deep-lss/configs/probes/${probe}.yaml"
    for bs in ${BATCHES[$i]}; do
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
            echo "${row/\{/\{\"probe\": \"$probe\", \"cfg_name\": \"$cfg_name\", }" >> "$JSONL"
            echo "    OK"
        else
            low="$(tr 'A-Z' 'a-z' < "$log")"
            if echo "$low" | grep -q 'resourceexhausted\|out of memory'; then st="OOM"
            elif echo "$low" | grep -q 'cannot use gpu when output.shape'; then st="SPARSE_LIMIT"
            elif echo "$low" | grep -q 'invalid configuration argument\|non-ok-status'; then st="KERNEL"
            elif echo "$low" | grep -q 'evenly divid'; then st="SPLIT_DIVIDE_BUG"
            else st="ERROR"; echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -12 | sed 's/^/      /'; fi
            printf '{"probe": "%s", "cfg_name": "%s", "config": "%s", "batch_size": %s, "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
                "$probe" "$cfg_name" "$(basename "$cfg")" "$bs" "$st" >> "$JSONL"
            echo "    $st"
        fi
        rm -f "$log"
    done
done

echo ""; echo "=== refined results ==="; cat "$JSONL"
