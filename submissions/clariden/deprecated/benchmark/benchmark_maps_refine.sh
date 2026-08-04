#!/bin/bash
# Moved to deprecated/ 2026-08-04: superseded by submissions/clariden/benchmark_sweep.sh, the
# generalized form of this harness (BENCH_SCRIPT/CONFIGS_GLOB/OUT_DIR/BATCH_SIZES env vars).
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_maps_ref
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/transformer/maps/slurm/slurm-%j.out

# Refine the maps.yaml batch ceilings from benchmark_maps.sh: probe the 16<->24 OOM gap for the
# full-res probes (lensing, combined) and push the nside-256 clustering probe higher. Same
# per-(config,batch) srun-step pattern as benchmark_maps.sh.

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/transformer/maps"
JSONL="$OUT_DIR/benchmark_results_refine.jsonl"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

PROBES=(lensing clustering combined)
CONFIGS=(
    "$R/y3-deep-lss/configs/transformer/lensing/maps.yaml"
    "$R/y3-deep-lss/configs/transformer/clustering/maps.yaml"
    "$R/y3-deep-lss/configs/transformer/combined/maps.yaml"
)
# per-probe refined batch lists
BATCHES=(
    "18 20 22"
    "80 96 112"
    "18 20 22"
)

mkdir -p "$OUT_DIR/slurm"
: > "$JSONL"

for i in "${!PROBES[@]}"; do
    probe="${PROBES[$i]}"
    cfg="${CONFIGS[$i]}"
    probes_config="$R/y3-deep-lss/configs/probes/${probe}.yaml"
    for bs in ${BATCHES[$i]}; do
        echo ">>> ${probe}/maps.yaml  batch=$bs ..."
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
            if echo "$low" | grep -q 'resourceexhausted\|out of memory'; then st="OOM"
            elif echo "$low" | grep -q 'invalid configuration argument\|non-ok-status'; then st="KERNEL"
            else st="ERROR"; echo "    ERROR tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -12 | sed 's/^/      /'; fi
            printf '{"probe": "%s", "config": "%s/maps.yaml", "batch_size": %s, "peak_gb": "-", "step_ms": "-", "throughput": "-", "status": "%s"}\n' \
                "$probe" "$probe" "$bs" "$st" >> "$JSONL"
            echo "    $st"
        fi
        rm -f "$log"
    done
done

echo ""; echo "=== refined results ==="; cat "$JSONL"
