#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_t6
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t6/slurm/slurm-%j.out

# Benchmark the bench_t6 nested-transformer configs (lighter global attention: global_blocks 4->1)
# for GPU-memory fit and step time, to pick per-config training batch sizes (fill the 120 GB GH200,
# keep NCCL comm-buffer headroom) and a step count for < 10 h of training.
#
# Sweeps default.yaml (maps+cls arch, global_blocks 1) and stem.yaml (default + patchified stem,
# one hierarchical level fewer -> much lighter) over per-config batch ladders. Also benchmarks
# maps+cls.yaml @ batch 20 as a synthetic->real calibration anchor: its real 4-GPU wall time is
# known (t5_cls job 2702990: 120000 steps in 9h22m31s = 281 ms/step effective, incl vali/ckpt).
#
# Each (config, batch) runs as its own `srun --environment=tensorflow` step (only form that
# reliably has TF in the CSCS container); a fresh process releases GPU memory between runs and
# contains OOM / kernel-launch crashes, which the loop classifies (OK / OOM / KERNEL / ERROR).
# precision (bfloat16) and jit_compile_body (true) come from each config, as in real training.

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
T6_DIR="$R/y3-deep-lss/configs/transformer/lensing/bench_t6"
ANCHOR="$R/y3-deep-lss/configs/transformer/lensing/maps+cls.yaml"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t6"
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

echo "===== bench_t6/default.yaml (global_blocks 1) ====="
for bs in 20 24 32 48 64; do
    run_one "$T6_DIR/default.yaml" "$bs"
done

echo ""
echo "===== bench_t6/stem.yaml (stem_levels 1, one hierarchical level fewer) ====="
for bs in 32 48 64 96 128; do
    run_one "$T6_DIR/stem.yaml" "$bs"
done

echo ""
echo "===== maps+cls.yaml calibration anchor (batch 20; real 4-GPU wall = 281 ms/step) ====="
run_one "$ANCHOR" 20

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
