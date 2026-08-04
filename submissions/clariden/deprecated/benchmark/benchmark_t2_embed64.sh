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
#SBATCH --job-name=bench_t2_embed64
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t2/slurm/slurm-%j.out

# Size the bench_t2/embed64.yaml clustering config (base_embed_dim 32 -> 64, the lensing-matched
# stem width): GPU-memory fit and step time over a batch ladder, to pick the training batch (largest
# inside the ~85 GB NCCL-safe band, then judged against the loader ceiling) and n_steps for ~10.5 h.
#
# Calibration anchor: ../maps+cls.yaml (base 32) @ batch 48 — its real 4-GPU wall is known
# (t1_cls job 2774050: 120000 steps in 10h03m30s = 302 ms/step effective, incl vali/ckpt), and its
# real loader budget is known (data_time 0.188 s ~= 158 examples/s at native-res reads). The
# synthetic->real factor K = 302 / bench_step_ms(anchor) converts embed64 bench times to expected
# real wall; if the implied examples/s exceeds ~158 the run stays I/O-bound and the base-32 read
# ceiling applies instead.
#
# Same harness as benchmark_t6.sh: each (config, batch) is its own srun step so OOM/kernel crashes
# are contained and classified (OK / OOM / KERNEL / ERROR); precision (bfloat16) and
# jit_compile_body (true) come from the config, as in real training.

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
CFG_DIR="$R/y3-deep-lss/configs/transformer/clustering"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t2"
JSONL="$OUT_DIR/benchmark_results.jsonl"
mkdir -p "$OUT_DIR/slurm"

MSFM="$R/multiprobe-simulation-forward-model/configs/v17/baseline.yaml"
PROBES="$R/y3-deep-lss/configs/probes/clustering.yaml"
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

echo "===== bench_t2/embed64.yaml (base_embed_dim 64, lensing-matched widths) ====="
for bs in 16 24 32 48 64 96; do
    run_one "$CFG_DIR/bench_t2/embed64.yaml" "$bs"
done

echo ""
echo "===== maps+cls.yaml calibration anchor (base 32, batch 48; real 4-GPU wall = 302 ms/step) ====="
run_one "$CFG_DIR/maps+cls.yaml" 48

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
