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
#SBATCH --job-name=bench_t2_combined
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t2_combined/slurm/slurm-%j.out

# Size the combined bench_t2/b10*.yaml configs (batch 20 -> 10 to trade examples/s for optimizer
# updates at fixed wall): step time over the small-batch ladder b8/b10/b12/b16, where the fixed
# per-step costs (NCCL gradient all-reduce, optimizer, launch overhead) grow fastest as a fraction
# and the t(B) = a + b*B extrapolation from b20 (~170-180 ms projected at b10) is least trusted.
#
# Calibration anchor: ../maps+cls.yaml (b20) — its real 4-GPU wall is known from the t1_cls
# combined run (job 2774051: 130000 steps, ~294 ms/step effective incl vali/ckpt). K = 294 /
# bench_step_ms(anchor b20) converts the b8-b16 bench times to expected real wall; set n_steps in
# b10.yaml / b10_no_dropout.yaml from that before training (SIZING PROVISIONAL markers there).
#
# Same harness as benchmark_t2_embed64.sh: each (config, batch) is its own srun step so OOM/kernel
# crashes are contained and classified; precision (bfloat16) and jit_compile_body (true) come from
# the config, as in real training. Probes = combined_nla (the v17 bta-free pairing, matching what
# training.sh uses for PROBE=combined on v17/baseline).

R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark/benchmark_transformer.py"
CFG_DIR="$R/y3-deep-lss/configs/transformer/combined"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t2_combined"
JSONL="$OUT_DIR/benchmark_results.jsonl"
mkdir -p "$OUT_DIR/slurm"

MSFM="$R/multiprobe-simulation-forward-model/configs/v17/baseline.yaml"
PROBES="$R/y3-deep-lss/configs/probes/combined_nla.yaml"
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

echo "===== bench_t2/b10.yaml small-batch ladder ====="
for bs in 8 10 12 16; do
    run_one "$CFG_DIR/bench_t2/b10.yaml" "$bs"
done

echo ""
echo "===== maps+cls.yaml calibration anchor (batch 20; real 4-GPU wall = 294 ms/step) ====="
run_one "$CFG_DIR/maps+cls.yaml" 20

echo ""
echo "=== aggregating ==="
srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
    bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --aggregate --jsonl '$JSONL' --out_dir '$OUT_DIR'"
