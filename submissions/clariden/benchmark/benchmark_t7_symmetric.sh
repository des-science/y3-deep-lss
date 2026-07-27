#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bench_t7_sym
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t7/slurm/slurm-%j.out

# symmetric.yaml at b20 lands at 89.6 GB (above the ~85 GB NCCL-safe band). Find the largest
# batch that stays inside the band (b16, b18) so we can re-size its n_steps for ~11 h at 4x GH200.
R="/users/athomsen/dlss/repos"
SCRIPT="$R/y3-deep-lss/deep_lss/apps/benchmark_transformer.py"
CFG="$R/y3-deep-lss/configs/transformer/lensing/bench_t7/symmetric.yaml"
OUT_DIR="/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/t7"
JSONL="$OUT_DIR/symmetric_batch.jsonl"
mkdir -p "$OUT_DIR/slurm"

MSFM="$R/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml"
PROBES="$R/y3-deep-lss/configs/probes/lensing.yaml"
SCALES="$R/y3-deep-lss/configs/scales/8wl,32gc.yaml"
LOSS="$R/y3-deep-lss/configs/loss/vmim.yaml"
DATA="$R/y3-deep-lss/configs/data/default.yaml"

: > "$JSONL"
for bs in 16 18; do
    echo ">>> symmetric.yaml batch=$bs ..."
    log="$(mktemp)"
    srun --overlap --environment=tensorflow --gpu-bind=none --ntasks=1 \
        bash -c "source ~/dlss/tf_env/bin/activate && python '$SCRIPT' --single \
            --net_config '$CFG' --batch_size $bs \
            --msfm_config '$MSFM' --probes_config '$PROBES' --scales_config '$SCALES' \
            --loss_config '$LOSS' --data_config '$DATA'" \
        > "$log" 2>&1 || true
    if grep -q '^BENCH_JSON ' "$log"; then
        grep '^BENCH_JSON ' "$log" | head -1 | sed 's/^BENCH_JSON //' | tee -a "$JSONL"
    else
        echo "    non-OK; tail:"; grep -viE 'ptx85|gpu_timer' "$log" | tail -8 | sed 's/^/      /'
    fi
    rm -f "$log"
done
