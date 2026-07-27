#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_tf_seeds
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/jobs/misc/slurm-%j.out

# Phase 1b: SEED REPLICATION of the size ladder's top rungs.
#
# The seed-42 ladder (job 2770501) separated mlp_ref (0.4447), d128_L2 (0.4460) and d128_L4 (0.4466)
# by only 0.3-0.4% in vali_nmse_cosmo -- far too close to call from one seed, especially since the
# early-stopping step is itself stochastic (they stopped at 58k/80k/62k). This run adds seeds 43 and
# 44 so the spread WITHIN a config can be compared against the spread BETWEEN configs. Without that
# noise floor, ranking the top three is numerology.
#
# d64_L2 is included as a negative control: at seed 42 it was 0.4652, ~4% worse than the top three.
# If that gap is real it should survive replication; if it does not, the whole ladder is flat and the
# architecture question is moot.
#
# d32_L1 is dropped -- at 0.4716 it is the clear loser and needs no further resolution.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v17"
SUBVERSION="baseline"
PROBE="lensing_nla"

NET_DIR="cls/transformer/v1_bench/seeds"
NET_CONFIGS=("d64_L2_s43" "d128_L2_s43" "d128_L4_s43" "mlp_ref_s43"
             "d64_L2_s44" "d128_L2_s44" "d128_L4_s44" "mlp_ref_s44")

LOSS="vmim"
SCALES="8wl,32gc"
DATA="default"
BASE_MODEL_NAME="v1_seeds"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"

mkdir -p "$MYSCRATCH/deep_lss/claude/jobs/misc"

# The hard_rebinned cache was written by job 2770501 and is net/seed-independent, so no precache step.
# 8 runs over 4 GPUs = 2 rounds; each rung early-stopped in 3-6 min at seed 42, so ~15 min total.
# Cap concurrency at 4 so the second round starts only as slots free.
i=0
for NET in "${NET_CONFIGS[@]}"; do
    MODEL_NAME="${BASE_MODEL_NAME}_${NET}"
    LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$LOG")"

    (
        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --environment=tensorflow \
            --output="${LOG}_training.log" \
            python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
                --probes_config="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml" \
                --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
                --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
                --net_config="$REPOS/y3-deep-lss/configs/${NET_DIR}/${NET}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --data_dir="$INPUT" \
                --out_dir="$OUTPUT" \
                --model_name="$MODEL_NAME" \
                --include_grid
    ) &

    i=$((i + 1))
    if [ $((i % 4)) -eq 0 ]; then
        wait
    fi
done

wait
