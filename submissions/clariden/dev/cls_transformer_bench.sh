#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_tf_bench
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/slurm/slurm-%j.out

# Phase 1 of the Cls transformer hyperparameter search: a SIZE LADDER, training only.
#
# Rank the rungs on vali_nmse_cosmo (per-parameter-normalized posterior-mean MSE over the
# cosmological parameters). Do NOT rank on vali_loss: it is the VMIM bound, which equals
# I(theta;s) - E[KL(p(theta|s) || q(theta|s))], and the head-gap term moves with the learned summary
# geometry -- the 2026-07-12 head ablation in configs/loss/vmim.yaml shifted the bound by 0.45 nats
# with mock FoM flat at 526-547. Inference/FoM is run separately on the surviving rungs.
#
# The ladder spans 58x in parameters (13.8k -> 798k) and 60x in FLOPs (0.04x -> 2.41x the MLP), with
# mlp_ref as the baseline. Everything but (d_model, num_layers) is fixed, so this is one factor.
# Start at the bottom; only climb if the metric actually improves.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v17"
SUBVERSION="baseline"
# v17 is standard-NLA: bta does not exist in the label table, so the bta-containing probes configs
# (lensing.yaml) fail at the param-column gather. lensing_nla.yaml is the matching one.
PROBE="lensing_nla"

NET_DIR="cls/transformer/v1_bench"
NET_CONFIGS=("d32_L1" "d64_L2" "d128_L2" "d128_L4" "mlp_ref")

LOSS="vmim"
SCALES="8wl,32gc"
DATA="default"
BASE_MODEL_NAME="v1_bench"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"

# SLURM --output goes to scratch, not home: the home VAST quota silently drops writes mid-job.
mkdir -p "$MYSCRATCH/deep_lss/slurm"

# The hard_rebinned cache is net/loss-independent and all rungs share scale_cut + cls_n_bins, so one
# precache serves all of them. Idempotent: a no-op (~40 s) when the file is already there, which it
# is for v17/baseline @ nb16 8wl,32gc.
LOG_PRECACHE="$OUTPUT/precache/logs/${SLURM_JOB_ID}_precache"
mkdir -p "$(dirname "$LOG_PRECACHE")"
srun -N1 --ntasks-per-node=1 --exclusive --cpus-per-task=288 --mem=450G \
    --environment=tensorflow \
    --output="${LOG_PRECACHE}.log" \
    python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
        --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --probes_config="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml" \
        --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
        --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
        --net_config="$REPOS/y3-deep-lss/configs/${NET_DIR}/${NET_CONFIGS[0]}.yaml" \
        --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
        --data_dir="$INPUT" \
        --out_dir="$INPUT" \
        --model_name="precache" \
        --precache_only

# One GPU per rung. There are 5 rungs and 4 GPUs, so the last one starts as a slot frees up; with
# early stopping each rung is minutes, so the tail cost is small.
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
done

wait
