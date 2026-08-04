#!/bin/bash
# Moved to deprecated/ 2026-08-04: hardcoded stale VERSION=v16/rot_in_place and a dead model
# name (v29_vmim_fac1), no env-var overrides at all. Same shape as the already-deprecated
# inference_probes.sh/inference_steps.sh -- for a one-off flow-config sweep today, copy this
# pattern with current VERSION/SUBVERSION and a live model_name instead of resurrecting it.
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_inference_flows
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
PROBE="combined"

SUMMARY_TYPE="cls"
# SUMMARY_TYPE="maps"

# MODEL_NAME="v26"
MODEL_NAME="v29_vmim_fac1"
# MODEL_NAME="v6_cls"

# CONFIG_SET="plateau"
# CONFIG_SET="cosine"
# CONFIG_SET="maf_cosine"
# CONFIG_SET="maf_regu"
CONFIG_SET="maf_convergence"

RUN_VERSION="v9"

CONFIG_DIR="$REPOS/multiprobe-simulation-inference/configs/flow/$CONFIG_SET"
FLOW_LABEL_SUFFIX="${RUN_VERSION:+${RUN_VERSION}_}${CONFIG_SET}"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/$SUMMARY_TYPE/$PROBE"
LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$LOG")"

# FLOW_CONFIGS=("sigmoid" "lipschitz" "spline" "maf")
# FLOW_CONFIGS=("maf" "maf_short" "maf_narrow" "maf_small")
# FLOW_CONFIGS=("maf" "maf_cond" "maf_dropout" "maf_wd")
FLOW_CONFIGS=("maf_cosine_100" "maf_cosine_300" "maf_cosine_500" "maf_plateau")
# FLOW_CONFIGS=("maf" "maf_1e-4" "maf_1e-3" "maf_1e-2")
# FLOW_CONFIGS=("maf" "maf_1e-4" "maf_1e-3" "maf_dropout")
# FLOW_CONFIGS=("maf" "maf_wd" "maf_dropout" "maf_wd+dropout")

for FLOW_CONFIG in "${FLOW_CONFIGS[@]}"; do
    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${LOG}_${FLOW_CONFIG}_inference.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && \
            python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$MODEL_NAME\" \
                --flow_config=\"$CONFIG_DIR/${FLOW_CONFIG}.yaml\" \
                --flow_label="${FLOW_LABEL_SUFFIX}_${FLOW_CONFIG}" \
                --n_flows=4 \
                --sample_posterior \
                --include_grid" &
done

wait
