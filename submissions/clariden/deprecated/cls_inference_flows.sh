#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_inference_flows
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Run all four flow configs in parallel (one per GPU) for a single probe.
# Override the probe at submission time: PROBE=clustering sbatch cls_inference_flows.sh

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
PROBE="combined"
MODEL_NAME="v26"

# CONFIG_SET="baseline"
# CONFIG_SET="early_stop_exp"
# CONFIG_SET="plateau"
# CONFIG_SET="maf_cosine"
CONFIG_SET="maf_wd"

# RUN_VERSION=""
RUN_VERSION="v5"

CONFIG_DIR="$REPOS/multiprobe-simulation-inference/configs/flow/$CONFIG_SET"
FLOW_LABEL_SUFFIX="${RUN_VERSION:+${RUN_VERSION}_}${CONFIG_SET}"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$LOG")"

# FLOW_CONFIGS=("sigmoid" "spline" "maf" "lipschitz")
# FLOW_CONFIGS=("maf" "maf_short" "maf_small" "maf_large")
# FLOW_CONFIGS=("maf" "maf_short" "maf_narrow" "maf_small")
# FLOW_CONFIGS=("maf" "maf_short" "maf_narrow" "maf_wd")
FLOW_CONFIGS=("maf" "maf_1e-4" "maf_1e-3" "maf_1e-2")

# FLOW_CONFIGS=("sigmoid" "spline" "maf" "lipschitz")
# FLOW_CONFIGS=("spline" "spline_wd" "maf" "maf_wd")

for FLOW_CONFIG in "${FLOW_CONFIGS[@]}"; do
    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${LOG}_${FLOW_CONFIG}.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && \
            python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$MODEL_NAME\" \
                --flow_config=\"$CONFIG_DIR/${FLOW_CONFIG}.yaml\" \
                --flow_label="${FLOW_LABEL_SUFFIX}_${FLOW_CONFIG}" \
                --sample_posterior \
                --include_grid" &
done
                # --n_flows=8 \

wait

# Heterogeneous ensemble (one member per architecture instead of seed-clones of one) -- usually a
# stronger lever on posterior overconfidence than --n_flows seed-clones, since members disagree more.
# Run it as a single srun with --flow_configs (mutually exclusive with --flow_config); --n_flows then
# replicates each listed config (total members = n_configs * n_flows). Drops lipschitz, which cannot be
# reloaded from a checkpoint (state_dict-only); fine here since this trains+samples in one job.
# srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
#     --uenv=pytorch/v2.9.1:v2 --view=default \
#     --output="${LOG}_hetero.log" \
#     bash -c "source ~/dlss/torch_env/bin/activate && \
#         python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
#             --out_dir=\"$OUTPUT\" \
#             --model_name=\"$MODEL_NAME\" \
#             --flow_configs \"$CONFIG_DIR/sigmoid.yaml\" \"$CONFIG_DIR/spline.yaml\" \"$CONFIG_DIR/maf.yaml\" \
#             --flow_label=\"${FLOW_LABEL_SUFFIX}_hetero\" \
#             --sample_posterior \
#             --include_grid \
#             --n_flows=1 \
#             --include_mocks"
