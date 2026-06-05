#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Run all four flow configs in parallel (one per GPU) for a single probe.
# Override the probe at submission time: PROBE=clustering sbatch cls_inference.sh

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"

PROBE="clustering"
MODEL_NAME="v6"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$LOG")"

FLOW_CONFIGS=("default" "smaller" "old_default" "lipschitz")

for FLOW_CONFIG in "${FLOW_CONFIGS[@]}"; do
    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${LOG}_${FLOW_CONFIG}.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && \
            python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$MODEL_NAME\" \
                --flow_config=\"$REPOS/multiprobe-simulation-inference/configs/flow/${FLOW_CONFIG}.yaml\" \
                --flow_label=\"${FLOW_CONFIG}\" \
                --include_grid \
                --include_des \
                --include_mocks" &
done

wait
