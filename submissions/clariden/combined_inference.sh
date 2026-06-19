#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=combined_inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
# SUBVERSION="default"
# SUBVERSION="no_sc"
SUBVERSION="rot_in_place"

PROBES=("lensing" "clustering" "combined")

# Map-level model (trained by training.sh, written under runs/.../maps/$PROBE/$MAPS_MODEL)
MAPS_MODEL="v6"

# Power-spectrum / Cls-level model (trained by cls_training.sh, written under runs/.../cls/$PROBE/$CLS_MODEL_NAME)
CLS_MODEL_NAME="v22"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

for PROBE in "${PROBES[@]}"; do
    MAPS_OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
    CLS_OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"

    LOG="$MAPS_OUTPUT/$MAPS_MODEL/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$LOG")"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${LOG}_combined_flow_inference.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
            --out_dir=\"$MAPS_OUTPUT\" \
            --model_name=\"$MAPS_MODEL\" \
            --out_dir_2=\"$CLS_OUTPUT\" \
            --model_name_2=\"$CLS_MODEL_NAME\" \
            --flow_label=\"combined_maps_cls\" \
            --flow_config=\"$FLOW_CONFIG\" \
            --sample_posterior \
            --include_grid \
            --include_des \
            --include_mocks" &
done

wait
