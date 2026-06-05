#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=evaluation
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

REPOS="/users/athomsen/dlss/repos"
STRATEGY="mirrored"
FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/default.yaml"

VERSION="v16"
# SUBVERSION="default"
# SUBVERSION="no_sc"
SUBVERSION="rot_in_place"

MODEL="v2"
# MODEL="40Mpc"

# PROBE="lensing"
PROBE="clustering"
# PROBE="cross"
# PROBE="combined"

MYSCRATCH="/iopsstor/scratch/cscs/athomsen"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL/logs/"$STRATEGY"_"$SLURM_JOB_ID""

GRID_EVAL_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_evaluation.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_evaluation.py\
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
        --data_dir=$INPUT \
        --dir_model="$OUTPUT/$MODEL" \
        --include_grid \
        --include_des \
        --include_mocks

sleep 30

srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output=""$LOG"_flow_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --include_grid \
        --include_des \
        --include_mocks"
