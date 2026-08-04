#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=evaluation

VERSION="v16"
SUBVERSION="default"
# MODEL="v1"
MODEL="v2"

PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"

BASE="/pscratch/sd/a/athomsen/deep_lss/$VERSION/$SUBVERSION/maps/$PROBE"
OUTPUT="/$BASE/$MODEL/logs/separate_"$STRATEGY"_"$SLURM_JOB_ID""

GRID_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_inference.log" \
    python ../../deep_lss/apps/run_evaluation.py \
        --dist_strategy="mirrored" \
        --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
        --dir_model="/pscratch/sd/a/athomsen/deep_lss/$VERSION/$SUBVERSION/maps/$PROBE/$MODEL"

# python ../../deep_lss/apps/run_evaluation.py \
#     --dist_strategy="mirrored" \
#     --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
#     --dir_model="/pscratch/sd/a/athomsen/deep_lss/$VERSION/$SUBVERSION/maps/$PROBE/$MODEL"