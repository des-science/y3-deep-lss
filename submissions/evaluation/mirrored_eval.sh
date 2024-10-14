#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=mirrored_eval
#SBATCH --output=./logs/mirrored_eval.%j.log

VERSION="v10"
# linear_bias, quadratic_bias
BIAS="linear_bias"

FIDU_VALI_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord"
GRID_VALI_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="mirrored" \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --grid_vali_tfr_pattern=$GRID_VALI_TFR \
    --dir_model="/pscratch/sd/a/athomsen/run_files/v10/clustering/mutual_info/2024-08-29_02-37-49_deepsphere_default"
# --evaluate_all_checkpoints
