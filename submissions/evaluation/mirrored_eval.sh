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

VERSION="v13"
# BIAS="linear_bias"
BIAS="extended"

VERSION="v13"

SUBVERSION=$BIAS
# SUBVERSION=""$BIAS"_octant"
# SUBVERSION=""$BIAS"_hard_cut"

FIDU_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord"
GRID_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="mirrored" \
    --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
    --dir_model="/pscratch/sd/a/athomsen/run_files/v13/extended/lensing/mutual_info/2025-01-11_07-17-28_deepsphere_default" \
    --evaluate_all_checkpoints
# --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
