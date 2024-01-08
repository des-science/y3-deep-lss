#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=mirrored_eval
#SBATCH --output=./logs/mirrored_eval.%j.log

srun --cpu-bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_evaluation.py \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/validation/DESy3_grid_*.tfrecord" \
    --dir_model="/pscratch/sd/a/athomsen/run_files/v6/lensing_only/mse/2024-01-05_22-04-15_resnet_vanilla"
