#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=mirrored_training
#SBATCH --output=./logs/mirrored_training_%j.log

# srun --cpu-bind=threads --gpu-bind=none \
#     python ../../deep_lss/apps/run_training.py \
#     --loss_function="delta" \
#     --dist_strategy="mirrored" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/DESy3_fiducial_*.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/v6/lensing_only/delta" \
#     --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
#     --net_config="configs/v6/lensing_only/resnet.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v6/linear_bias.yaml" \
#     --slurm_output="./logs/mirrored_training_$SLURM_JOB_ID.log" \
#     --wandb \
#     --wandb_tags v6 mirrored lensing delta

srun --cpu-bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_training.py \
    --loss_function="likelihood" \
    --dist_strategy="mirrored" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/lensing_only/likelihood" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v6/linear_bias.yaml" \
    --slurm_output="./logs/mirrored_training_$SLURM_JOB_ID.log" \
    --wandb \
    --wandb_tags v6 mirrored lensing likelihood
