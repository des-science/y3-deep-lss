#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=mirrored_training
#SBATCH --output=./logs/mirrored_training.%j.log

# export NCCL_DEBUG=INFO

srun --cpu-bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_training.py \
    --loss_function="likelihood" \
    --dist_strategy="mirrored" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/lensing_only/mse" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet_vanilla.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml" \
    --wandb \
    --wandb_tags v6 mirrored lensing likelihood
# --wandb_notes="like a commit message?" \
# --evaluate_training_set
# --dir_model="2023-11-13_02-39-14_resnet_vanilla" \
# --restore_checkpoint
