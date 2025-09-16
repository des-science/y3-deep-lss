#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=hybrid_training
#SBATCH --output=./logs/hybrid_training.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=32
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

# srun --cpu_bind=threads --cpus-per-task=32 --gpu-bind=single:1 \
# srun --ntasks=4 --cpus-per-task=32 --cpu-bind=threads --gpus=4 --gpu-bind=single:1 \

export NCCL_DEBUG=INFO

srun --cpus-per-task=128 --cpu-bind=threads --gpu-bind=single:1 \
    python ../../deep_lss/apps/run_training.py \
    --fidu_train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/fiducial/DESy3_fiducial_???.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_???.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/grid/DESy3_grid_???.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v5/lensing_only" \
    --dlss_config="configs/v5/lensing_only/dlss_config.yaml" \
    --net_config="configs/v5/lensing_only/resnet_vanilla.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml"
# --dir_model="2023-11-13_02-39-14_resnet_vanilla" \
# --restore_checkpoint
