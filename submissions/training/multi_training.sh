#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=multi_training
#SBATCH --output=./logs/multi_training.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

export SLURM_CPU_BIND="cores"

srun python ../../deep_lss/apps/run_training.py \
    --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/fiducial/DESy3_fiducial_???.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/fiducial/validation/DESy3_fiducial_???.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/grid/DESy3_grid_???.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v3/multi_worker" \
    --dlss_config="configs/probe_combination/dlss_config.yaml" \
    --net_config="configs/probe_combination/resnet_small.yaml"
