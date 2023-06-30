#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=debug_training
#SBATCH --output=./logs/debug_training.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=128
# export OMP_PLACES=threads
# export OMP_PROC_BIND=close

#run the application:
srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/DESy3_fiducial_???.tfrecord" --net_config="/global/u2/a/athomsen/y3-deep-lss/configs/resnet_debug.yaml" --dir_base="/pscratch/sd/a/athomsen/run_files/v3/debug"
# srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern=${1} --net_config=${2} --dir_base=${3}

