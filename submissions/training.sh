#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --qos=regular
#SBATCH --job-name=training
#SBATCH --time=08:00:00
#SBATCH --account=des_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=./log_training/training.%j.log

#OpenMP settings:
export OMP_NUM_THREADS=64

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1 
srun --cpu_bind=cores --gpus 4 --gpu-bind=none python ../deep_lss/apps/run_training.py --tfr_pattern=${1} --net_config=${2} --dir_base=${3}