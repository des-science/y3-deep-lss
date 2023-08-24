#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=multi_noise
#SBATCH --output=./logs/multi_noise.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

#run the application:

srun --cpus-per-task=128 --cpu_bind=threads --gpu-bind=none \
    python fiducial_pipe_multi_noise.py \
