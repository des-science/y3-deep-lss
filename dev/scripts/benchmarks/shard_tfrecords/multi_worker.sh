#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=multi_shard
#SBATCH --output=./logs/multi_shard.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

srun --cpus-per-task=32 --cpu-bind=threads --gpu-bind=single:1 \
    python shard_tfrecords.py
