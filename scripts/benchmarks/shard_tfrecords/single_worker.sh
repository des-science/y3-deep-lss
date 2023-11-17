#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=single_shard
#SBATCH --output=./logs/single_shard.%j.log

srun --cpus-per-task=32 --cpu-bind=threads --gpu-bind=single:1 python shard_tfrecords.py