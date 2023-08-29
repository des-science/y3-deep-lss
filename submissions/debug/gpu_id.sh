#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=gpu_id
#SBATCH --output=./logs/gpu_id.%j.log

export SLURM_CPU_BIND="cores"

# srun echo $SLURM_LOCALID $SLURM_PROCID $SLURM_NTASKS_PER_NODE
srun task_script.sh

