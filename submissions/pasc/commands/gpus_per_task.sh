#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=hvd_training
#SBATCH --output="./logs/v6/training_%j.log"
