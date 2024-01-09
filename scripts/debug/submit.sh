#!/bin/bash
#SBATCH --account=des
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --job-name=output_test
#SBATCH --output=./logs/test%j.log

srun --cpu-bind=threads --gpu-bind=none python slurm_output.py --sbatch_output=./logs/test$SLURM_JOB_ID.log
