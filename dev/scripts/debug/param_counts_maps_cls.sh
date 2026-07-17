#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=param_counts_maps_cls
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/smoke_multires_gcnn/slurm-%j.out

# parameter counts for the current deepsphere maps+cls configs (all three probes);
# see param_counts_maps_cls.py next to this file

ulimit -c 0

REPOS="/users/athomsen/dlss/repos"
mkdir -p /iopsstor/scratch/cscs/athomsen/deep_lss/runs/smoke_multires_gcnn

srun --environment=tensorflow --gpu-bind=none \
    python -u $REPOS/y3-deep-lss/dev/scripts/debug/param_counts_maps_cls.py
