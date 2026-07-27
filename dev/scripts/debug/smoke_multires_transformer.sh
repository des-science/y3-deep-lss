#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=smoke_multires_transformer
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/tmp/smoke_multires_gcnn/slurm-%j.out

# smoke test for the multi-resolution transformer path (HealpixMultiResMapEncoder), added with
# the shared MultiResEncoderMixin refactor; see smoke_multires_transformer.py next to this file

ulimit -c 0

REPOS="/users/athomsen/dlss/repos"
mkdir -p /iopsstor/scratch/cscs/athomsen/deep_lss/claude/tmp/smoke_multires_gcnn

srun --environment=tensorflow --gpu-bind=none \
    python -u $REPOS/y3-deep-lss/dev/scripts/debug/smoke_multires_transformer.py
