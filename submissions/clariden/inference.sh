#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Standalone re-run of the inference tail of training.sh, against an existing preds_*.h5 (no retrain).
# Use it to recover a run whose training+eval finished but whose inference step failed to launch.
# Override OUTPUT / MODEL_DIR from the environment to point at the run directory:
#   OUTPUT=/iopsstor/.../runs/v17/baseline/maps/clustering MODEL_DIR=t1_cls sbatch inference.sh
# (submit with --uenv-passthrough=ignore from inside a uenv session).

ulimit -c 0

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"
PROBE="${PROBE:-lensing}"
MODEL_DIR="${MODEL_DIR:-t1_cls}"
RUN_NUM="${RUN_NUM:-1}"
STRATEGY="mirrored"

OUTPUT="${OUTPUT:-$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE}"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# --cpu-bind=none: the 1-GPU / 72-CPU sub-allocation otherwise fails to launch with
# "Unable to satisfy cpu bind request" (see the matching note in training.sh).
srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G --cpu-bind=none \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output=""$LOG"_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL_DIR\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        --sample_posterior \
        --include_grid \
        --include_des \
        --include_mocks"
