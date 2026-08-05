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

# Standalone re-run of the inference tail of ../training.sh against an existing preds_*.h5 (no
# retrain) -- use to recover a run whose inference step failed to launch. Override OUTPUT/MODEL_DIR
# to target a specific run directory. Needs eval too? Use eval_inference.sh instead.
# Submit with --uenv-passthrough=ignore from inside a uenv session.

# --- Runtime environment ---------------------------------------------------------------------

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults ----------------------------------------------------------------------

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

PROBE="${PROBE:-lensing}"         # run dir under maps/<probe>/; ignored if OUTPUT is set directly
MODEL_DIR="${MODEL_DIR:-t1_cls}"  # the run to re-infer
RUN_NUM="${RUN_NUM:-1}"           # names the log only; there is no chain here

# --- Fixed settings ----------------------------------------------------------------------------

STRATEGY="mirrored"  # names the logs only -- inference itself is single-GPU pytorch

# --- Derived paths, configs and flags ----------------------------------------------------------

OUTPUT="${OUTPUT:-$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE}"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

FLOW_CONFIG="$MSI/configs/flow/maf.yaml"

# --- Stage 1: Inference ------------------------------------------------------------------------

# --cpu-bind=none: otherwise this 1-GPU/72-CPU sub-allocation fails to launch
srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G --cpu-bind=none \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL_DIR\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        --sample_posterior \
        --include_grid \
        --include_des \
        --include_mocks"
