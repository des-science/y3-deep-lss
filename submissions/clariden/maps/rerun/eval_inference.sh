#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=eval_inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Runs ONLY the evaluation + inference stages of ../training.sh against an already-trained model
# (restores the latest checkpoint) -- use when training finished but eval/inference didn't run.
# See EVAL_SCOPE/LOAD_FLOW below for a mocks-only, no-flow-retrain quick check.

# --- Runtime environment ---------------------------------------------------------------------

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)  # exported so it reaches the container
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

DEEP_LSS="$REPOS/y3-deep-lss"
MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults ----------------------------------------------------------------------

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

# Eval and inference restore their config from the model dir, so only the run's location matters.
PROBE="${PROBE:-lensing_nla}"     # run dir under maps/<probe>/
MODEL_DIR="${MODEL_DIR:-t1_cls}"  # the run to re-evaluate

EVAL_SCOPE="${EVAL_SCOPE:-full}"  # mocks = --include_mocks only; full = grid+des+mocks
LOAD_FLOW="${LOAD_FLOW:-0}"       # 1 reuses the existing flow (--load_flow) instead of retraining it
RUN_NUM="${RUN_NUM:-1}"           # names the log only; there is no chain here

# --- Fixed settings ----------------------------------------------------------------------------

STRATEGY="mirrored"  # TF distribution strategy; also names the logs

# --- Derived paths, configs and flags ----------------------------------------------------------

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"
FLOW_CONFIG="$MSI/configs/flow/maf.yaml"

EVAL_SCOPE_FLAGS="--include_grid --include_des --include_mocks"
[ "$EVAL_SCOPE" = "mocks" ] && EVAL_SCOPE_FLAGS="--include_mocks"

FLOW_FLAG="--sample_posterior"
[ "$LOAD_FLOW" = "1" ] && FLOW_FLAG="--load_flow"

# Abort rather than let inference run against stale output from an evaluation that just failed.
check_stage() {
    local status=$1 stage=$2 log=$3
    if [ "$status" -ne 0 ]; then
        echo "$stage failed (exit $status) — see $log. Aborting before the next stage." >&2
        exit "$status"
    fi
}

# --- Stage 1: Evaluation -----------------------------------------------------------------------

# --dir_model passed explicitly: unlike ../training.sh's chained eval step, a standalone job has no
# ../../.env_var/id_<JOB_ID>.txt to read it from.
srun --environment=tensorflow --gpu-bind=none --output="${LOG}_evaluation.log" \
    python "$DEEP_LSS/deep_lss/apps/run_evaluation.py" \
        --dir_model="$OUTPUT/$MODEL_DIR" \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern="$TRAIN_TFR" \
        --data_dir="$INPUT" \
        $EVAL_SCOPE_FLAGS
check_stage $? "Evaluation" "${LOG}_evaluation.log"

# --- Stage 2: Inference ------------------------------------------------------------------------

sleep 30

srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G --cpu-bind=none \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL_DIR\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        $FLOW_FLAG \
        $EVAL_SCOPE_FLAGS"
