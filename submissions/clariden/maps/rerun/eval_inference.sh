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

# Runs ONLY the evaluation + inference stages of ../training.sh against an already-trained
# model (restores the latest checkpoint) -- use when training finished but eval/inference
# didn't run. See EVAL_SCOPE/LOAD_FLOW below for a mocks-only, no-flow-retrain quick check.

export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

RUN_NUM=${RUN_NUM:-1}

# Aborts instead of letting inference silently run against a stale preds_*.h5 if evaluation fails.
check_stage() {
    local status=$1 stage=$2 log=$3
    if [ "$status" -ne 0 ]; then
        echo "$stage failed (exit $status) — see $log. Aborting before the next stage." >&2
        exit "$status"
    fi
}

# EVAL_SCOPE=mocks: --include_mocks only (default "full" = grid+des+mocks).
EVAL_SCOPE="${EVAL_SCOPE:-full}"
# LOAD_FLOW=1: inference reuses the existing flow (--load_flow) instead of retraining it.
LOAD_FLOW="${LOAD_FLOW:-0}"

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="${VERSION:-v16}"
SUBVERSION="${SUBVERSION:-rot_in_place}"

STRATEGY="mirrored"

# eval/inference restore their config from the model dir, so only PROBE (path) and MODEL matter
PROBE="${PROBE:-lensing}"
MODEL="${MODEL:-t1}"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

EVAL_SCOPE_FLAGS="--include_grid --include_des --include_mocks"
if [ "$EVAL_SCOPE" = "mocks" ]; then
    EVAL_SCOPE_FLAGS="--include_mocks"
fi

# --dir_model passed explicitly: unlike ../training.sh's chained eval step, there's no
# ../../.env_var/id_<JOB_ID>.txt to read it from in a standalone job.
srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_evaluation.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_evaluation.py \
        --dir_model="$OUTPUT/$MODEL" \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern=$TRAIN_TFR \
        --data_dir=$INPUT \
        $EVAL_SCOPE_FLAGS
check_stage $? "Evaluation" "${LOG}_evaluation.log"

sleep 30

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

INFERENCE_FLOW_FLAG="--sample_posterior"
if [ "$LOAD_FLOW" = "1" ]; then
    INFERENCE_FLOW_FLAG="--load_flow"
fi

srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output=""$LOG"_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        $INFERENCE_FLOW_FLAG \
        $EVAL_SCOPE_FLAGS"
