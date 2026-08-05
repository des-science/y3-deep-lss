#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Maps-domain train+eval+infer in one 12 h job. Chain several via training_chainer.sh; recover a run
# whose eval/inference didn't complete from rerun/; benchmarks/ and experiments/ sit alongside.

# --- Runtime environment ---------------------------------------------------------------------

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)  # exported so it reaches the container
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

DEEP_LSS="$REPOS/y3-deep-lss"
MSFM="$REPOS/multiprobe-simulation-forward-model"
MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults (set from the environment, e.g. by training_chainer.sh) -------------

# Dataset; on v16 also set PROBE=lensing and CLS_PROBES_CONFIG=.../combined.yaml (no _nla configs).
VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

SCALES="${SCALES:-8wl,32gc}"   # configs/scales/<SCALES>.yaml, e.g. unsmoothed, lmax_1024
LOSS="${LOSS:-vmim}"           # configs/loss/<LOSS>.yaml: vmim = flow head, vmim_gmm = mixture head
PROBE="${PROBE:-lensing_nla}"  # configs/probes/<PROBE>.yaml; also the run dir and the wandb tag

# Cls precache only: it spans ALL probe pairs, so it uses the combined config rather than PROBE's.
CLS_PROBES_CONFIG="${CLS_PROBES_CONFIG:-$DEEP_LSS/configs/probes/combined_nla.yaml}"

# The per-probe net configs differ in n_steps, smooth_nside and local_batch_size, so set NET_CONFIG
# together with PROBE. ARCH only tags the run in wandb -- keep it in step with NET_CONFIG.
ARCH="${ARCH:-deepsphere}"
NET_CONFIG="${NET_CONFIG:-$DEEP_LSS/configs/deepsphere/lensing/maps+cls.yaml}"

# Run dir under maps/<probe>/; set it explicitly for anything worth keeping, e.g. MODEL_DIR=t2_cls.
NET_NAME="$(basename "${NET_CONFIG%.yaml}")"
MODEL_DIR="${MODEL_DIR:-$NET_NAME}"

RUN_NUM="${RUN_NUM:-1}"      # position in a training_chainer.sh chain; >1 restores the checkpoint
PROFILE="${PROFILE:-0}"      # 1 traces steps 800->805 (run_training.py --profile); diagnostics only
SKIP_EVAL="${SKIP_EVAL:-0}"  # 1 stops after training -- for benchmarks whose model is throwaway

# --- Fixed settings --------------------------------------------------------------------------

STRATEGY="mirrored"  # TF distribution strategy; also tags the run and names the logs
DATA="default"       # configs/data/<DATA>.yaml

# --- Derived paths, configs and flags --------------------------------------------------------

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

MSFM_CONFIG="$MSFM/configs/$VERSION/$SUBVERSION.yaml"
PROBES_CONFIG="$DEEP_LSS/configs/probes/${PROBE}.yaml"
SCALES_CONFIG="$DEEP_LSS/configs/scales/${SCALES}.yaml"
LOSS_CONFIG="$DEEP_LSS/configs/loss/${LOSS}.yaml"
DATA_CONFIG="$DEEP_LSS/configs/data/${DATA}.yaml"
FLOW_CONFIG="$MSI/configs/flow/maf.yaml"

RESTORE_FLAG=""; [ "$RUN_NUM" -gt 1 ] && RESTORE_FLAG="--restore_checkpoint"
PROFILE_FLAG=""; [ "$PROFILE" = "1" ] && PROFILE_FLAG="--profile"

# Abort rather than let a stage run against stale output from one that just failed.
check_stage() {
    local status=$1 stage=$2 log=$3
    if [ "$status" -ne 0 ]; then
        echo "$stage failed (exit $status) — see $log. Aborting before the next stage." >&2
        exit "$status"
    fi
}

# --- Stage 0: Cls precache (maps+cls runs only) ----------------------------------------------

# run_training.py only reads the rebinned-Cls calibration cache and aborts if it is missing, so
# build it here when the net config carries a cls block and the cache is absent. Idempotent.
CLS_N_BINS=$(grep -E '^\s*n_bins:' "$NET_CONFIG" | head -1 | grep -oE '[0-9]+')
CLS_CACHE="$INPUT/cls/rebinned_nb${CLS_N_BINS:-16}_${SCALES}.h5"

if grep -qE '^\s*cls:' "$NET_CONFIG" && [ ! -f "$CLS_CACHE" ]; then
    echo "maps+cls run: Cls cache $CLS_CACHE missing — building it before training."
    srun --environment=tensorflow --gpu-bind=none --output="${LOG}_precache.log" \
        python "$DEEP_LSS/deep_lss/apps/run_cls_training+evaluation.py" \
            --msfm_config="$MSFM_CONFIG" \
            --probes_config="$CLS_PROBES_CONFIG" \
            --scales_config="$SCALES_CONFIG" \
            --loss_config="$LOSS_CONFIG" \
            --net_config="$DEEP_LSS/configs/cls/mlp/default.yaml" \
            --data_config="$DATA_CONFIG" \
            --data_dir="$INPUT" \
            --out_dir="$INPUT" \
            --model_name="precache" \
            --precache_only
    sleep 10
fi

# --- Stage 1: Training -----------------------------------------------------------------------

srun --environment=tensorflow --gpu-bind=none --output="${LOG}_training.log" \
    python "$DEEP_LSS/deep_lss/apps/run_training.py" \
        --dir_base="$OUTPUT" \
        --dir_model="$MODEL_DIR" \
        --train_tfr_pattern="$TRAIN_TFR" \
        --data_dir="$INPUT" \
        --msfm_config="$MSFM_CONFIG" \
        --probes_config="$PROBES_CONFIG" \
        --scales_config="$SCALES_CONFIG" \
        --loss_config="$LOSS_CONFIG" \
        --data_config="$DATA_CONFIG" \
        --net_config="$NET_CONFIG" \
        --dist_strategy="$STRATEGY" \
        --wandb \
        --wandb_tags "$VERSION" "$SUBVERSION" "$PROBE" "$LOSS" "$STRATEGY" "$ARCH" "$NET_NAME" "$SCALES" \
        $RESTORE_FLAG $PROFILE_FLAG
check_stage $? "Training" "${LOG}_training.log"

if [ "$SKIP_EVAL" = "1" ]; then
    echo "SKIP_EVAL=1: skipping evaluation and inference."
    exit 0
fi

# --- Stage 2: Evaluation ---------------------------------------------------------------------

sleep 30

srun --environment=tensorflow --gpu-bind=none --output="${LOG}_evaluation.log" \
    python "$DEEP_LSS/deep_lss/apps/run_evaluation.py" \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern="$TRAIN_TFR" \
        --data_dir="$INPUT" \
        --include_grid \
        --include_des \
        --include_mocks
check_stage $? "Evaluation" "${LOG}_evaluation.log"

# --- Stage 3: Inference ----------------------------------------------------------------------

sleep 30

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
