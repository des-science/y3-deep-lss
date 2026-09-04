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

# Dataset. The probes config has to match the dataset's IA model: v18/default and v16 are
# extended-NLA (bta in msfm's params list) and take the plain configs/probes/*.yaml, while v17 is
# standard-NLA and takes the bta-free *_nla ones -- on v17 set PROBE=lensing_nla and
# CLS_PROBES_CONFIG=.../combined_nla.yaml. Getting this wrong does NOT raise: an _nla config on
# extended-NLA data silently marginalizes ds, and it is only the reverse (bta on v17) that fails at
# the param-column gather. See the header of configs/probes/lensing_nla.yaml.
VERSION="${VERSION:-v18}"
SUBVERSION="${SUBVERSION:-default}"

SCALES="${SCALES:-8wl,32gc}"   # configs/scales/<SCALES>.yaml, e.g. unsmoothed, lmax_1024
LOSS="${LOSS:-vmim}"           # configs/loss/<LOSS>.yaml: vmim = flow head, vmim_gmm = mixture head
PROBE="${PROBE:-lensing}"      # configs/probes/<PROBE>.yaml; also the run dir and the wandb tag

# Cls precache only: it spans ALL probe pairs, so it uses the combined config rather than PROBE's.
CLS_PROBES_CONFIG="${CLS_PROBES_CONFIG:-$DEEP_LSS/configs/probes/combined.yaml}"

# The per-probe net configs differ in n_steps, smooth_nside and local_batch_size, so set NET_CONFIG
# together with PROBE. ARCH only tags the run in wandb -- keep it in step with NET_CONFIG.
ARCH="${ARCH:-deepsphere}"
NET_CONFIG="${NET_CONFIG:-$DEEP_LSS/configs/maps/prod/deepsphere/lensing/maps+cls.yaml}"

# Run dir under maps/<probe>/; set it explicitly for anything worth keeping, e.g. MODEL_DIR=t2_cls.
NET_NAME="$(basename "${NET_CONFIG%.yaml}")"
MODEL_DIR="${MODEL_DIR:-$NET_NAME}"

RUN_NUM="${RUN_NUM:-1}"      # position in a training_chainer.sh chain; >1 restores the checkpoint
PROFILE="${PROFILE:-0}"      # 1 traces steps 800->805 (run_training.py --profile); diagnostics only
SKIP_EVAL="${SKIP_EVAL:-0}"  # 1 stops after training -- for benchmarks whose model is throwaway

# Extra flags spliced into run_training.py, unquoted on purpose so several can be passed at once.
# The ones worth knowing: --n_steps overrides the config's step budget (also on a RESTORED run, where
# editing the repo yaml does nothing because the run reads its own configs.yaml); and
# --wall_budget_seconds trains for a fixed number of SECONDS instead, annealing the cosine to zero
# exactly when the allocation runs out. Example:
#   TRAIN_EXTRA="--wall_budget_seconds=41000"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"

# --- Fixed settings --------------------------------------------------------------------------

STRATEGY="mirrored"  # TF distribution strategy; also tags the run and names the logs
DATA="default"       # configs/data/<DATA>.yaml

# --- Derived paths, configs and flags --------------------------------------------------------

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
# Overridable so throwaway runs (SKIP_EVAL=1 sizing probes, smoke tests) can write under
# deep_lss/claude/ instead of polluting runs/. Unset => the production path, unchanged.
OUTPUT="${OUTPUT:-$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE}"
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

# The net config is composed: prod configs pull their cls: block in from maps/shared/cls_branch.yaml
# via extends:, so grepping the file itself finds nothing. Resolve it once and read the resolved
# key=value dump instead. This runs under srun because a bare python3 on a Clariden node is 3.6
# with no yaml.
RESOLVED_NET="${LOG}_net_config.txt"
srun --environment=tensorflow --gpu-bind=none --output="$RESOLVED_NET" \
    python -m deep_lss.utils.config_check resolve "$NET_CONFIG" --flat
check_stage $? "Net config resolution" "$RESOLVED_NET"

# run_training.py only reads the rebinned-Cls calibration cache and aborts if it is missing, so
# build it here when the net config carries a cls block and the cache is absent. Idempotent.
IS_MAPS_CLS=0; grep -qE '^network\.cls\.' "$RESOLVED_NET" && IS_MAPS_CLS=1
CLS_N_BINS=$(grep -E '^network\.cls\.n_bins=' "$RESOLVED_NET" | cut -d= -f2)

# A maps+cls config without an n_bins is a broken config, not a reason to guess: the cache name has
# to match what the training job will look for, so fail rather than build the wrong file. (This
# used to default to 16, which would silently point at another config's cache.)
if [ "$IS_MAPS_CLS" = "1" ] && [ -z "$CLS_N_BINS" ]; then
    echo "maps+cls run but network.cls.n_bins is unset in $NET_CONFIG — cannot name the Cls cache." >&2
    exit 1
fi
CLS_CACHE="$INPUT/cls/rebinned_nb${CLS_N_BINS}_${SCALES}.h5"

if [ "$IS_MAPS_CLS" = "1" ] && [ ! -f "$CLS_CACHE" ]; then
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
        $RESTORE_FLAG $PROFILE_FLAG $TRAIN_EXTRA
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
