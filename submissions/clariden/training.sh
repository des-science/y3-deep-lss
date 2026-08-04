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

# avoid a crashing task filling the /users quota with a core dump
ulimit -c 0

# W&B API key from .netrc, passed explicitly so it reaches the container
export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

# Aborts the job instead of silently letting a later stage run against stale/incomplete output
# from a stage that just failed (e.g. training.sh used to run eval+inference against an old
# preds_*.h5 after a mid-chain evaluation crash, with no indication anything was wrong).
check_stage() {
    local status=$1 stage=$2 log=$3
    if [ "$status" -ne 0 ]; then
        echo "$stage failed (exit $status) — see $log. Aborting before the next stage." >&2
        exit "$status"
    fi
}

RUN_NUM=${RUN_NUM:-1}
RESTORE_FLAG=""
if [ "$RUN_NUM" -gt 1 ]; then
    RESTORE_FLAG="--restore_checkpoint"
fi

# PROFILE=1 traces steps 800->805 (run_training.py --profile); one-off diagnostics only.
PROFILE_FLAG=""
if [ "${PROFILE:-0}" = "1" ]; then
    PROFILE_FLAG="--profile"
fi

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

# To use an older dataset, override these plus PROBE_CONFIG/CLS_PROBES_CONFIG (see below), e.g.:
#   VERSION=v16 SUBVERSION=rot_in_place PROBE_CONFIG=lensing CLS_PROBES_CONFIG=.../combined.yaml
VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

STRATEGY="mirrored"
DATA="default"

# SCALES may be overridden from the environment (e.g. SCALES=unsmoothed, SCALES=lmax_1024)
SCALES="${SCALES:-8wl,32gc}"

# PROBE: lensing / clustering / combined -- keys the run directory and wandb tag.
# PROBE_CONFIG selects configs/probes/*.yaml; defaults to the _nla variant for lensing/combined
# (v17 data). Set PROBE_CONFIG=<probe> explicitly on v16 to pick the non-nla config.
PROBE="${PROBE:-lensing}"
if [ -z "${PROBE_CONFIG:-}" ]; then
    case "$PROBE" in
        lensing|combined) PROBE_CONFIG="${PROBE}_nla" ;;
        *)                PROBE_CONFIG="$PROBE" ;;
    esac
fi

# LOSS selects the VMIM head via configs/loss/<LOSS>.yaml (vmim = flow, vmim_gmm = Gaussian
# mixture); orthogonal to ARCH/PROBE. Default: flow on every probe.
LOSS="${LOSS:-vmim}"

ARCH="${ARCH:-transformer}"

PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/${PROBE_CONFIG}.yaml"

# The rebinned-Cls precache spans ALL probe pairs, so it always uses the combined config
# regardless of PROBE (on v16 set CLS_PROBES_CONFIG=.../combined.yaml).
CLS_PROBES_CONFIG="${CLS_PROBES_CONFIG:-$REPOS/y3-deep-lss/configs/probes/combined_nla.yaml}"
# NET_CONFIG: network + training config, e.g. override to
#   $REPOS/y3-deep-lss/configs/transformer/lensing/maps.yaml (maps-only) or a bench_*/ variant.
NET_CONFIG="${NET_CONFIG:-$REPOS/y3-deep-lss/configs/transformer/${PROBE}/maps+cls.yaml}"

# NET_NAME (config basename) tags the run in wandb. MODEL_DIR (run dir under maps/<probe>/)
# defaults to it -- set MODEL_DIR explicitly for anything to keep (e.g. MODEL_DIR=t2_cls).
NET_NAME="$(basename "${NET_CONFIG%.yaml}")"
MODEL_DIR="${MODEL_DIR:-$NET_NAME}"


INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

# maps+cls runs need a rebinned-Cls calibration cache; run_training.py only reads it (aborts if
# missing), so build it here when the net_config carries a cls block and it's absent. Idempotent.
CLS_N_BINS=$(grep -E '^\s*n_bins:' "$NET_CONFIG" | head -1 | grep -oE '[0-9]+')
CLS_N_BINS=${CLS_N_BINS:-16}
CLS_CACHE="$INPUT/cls/rebinned_nb${CLS_N_BINS}_${SCALES}.h5"
if grep -qE '^\s*cls:' "$NET_CONFIG" && [ ! -f "$CLS_CACHE" ]; then
    echo "maps+cls run: Cls cache $CLS_CACHE missing — building it before training."
    srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_precache.log" \
        python $REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py \
            --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
            --probes_config="$CLS_PROBES_CONFIG" \
            --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
            --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
            --net_config="$REPOS/y3-deep-lss/configs/cls/mlp/default.yaml" \
            --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
            --data_dir="$INPUT" \
            --out_dir="$INPUT" \
            --model_name="precache" \
            --precache_only
    sleep 10
fi

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_training.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_training.py \
        --dir_base=$OUTPUT \
        --dir_model=$MODEL_DIR \
        --train_tfr_pattern=$TRAIN_TFR \
        --data_dir=$INPUT \
        --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --probes_config=$PROBES_CONFIG \
        --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
        --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
        --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
        --net_config=$NET_CONFIG \
        --dist_strategy="$STRATEGY" \
        --wandb \
        --wandb_tags "$VERSION" "$SUBVERSION" "$PROBE" "$LOSS" "$STRATEGY" "$ARCH" "$NET_NAME" "$SCALES" \
        $RESTORE_FLAG $PROFILE_FLAG
check_stage $? "Training" "${LOG}_training.log"

# SKIP_EVAL=1 stops after training and skips the evaluation + inference tail. Use it for short
# benchmark/profiling jobs where the model is undertrained and eval/inference would only waste the
# allocation. Leave unset for production runs.
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    echo "SKIP_EVAL=1: skipping evaluation and inference."
    exit 0
fi

sleep 30

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_evaluation.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_evaluation.py \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern=$TRAIN_TFR \
        --data_dir=$INPUT \
        --include_grid \
        --include_des \
        --include_mocks
check_stage $? "Evaluation" "${LOG}_evaluation.log"

sleep 30

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# --cpu-bind=none: without it, this 1-GPU/72-CPU sub-allocation fails to launch ("Unable to
# satisfy cpu bind request" -> step CANCELLED, empty log) since SLURM maps the 72 CPUs to a
# single socket's mask.
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