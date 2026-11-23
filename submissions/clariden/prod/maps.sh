#!/bin/bash
#SBATCH --account=lp184
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=maps
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Map-level network: train, evaluate, infer. VERSION picks the dataset (no default); the run defaults to
# the production run maps_gcnn/<probe>/v1. Training over an existing run needs OVERWRITE=1, which deletes
# that run first; eval or infer on a run that already holds predictions need OVERWRITE=1 too. Train a
# prod config through maps_chain.sh: its 11 h budget leaves too little of this 12 h job for eval and infer.
#   VERSION=desy6/v1 PROBE=clustering ./maps_chain.sh
#   VERSION=desy6/v1 STAGES="eval infer" PROBE=lensing sbatch --time=02:30:00 maps.sh   # recover a failed tail
#   VERSION=desy3/v18 STAGES=eval MOCK_LABELS=<label> OVERWRITE=1 sbatch maps.sh   # append one mock

source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

PROBE="${PROBE:-lensing}"             # configs/probes/[<survey>/]<PROBE>.yaml
ARCH="${ARCH:-deepsphere}"            # configs/maps/prod/<ARCH>/: deepsphere | deepsphere_convnext | transformer
INPUTS="${INPUTS:-maps+cls}"          # maps+cls | maps
STAGES="${STAGES:-train eval infer}"
MOCK_LABELS="${MOCK_LABELS:-}"        # without train: evaluate only these mocks, appended to the preds
RUN_NUM="${RUN_NUM:-1}"               # position in a chain; >1 restores the checkpoint
TRAIN_EXTRA="${TRAIN_EXTRA:-}"        # extra run_training.py flags, e.g. --profile, --n_steps=N

# An explicit NET_CONFIG is an experiment and lands in maps/<probe>/<config name> by default.
if [ -n "${NET_CONFIG:-}" ]; then
    METHOD_DIR="${METHOD_DIR:-maps}"
    RUN="${RUN:-$(basename "${NET_CONFIG%.yaml}")}"
else
    NET_CONFIG="$DEEP_LSS/configs/maps/prod/$ARCH/$PROBE/$INPUTS.yaml"
    case "$ARCH" in
        deepsphere) method=maps_gcnn ;;
        deepsphere_convnext) method=maps_convnext ;;
        *) method="maps_$ARCH" ;;
    esac
    [ "$INPUTS" = "maps" ] && method="${method}_nocls"
    METHOD_DIR="${METHOD_DIR:-$method}"
    RUN="${RUN:-v1}"
fi

STRATEGY="mirrored"
OUTPUT="$RUNS_ROOT/$METHOD_DIR/$PROBE"
LOG="$OUTPUT/$RUN/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
has_stage train && [ "$RUN_NUM" = "1" ] && fresh_run "$OUTPUT/$RUN"
{ has_stage eval || has_stage infer; } && refuse_existing_preds "$OUTPUT/$RUN"
make_log_dir "$(dirname "$LOG")"
echo "maps: $NET_CONFIG -> $OUTPUT/$RUN ($STAGES)"

# --- Training ----------------------------------------------------------------------------------

if has_stage train; then
    resolve_net "$NET_CONFIG" "${LOG}_net_config.txt"
    check_stage $? "Net config resolution" "$RESOLVED"
    ensure_cls_cache "$NET_CONFIG" "${LOG}_precache.log"
    check_stage $? "Cls cache" "${LOG}_precache.log"

    RESTORE=""; [ "$RUN_NUM" -gt 1 ] && RESTORE="--restore_checkpoint"
    $SRUN --environment=tensorflow --gpu-bind=none --output="${LOG}_training.log" \
        python "$DEEP_LSS/deep_lss/apps/run_training.py" \
            --dir_base="$OUTPUT" \
            --dir_model="$RUN" \
            --train_tfr_pattern="$TRAIN_TFR" \
            --data_dir="$INPUT" \
            --msfm_config="$MSFM_CONFIG" \
            --probes_config="$PROBES_DIR$PROBE.yaml" \
            --scales_config="$SCALES_CONFIG" \
            --loss_config="$LOSS_CONFIG" \
            --data_config="$DATA_CONFIG" \
            --net_config="$NET_CONFIG" \
            --dist_strategy="$STRATEGY" \
            --wandb \
            --wandb_tags "$VERSION" "$SUBVERSION" "$PROBE" "$LOSS" "$STRATEGY" "$ARCH" \
                "$(basename "${NET_CONFIG%.yaml}")" "$SCALES" \
            $RESTORE $TRAIN_EXTRA
    check_stage $? "Training" "${LOG}_training.log"
fi

# --- Evaluation --------------------------------------------------------------------------------

EVAL_FLAGS="--grid_vali_tfr_pattern=$TRAIN_TFR --include_grid --include_des --include_mocks"
if [ -n "$MOCK_LABELS" ] && ! has_stage train; then
    # No grid pattern: a grid pass would rewrite grid/preds/test under the already trained flow.
    EVAL_FLAGS="--include_mocks"
    export INCLUDE_OBS="--include_mocks" FLOW_MEMBERS=""
fi
MOCK_FLAG=""; [ -n "$MOCK_LABELS" ] && MOCK_FLAG="--mock_labels $MOCK_LABELS"

if has_stage eval; then
    has_stage train && settle
    $SRUN --environment=tensorflow --gpu-bind=none --output="${LOG}_evaluation.log" \
        python "$DEEP_LSS/deep_lss/apps/run_evaluation.py" \
            --dir_model="$OUTPUT/$RUN" \
            --dist_strategy="$STRATEGY" \
            --obs_dir="$OBS_DIR" \
            $EVAL_FLAGS $MOCK_FLAG
    check_stage $? "Evaluation" "${LOG}_evaluation.log"
fi

# --- Inference ---------------------------------------------------------------------------------

if has_stage infer; then
    has_stage eval && settle
    run_inference "$RUNS_ROOT" "$METHOD_DIR/$PROBE/$RUN"
fi
