#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=maps
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Map-level network: train, evaluate, infer. The defaults are the v18 production run
# maps_gcnn/<probe>/v1, which training refuses to overwrite; set RUN for a new one. Combined needs
# 2 x 12 h, so submit it through maps_chain.sh.
#   PROBE=clustering sbatch maps.sh
#   STAGES="eval infer" PROBE=lensing sbatch maps.sh                  # recover a failed tail
#   STAGES=eval MOCK_LABELS=<label> RUNS_ROOT=<store runs> sbatch maps.sh   # append one mock

source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

PROBE="${PROBE:-lensing}"             # configs/probes/<PROBE>.yaml
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
make_log_dir "$(dirname "$LOG")"
echo "maps: $NET_CONFIG -> $OUTPUT/$RUN ($STAGES)"

# --- Training ----------------------------------------------------------------------------------

if has_stage train; then
    if [ "$RUN_NUM" = "1" ] && [ -f "$OUTPUT/$RUN/configs.yaml" ] && [ "${OVERWRITE:-0}" != "1" ]; then
        echo "$OUTPUT/$RUN already holds a run; choose another RUN, or set OVERWRITE=1." >&2
        exit 1
    fi
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
            --probes_config="$DEEP_LSS/configs/probes/$PROBE.yaml" \
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
            --data_dir="$OBS_INPUT" \
            $EVAL_FLAGS $MOCK_FLAG
    check_stage $? "Evaluation" "${LOG}_evaluation.log"
fi

# --- Inference ---------------------------------------------------------------------------------

if has_stage infer; then
    has_stage eval && settle
    run_inference "$RUNS_ROOT" "$METHOD_DIR/$PROBE/$RUN"
fi
