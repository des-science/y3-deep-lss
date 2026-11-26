#!/bin/bash
#SBATCH --account=lp184
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Cl-level network for several probes at once, one GPU each: train+evaluate, then infer. VERSION picks
# the dataset (no default); the runs default to the production runs cls/<probe>/v1. Training over an
# existing run needs OVERWRITE=1, which deletes that run first; eval or infer on runs that already
# hold predictions need OVERWRITE=1 too.
#   VERSION=desy6/v1 RUN=v2 sbatch cls.sh
#   VERSION=desy6/v1 STAGES=cache sbatch --time=02:00:00 cls.sh             # build the cache on a new dataset
#   VERSION=desy3/v18 STAGES=eval MOCK_LABELS=<label> OVERWRITE=1 sbatch cls.sh   # append one mock
#   VERSION=desy6/v2 STAGES="eval infer" DES_LABELS=DESy6_lens_isd PROBES="clustering combined" OVERWRITE=1 \
#       sbatch --partition=debug --time=00:30:00 cls.sh   # add a data observation variant to finished runs
#   VERSION=desy6/v1 NET_CONFIG=<yaml> METHOD_DIR=cls_bench RUN=<name> sbatch cls.sh   # an experiment

export SLURM_CPUS_PER_TASK=72  # each probe is its own 1-GPU step
source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

# "probe" or "probe:probes_config"; v17 needs e.g. "lensing:lensing_nla combined:combined_nla"
read -r -a PROBES <<< "${PROBES:-lensing clustering combined}"
STAGES="${STAGES:-train infer}"       # cache | train | eval | infer
MOCK_LABELS="${MOCK_LABELS:-}"        # eval without train: mocks appended to the preds
DES_LABELS="${DES_LABELS:-}"          # eval without train: DES labels (data_observations.yaml) appended, trained flows

PROD_NET_CONFIG="$DEEP_LSS/configs/cls/branch/default.yaml"
NET_CONFIG="${NET_CONFIG:-$PROD_NET_CONFIG}"  # configs/cls/<net>/<name>.yaml
METHOD_DIR="${METHOD_DIR:-cls}"
RUN="${RUN:-v1}"

if [ "$NET_CONFIG" != "$PROD_NET_CONFIG" ] && [ "$METHOD_DIR/$RUN" = "cls/v1" ]; then
    echo "A non-production NET_CONFIG needs its own METHOD_DIR or RUN, not cls/<probe>/v1." >&2
    exit 1
fi

if [ -n "$DES_LABELS" ] && has_stage train; then
    echo "DES_LABELS adds to finished runs, so STAGES cannot train." >&2
    exit 1
fi
if has_stage eval && ! has_stage train && [ -z "$MOCK_LABELS$DES_LABELS" ]; then
    echo "STAGES=eval appends and needs MOCK_LABELS or DES_LABELS." >&2
    exit 1
fi

# without eval too, so that STAGES=infer reruns an add-on's inference on the same flows
if [ -n "$MOCK_LABELS$DES_LABELS" ] && ! has_stage train; then
    # mocks retrain the flows and DES labels load them, so they are separate jobs
    [ -z "$MOCK_LABELS" ] || [ -z "$DES_LABELS" ] || { echo "MOCK_LABELS and DES_LABELS are separate jobs." >&2; exit 1; }
    if [ -n "$MOCK_LABELS" ]; then
        EVAL_FLAGS="--eval_only --include_mocks --mock_labels $MOCK_LABELS"; INCLUDE_OBS="--include_mocks"
    else
        # the trained flows are loaded, and inference resamples every DES chain of the runs on them
        EVAL_FLAGS="--eval_only --include_des $DATA_OBS_FLAG --des_labels $DES_LABELS"; INCLUDE_OBS="--include_des"
        export LOAD_FLOW="${LOAD_FLOW---load_flow}" SAMPLE_POSTERIOR="${SAMPLE_POSTERIOR-}"
    fi
    export INCLUDE_OBS FLOW_MEMBERS=""
else
    EVAL_FLAGS="--include_grid --include_des $DATA_OBS_FLAG --include_mocks"
fi

OUTPUT="$RUNS_ROOT/$METHOD_DIR"
has_stage train && fresh_run $(for e in "${PROBES[@]}"; do echo "$OUTPUT/${e%%:*}/$RUN"; done)
if has_stage eval || has_stage infer; then
    refuse_existing_preds $(for e in "${PROBES[@]}"; do echo "$OUTPUT/${e%%:*}/$RUN"; done)
fi
LOG_DIR="$INPUT/precache/logs"
make_log_dir "$LOG_DIR"
echo "cls: $NET_CONFIG -> $OUTPUT/{${PROBES[*]}}/$RUN ($STAGES)"

# --- Cls cache, once for all probes ------------------------------------------------------------

resolve_net "$NET_CONFIG" "$LOG_DIR/${SLURM_JOB_ID}_net_config.txt"
check_stage $? "Net config resolution" "$RESOLVED"
ensure_cls_cache "$NET_CONFIG" "$LOG_DIR/${SLURM_JOB_ID}_precache.log"
check_stage $? "Cls cache" "$LOG_DIR/${SLURM_JOB_ID}_precache.log"
[ "$STAGES" = "cache" ] && exit 0

# --- Training and evaluation, one probe per GPU ------------------------------------------------

if has_stage train || has_stage eval; then
    pids=()
    for ENTRY in "${PROBES[@]}"; do
        PROBE="${ENTRY%%:*}"
        LOG="$OUTPUT/$PROBE/$RUN/logs/${SLURM_JOB_ID}"
        make_log_dir "$(dirname "$LOG")"
        $SRUN -N1 -n1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G --cpu-bind=none \
            --environment=tensorflow --output="${LOG}_training.log" \
            python "$DEEP_LSS/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$MSFM_CONFIG" \
                --probes_config="$PROBES_DIR${ENTRY##*:}.yaml" \
                --scales_config="$SCALES_CONFIG" \
                --loss_config="$LOSS_CONFIG" \
                --net_config="$NET_CONFIG" \
                --data_config="$DATA_CONFIG" \
                --data_dir="$INPUT" \
                --obs_dir="$OBS_DIR" \
                --out_dir="$OUTPUT/$PROBE" \
                --model_name="$RUN" \
                $EVAL_FLAGS &
        pids+=($!)
    done
    rc=0
    for i in "${!pids[@]}"; do
        wait "${pids[i]}" || { echo "Training+evaluation failed: ${PROBES[i]}" >&2; rc=1; }
    done
    [ "$rc" -eq 0 ] || exit 1
fi

# --- Inference, all probes in one call ---------------------------------------------------------

if has_stage infer; then
    run_inference "$RUNS_ROOT" $(for e in "${PROBES[@]}"; do echo "$METHOD_DIR/${e%%:*}/$RUN"; done)
fi
