#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Cl-level network for several probes at once, one GPU each: train+evaluate, then infer. The
# defaults are the v18 production runs cls/<probe>/v1, which training refuses to overwrite.
#   RUN=v2 sbatch cls.sh
#   STAGES=cache sbatch --time=02:00:00 cls.sh                   # build the cache on a new dataset
#   STAGES=eval MOCK_LABELS=<label> RUNS_ROOT=<store runs> sbatch cls.sh   # append one mock
#   NET_CONFIG=<yaml> METHOD_DIR=cls_bench RUN=<name> sbatch cls.sh        # an experiment

export SLURM_CPUS_PER_TASK=72  # each probe is its own 1-GPU step
source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

# "probe" or "probe:probes_config"; v17 needs e.g. "lensing:lensing_nla combined:combined_nla"
read -r -a PROBES <<< "${PROBES:-lensing clustering combined}"
STAGES="${STAGES:-train infer}"       # cache | train | eval | infer
MOCK_LABELS="${MOCK_LABELS:-}"        # required with eval and no train: mocks appended to the preds

PROD_NET_CONFIG="$DEEP_LSS/configs/cls/branch/default.yaml"
NET_CONFIG="${NET_CONFIG:-$PROD_NET_CONFIG}"  # configs/cls/<net>/<name>.yaml
METHOD_DIR="${METHOD_DIR:-cls}"
RUN="${RUN:-v1}"

if [ "$NET_CONFIG" != "$PROD_NET_CONFIG" ] && [ "$METHOD_DIR/$RUN" = "cls/v1" ]; then
    echo "A non-production NET_CONFIG needs its own METHOD_DIR or RUN, not cls/<probe>/v1." >&2
    exit 1
fi

if has_stage eval && ! has_stage train; then
    [ -n "$MOCK_LABELS" ] || { echo "STAGES=eval appends mocks and needs MOCK_LABELS." >&2; exit 1; }
    EVAL_FLAGS="--eval_only --include_mocks --mock_labels $MOCK_LABELS"
    export INCLUDE_OBS="--include_mocks" FLOW_MEMBERS=""
else
    EVAL_FLAGS="--include_grid --include_des --include_mocks"
fi

OUTPUT="$RUNS_ROOT/$METHOD_DIR"
if has_stage train && [ "${OVERWRITE:-0}" != "1" ]; then
    for ENTRY in "${PROBES[@]}"; do
        if [ -f "$OUTPUT/${ENTRY%%:*}/$RUN/configs.yaml" ]; then
            echo "$OUTPUT/${ENTRY%%:*}/$RUN already holds a run; choose another RUN, or set OVERWRITE=1." >&2
            exit 1
        fi
    done
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
                --probes_config="$DEEP_LSS/configs/probes/${ENTRY##*:}.yaml" \
                --scales_config="$SCALES_CONFIG" \
                --loss_config="$LOSS_CONFIG" \
                --net_config="$NET_CONFIG" \
                --data_config="$DATA_CONFIG" \
                --data_dir="$INPUT" \
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
