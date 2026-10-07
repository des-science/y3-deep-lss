#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=rate_probe
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Sustained 4-GPU it/s of a maps net config: prod/maps.sh's training for PROBE_STEPS steps, without
# wandb, eval or inference, into claude/bench/rate_probe/. The single-GPU benchmark_sweep.sh and
# extrapolations across widths both missed the real rate by 25-40%.
#   PROBE=combined NET_CONFIG=$PWD/configs/maps/dev/deepsphere/combined/bench_v12/classic_nodrop.yaml \
#       sbatch --job-name=rate_combined rate_probe.sh

source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

PROBE="${PROBE:-lensing}"
ARCH="${ARCH:-deepsphere}"
NET_CONFIG="${NET_CONFIG:-$DEEP_LSS/configs/maps/prod/$ARCH/$PROBE/maps+cls.yaml}"
# Must clear compilation and the dataloader ramp, and cross checkpoint_every (5000) as the real run does.
PROBE_STEPS="${PROBE_STEPS:-6000}"

TAG="${TAG:-${ARCH}_${PROBE}_$(basename "${NET_CONFIG%.yaml}")}"
OUT_DIR="${OUT_DIR:-$MYSCRATCH/deep_lss/claude/bench/rate_probe}"
RUN_DIR="$OUT_DIR/$TAG"
make_log_dir "$RUN_DIR"
LOG="$RUN_DIR/${SLURM_JOB_ID}_training.log"

resolve_net "$NET_CONFIG" "${LOG%.log}_net_config.txt"
check_stage $? "Net config resolution" "$RESOLVED"
ensure_cls_cache "$NET_CONFIG" "${LOG%.log}_precache.log"
check_stage $? "Cls cache" "${LOG%.log}_precache.log"

echo "=== rate_probe: $TAG, $PROBE_STEPS steps, batch" \
    "$(grep -E '^dset\.training\.grid\.local_batch_size=' "$RESOLVED" | cut -d= -f2), $NET_CONFIG"

$SRUN --environment=tensorflow --gpu-bind=none --output="$LOG" \
    python "$DEEP_LSS/deep_lss/apps/run_training.py" \
        --dir_base="$OUT_DIR" \
        --dir_model="$TAG" \
        --train_tfr_pattern="$TRAIN_TFR" \
        --data_dir="$INPUT" \
        --msfm_config="$MSFM_CONFIG" \
        --probes_config="$DEEP_LSS/configs/probes/$PROBE.yaml" \
        --scales_config="$SCALES_CONFIG" \
        --loss_config="$LOSS_CONFIG" \
        --data_config="$DATA_CONFIG" \
        --net_config="$NET_CONFIG" \
        --n_steps="$PROBE_STEPS" \
        --dist_strategy=mirrored
status=$?
[ "$DRYRUN" = "1" ] && exit 0

# --- Report ------------------------------------------------------------------------------------

# throughput.json is binned past compilation; prefer it to the log.
if [ -f "$RUN_DIR/throughput.json" ]; then
    RATE=$(python3 -c "import json,sys; print('%.2f' % json.load(open(sys.argv[1]))['sustained_it_per_s'])" \
        "$RUN_DIR/throughput.json" 2>/dev/null)
    DRIFT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('drift_percent'))" \
        "$RUN_DIR/throughput.json" 2>/dev/null)
    echo "    drift: ${DRIFT}% (large drift: use wall_budget_seconds, not a fixed n_steps)"
fi

# Fallback, never tqdm's own it/s: that is the cumulative mean, 9-44% low from compilation. Take
# the rate over the second half of the progress bar instead.
RATE=${RATE:-$(tr '\r' '\n' < "$LOG" | awk -v total="$PROBE_STEPS" '
    match($0, /[0-9]+\/[0-9]+ \[[0-9:]+</) {
        seg = substr($0, RSTART, RLENGTH)
        split(seg, a, /[\/ \[<]/)
        step = a[1] + 0
        n = split(a[4], t, ":")
        secs = (n == 3) ? t[1]*3600 + t[2]*60 + t[3] : t[1]*60 + t[2]
        if (step >= total/2 && !mid_set) { mid_s = step; mid_t = secs; mid_set = 1 }
        last_s = step; last_t = secs
    }
    END {
        if (mid_set && last_t > mid_t && last_s > mid_s)
            printf "%.2f", (last_s - mid_s) / (last_t - mid_t)
    }')}

if [ -z "$RATE" ]; then
    echo "Could not parse a rate from $LOG" >&2
    tail -30 "$LOG" >&2
    exit "${status:-1}"
fi

# Training seconds in a 1 x 12 h and a 2 x 12 h budget, net of the measured ~35 min eval tail.
SINGLE=$(awk -v r="$RATE" 'BEGIN{printf "%d", int(r*41000/10000)*10000}')
CHAIN=$(awk -v r="$RATE" 'BEGIN{printf "%d", int(r*83900/10000)*10000}')
echo "    sustained rate : $RATE it/s"
echo "    n_steps  1x12h : $SINGLE   (rate x 41000, down to 10k)"
echo "    n_steps  2x12h : $CHAIN   (rate x 83900, down to 10k)"
echo "    Prefer n_steps: auto with wall_budget_seconds 41000 / 83900 over a fixed step count."

printf '{"tag": "%s", "probe": "%s", "arch": "%s", "config": "%s", "timed_steps": %s, "it_per_s": %s, "n_steps_1x12h": %s, "n_steps_2x12h": %s}\n' \
    "$TAG" "$PROBE" "$ARCH" "$NET_CONFIG" "$PROBE_STEPS" "$RATE" "$SINGLE" "$CHAIN" \
    >> "$OUT_DIR/rates.jsonl"
exit "$status"
