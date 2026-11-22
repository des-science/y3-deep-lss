#!/bin/bash
# Chains MAX_RUNS maps.sh jobs with afterany (a chained job ends in TIMEOUT, which afterok never
# releases). Run it on the login node; every maps.sh variable is forwarded.
#   PROBE=combined ./maps_chain.sh                 # fresh chain
#   PROBE=combined ./maps_chain.sh 3 2908306       # rescue: RUN_NUM=3 after job 2908306

PROBE="${PROBE:-lensing}"
DEFAULT_RUNS=1; [ "$PROBE" = "combined" ] && DEFAULT_RUNS=2  # the prod configs' wall budgets
MAX_RUNS="${MAX_RUNS:-$DEFAULT_RUNS}"
START_RUN="${1:-1}"
AFTER="${2:-}"
JOB_NAME="${JOB_NAME:-maps_${PROBE}_${RUN:-v1}}"
SCRIPT="/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/prod/maps.sh"
export PROBE

# Abort on a rejected submission; its successor would otherwise start at once, out of order.
submit() {
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "sbatch --parsable $* $SCRIPT" >&2
        echo "dryrun"
        return
    fi
    sbatch --parsable "$@" "$SCRIPT"
}

# A rescue job (START_RUN > MAX_RUNS) is still submitted.
LAST_RUN=$((MAX_RUNS > START_RUN ? MAX_RUNS : START_RUN))
dep=""; [ -n "$AFTER" ] && dep="--dependency=afterany:$AFTER"
for run in $(seq "$START_RUN" "$LAST_RUN"); do
    jid=$(submit $dep --job-name="${JOB_NAME}_${run}" --export=ALL,RUN_NUM="$run") \
        || { echo "sbatch failed; aborting the chain." >&2; exit 1; }
    echo "run $run: job $jid ${dep:+($dep)}"
    dep="--dependency=afterany:$jid"
done
