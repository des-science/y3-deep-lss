#!/bin/bash
# Trains a maps network in MAX_RUNS chained maps.sh jobs, then evaluates and infers in a job of its own: an 11 h
# training budget leaves too little of a 12 h job for eval plus the coverage chain. Every probe goes through here.
# The training jobs chain with afterany (a job that hits the wall clock ends in TIMEOUT, which afterok never
# releases); the eval + infer job waits on afterok, so a failed run is never evaluated. Run it on the login node;
# every maps.sh variable is forwarded, except STAGES, which this script sets per job.
#   VERSION=desy6/v1 PROBE=combined ./maps_chain.sh                 # fresh chain
#   VERSION=desy6/v1 PROBE=combined ./maps_chain.sh 3 2908306       # rescue: RUN_NUM=3 after job 2908306
#   VERSION=desy6/v1 AFTER_TYPE=afterok ./maps_chain.sh 1 5425960   # fresh, once a prerequisite (Cls cache) succeeded

PROBE="${PROBE:-lensing}"
DEFAULT_RUNS=1; [ "$PROBE" = "combined" ] && DEFAULT_RUNS=2  # the prod configs' wall budgets
MAX_RUNS="${MAX_RUNS:-$DEFAULT_RUNS}"
START_RUN="${1:-1}"
AFTER="${2:-}"
AFTER_TYPE="${AFTER_TYPE:-afterany}"  # how the first job waits on AFTER
JOB_NAME="${JOB_NAME:-maps_${PROBE}_${RUN:-v1}}"
TAIL_TIME="${TAIL_TIME:-02:30:00}"  # eval + infer took ~85 min on the DES Y6 prod runs
SCRIPT="/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/prod/maps.sh"
export PROBE

# submit <stages> <sbatch flags>...: prints the job id. Abort on a rejected submission; its successor would
# otherwise start at once, out of order.
submit() {
    local stages=$1
    shift
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "STAGES=\"$stages\" sbatch --parsable $* $SCRIPT" >&2
        echo "dryrun"
        return
    fi
    STAGES="$stages" sbatch --parsable "$@" "$SCRIPT"
}

# A rescue job (START_RUN > MAX_RUNS) is still submitted.
LAST_RUN=$((MAX_RUNS > START_RUN ? MAX_RUNS : START_RUN))
dep=""; [ -n "$AFTER" ] && dep="--dependency=$AFTER_TYPE:$AFTER"
for run in $(seq "$START_RUN" "$LAST_RUN"); do
    jid=$(submit train $dep --job-name="${JOB_NAME}_${run}" --export=ALL,RUN_NUM="$run") \
        || { echo "sbatch failed; aborting the chain." >&2; exit 1; }
    echo "run $run: job $jid ${dep:+($dep)}"
    dep="--dependency=afterany:$jid"
done

dep="--dependency=afterok:$jid"
jid=$(submit "eval infer" $dep --time="$TAIL_TIME" --job-name="${JOB_NAME}_eval_infer" \
    --export=ALL,RUN_NUM="$LAST_RUN") \
    || { echo "sbatch failed; the training chain is queued without its eval and infer." >&2; exit 1; }
echo "eval + infer: job $jid ($dep)"
