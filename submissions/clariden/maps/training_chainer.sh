#!/bin/bash
# Chains N training.sh runs across separate SLURM jobs, for training budgets beyond one 12h
# job. training.sh already restores the checkpoint whenever RUN_NUM>1, so this one script
# covers both a fresh chain (START_RUN=1, default) and resuming/rescuing an existing run
# (START_RUN>1, optionally gated on AFTER).
#
# --dependency=afterany, not afterok: a chained job is expected to hit the wall-clock TIMEOUT
# rather than COMPLETE, and afterok only fires on COMPLETED.
#
# Usage:
#   ARCH=deepsphere PROBE=combined NET_CONFIG=<path> MODEL_DIR=<run_name> ./training_chainer.sh
#   MAX_RUNS=3 ... ./training_chainer.sh              # longer chain
#   ... ./training_chainer.sh 3 2908306               # rescue job, gated on running job 2908306
#
# All training.sh env vars are forwarded via --export=ALL.

MAX_RUNS=${MAX_RUNS:-2}
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/training.sh"

START_RUN=${1:-1}
AFTER=${2:-}   # optional job id the first submitted run waits on via afterany

# Name the jobs after the run. Without this every chained job inherits training.sh's SBATCH default
# of "training", so squeue and sacct show a column of identical names and there is no way to tell
# which chain is which -- the bench_v7 GCNN chains were unreadable in exactly this way while the
# transformer ones, submitted with an explicit --job-name, were not. Defaults to the run directory.
JOB_NAME="${JOB_NAME:-${MODEL_DIR:-training}}"

dep=""
[ -n "$AFTER" ] && dep="--dependency=afterany:$AFTER"

# Abort the moment an sbatch fails. Without this a rejected submission (QOS limits are the usual
# cause) yields an empty job id, and the next job in the loop is then submitted with an EMPTY
# dependency -- so it starts immediately, out of order, and silently trains from the wrong step.
submit() {
    local out
    if ! out=$(sbatch --parsable "$@" $SCRIPT); then
        echo "sbatch failed; aborting the chain (jobs already submitted are left queued)." >&2
        exit 1
    fi
    echo "$out"
}

jid=$(submit $dep --job-name="${JOB_NAME}_${START_RUN}" --export=ALL,RUN_NUM=$START_RUN)
echo "Submitting run $START_RUN as job $jid (${JOB_NAME}_${START_RUN}) ${AFTER:+(after $AFTER)}"

for run in $(seq $((START_RUN + 1)) $MAX_RUNS); do
    jid=$(submit --dependency=afterany:$jid --job-name="${JOB_NAME}_${run}" --export=ALL,RUN_NUM=$run)
    echo "Submitting run $run as job $jid (${JOB_NAME}_${run}) (after previous, afterany)"
done
