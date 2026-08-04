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
SCRIPT=training.sh

START_RUN=${1:-1}
AFTER=${2:-}   # optional job id the first submitted run waits on via afterany

dep=""
[ -n "$AFTER" ] && dep="--dependency=afterany:$AFTER"

jid=$(sbatch --parsable $dep --export=ALL,RUN_NUM=$START_RUN $SCRIPT)
echo "Submitting run $START_RUN as job $jid ${AFTER:+(after $AFTER)}"

for run in $(seq $((START_RUN + 1)) $MAX_RUNS); do
    jid=$(sbatch --parsable --dependency=afterany:$jid --export=ALL,RUN_NUM=$run $SCRIPT)
    echo "Submitting run $run as job $jid (after previous, afterany)"
done
