#!/bin/bash
# Superseded 2026-08-04 by training_chainer.sh, which now covers both fresh-start and resume
# via START_RUN (plus the same AFTER-gating this script introduced) and uses
# --dependency=afterany (this used afterok, wrong for a job that's designed to TIMEOUT — see
# training_chainer.sh's header).
#
# Chain resume_training.sh runs for a given PROBE/MODEL. Each run restores the previous
# checkpoint (+ configs.yaml) and trains another n_steps, so runs 1/2/3 accumulate to
# 120k/240k/360k steps. This is the resume-side analogue of training_chainer.sh (which
# chains training.sh from a fresh run 1).
#
# Usage:
#   PROBE=<probe> MODEL=<model> ./resume_chainer.sh [START_RUN] [AFTER_JOBID]
#     START_RUN   first resume run number to submit (default 2)
#     AFTER_JOBID optional job id that START_RUN waits on via afterok (e.g. the still-running
#                 run-1 job); omit if run START_RUN-1 has already finished.
# Examples:
#   PROBE=lensing    MODEL=t4_cls ./resume_chainer.sh 2            # run 1 already done
#   PROBE=clustering MODEL=t4_cls ./resume_chainer.sh 2 2690623    # wait on running run 1
MAX_RUNS=${MAX_RUNS:-3}
SCRIPT=resume_training.sh

PROBE=${PROBE:-combined}
MODEL=${MODEL:-t4_cls}
START_RUN=${1:-2}
AFTER=${2:-}   # optional afterok dependency for the first submitted run

dep=""
[ -n "$AFTER" ] && dep="--dependency=afterok:$AFTER"

# First submitted run (optionally gated on AFTER)
jid=$(sbatch --parsable $dep --export=ALL,PROBE=$PROBE,MODEL=$MODEL,RUN_NUM=$START_RUN $SCRIPT)
echo "Submitting $PROBE $MODEL run $START_RUN as job $jid ${AFTER:+(after $AFTER)}"

# Chain the rest, each afterok on the previous
for run in $(seq $((START_RUN + 1)) $MAX_RUNS); do
    jid=$(sbatch --parsable --dependency=afterok:$jid --export=ALL,PROBE=$PROBE,MODEL=$MODEL,RUN_NUM=$run $SCRIPT)
    echo "Submitting $PROBE $MODEL run $run as job $jid (after previous)"
done
