#!/bin/bash
# Canonical launcher for an N x 12h chained training.sh run — the regime used for the
# bench_v5/bench_v6 DeepSphere combined-probe benchmarks (e.g. bench_v5_pool_head_w64,
# bench_v6_convnext). A single SLURM job is capped around 12h on the normal partition, so a
# run that needs more budget than that is chained: each job restores the previous job's
# checkpoint (training.sh already passes --restore_checkpoint whenever RUN_NUM>1) and
# continues the SAME cosine schedule (decay_steps is fixed by the total n_steps recorded in
# <dir_model>/configs.yaml at RUN_NUM=1, not recomputed per job).
#
# --dependency=afterany, never afterok: job 1 (and any middle job) is DESIGNED to hit the
# wall-clock limit and end in SLURM state TIMEOUT, not COMPLETED. afterok only fires on
# COMPLETED, so it would never release the next job. See
# configs/deepsphere/combined/bench_v6/README.md ("## Launching") for the full story,
# including how a run that's short on its budget after 2 jobs gets rescued with a 3rd.
#
# This single script covers both a FRESH chain (START_RUN=1, the default) and RESUMING an
# existing run or attaching a rescue job to one still in flight (START_RUN>1, optionally
# gated on AFTER) — training.sh's own RUN_NUM logic is what decides --restore_checkpoint, so
# there is no separate "resume" variant needed.
#
# Usage:
#   ARCH=deepsphere PROBE=combined \
#       NET_CONFIG="$PWD/configs/deepsphere/combined/<bench>/<config>.yaml" \
#       MODEL_DIR=<run_name> \
#       ./training_chainer.sh                    # fresh 2-job chain (RUN_NUM 1,2)
#
#   MAX_RUNS=3 ARCH=deepsphere PROBE=combined NET_CONFIG=... MODEL_DIR=... ./training_chainer.sh
#                                                 # fresh 3-job chain
#
#   ARCH=deepsphere PROBE=combined NET_CONFIG=... MODEL_DIR=<existing_run_name> \
#       ./training_chainer.sh 3 2908306          # attach rescue job RUN_NUM=3, waiting on
#                                                 # already-running job 2908306 (job 2 of that
#                                                 # run) — mirrors how bench_v6_convnext's
#                                                 # ConvNeXt arms were actually rescued.
#
# All training.sh env vars (VERSION, SUBVERSION, PROBE, PROBE_CONFIG, LOSS, ARCH, SCALES,
# NET_CONFIG, MODEL_DIR, CLS_PROBES_CONFIG, ...) are forwarded via --export=ALL, i.e. whatever
# you set in the calling shell before invoking this script.

MAX_RUNS=${MAX_RUNS:-2}
SCRIPT=training.sh

START_RUN=${1:-1}
AFTER=${2:-}   # optional job id the first submitted run waits on via afterany

dep=""
[ -n "$AFTER" ] && dep="--dependency=afterany:$AFTER"

# First submitted run (optionally gated on AFTER)
jid=$(sbatch --parsable $dep --export=ALL,RUN_NUM=$START_RUN $SCRIPT)
echo "Submitting run $START_RUN as job $jid ${AFTER:+(after $AFTER)}"

# Chain the rest, each afterany on the previous (job N is expected to TIMEOUT, not COMPLETE)
for run in $(seq $((START_RUN + 1)) $MAX_RUNS); do
    jid=$(sbatch --parsable --dependency=afterany:$jid --export=ALL,RUN_NUM=$run $SCRIPT)
    echo "Submitting run $run as job $jid (after previous, afterany)"
done
