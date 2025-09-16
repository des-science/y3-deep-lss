#!/bin/bash
MAX_RUNS=3
SCRIPT=single_node.sh
# SCRIPT=debug/debug_single_node.sh

# First job
jid=$(sbatch --parsable --export=ALL,RUN_NUM=1 $SCRIPT)
echo "Submitting run 1 with job $jid"

# Chain the rest
for run in $(seq 2 $MAX_RUNS); do
    echo "Submitting run $run after job $jid"
    jid=$(sbatch --parsable --dependency=afterok:$jid --export=ALL,RUN_NUM=$run $SCRIPT)
done