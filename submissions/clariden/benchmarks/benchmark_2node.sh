#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --job-name=bench_2node
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/2node/slurm/slurm-%j.out

# Multi-node / distribution-strategy throughput (run_training.py --pasc_throughput) of the v17
# clustering transformer. The topology comes from the sbatch flags, the strategy from the env.
#   STRATEGY=mirrored CFG=ctrl_b20 GPU_BIND=none VERSION=v17 SUBVERSION=baseline \
#       sbatch --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --gpus-per-task=4 --cpus-per-task=288 benchmark_2node.sh
#   STRATEGY=horovod CFG=b10 VERSION=v17 SUBVERSION=baseline \
#       sbatch --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --gpus-per-task=1 --cpus-per-task=72 benchmark_2node.sh

source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

STRATEGY="${STRATEGY:-multi_worker_mirrored}"  # mirrored | multi_worker_mirrored | horovod
CFG="${CFG:-b10}"                              # configs/maps/dev/transformer/clustering/bench_2node/
TAG="${TAG:-${STRATEGY}_${CFG}}"
GPU_BIND="${GPU_BIND:-single:1}"               # none for mirrored; single:1 gives each worker its GPU

PROBE="clustering"
NET_CONFIG="$DEEP_LSS/configs/maps/dev/transformer/clustering/bench_2node/${CFG}.yaml"
OUTPUT="$MYSCRATCH/deep_lss/claude/bench/2node"
MODEL_DIR="bench_2node_${TAG}_${SLURM_JOB_ID}"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${STRATEGY}"
make_log_dir "$(dirname "$LOG")"

resolve_net "$NET_CONFIG" "${LOG}_net_config.txt"
check_stage $? "Net config resolution" "$RESOLVED"
ensure_cls_cache "$NET_CONFIG" "${LOG}_precache.log"
check_stage $? "Cls cache" "${LOG}_precache.log"

SRUN_MPI=""; [ "$STRATEGY" = "horovod" ] && SRUN_MPI="--mpi=pmix"  # horovod's OpenMPI needs PMIx
echo "bench_2node: $TAG, nodes=${SLURM_JOB_NUM_NODES:-?} tasks/node=${SLURM_NTASKS_PER_NODE:-?}"

$SRUN $SRUN_MPI --environment=tensorflow --gpu-bind="$GPU_BIND" --output="${LOG}_training_%t.log" \
    python "$DEEP_LSS/deep_lss/apps/run_training.py" \
        --dir_base="$OUTPUT" \
        --dir_model="$MODEL_DIR" \
        --train_tfr_pattern="$TRAIN_TFR" \
        --data_dir="$INPUT" \
        --msfm_config="$MSFM_CONFIG" \
        --probes_config="$DEEP_LSS/configs/probes/$PROBE.yaml" \
        --scales_config="$SCALES_CONFIG" \
        --loss_config="$LOSS_CONFIG" \
        --data_config="$DATA_CONFIG" \
        --net_config="$NET_CONFIG" \
        --dist_strategy="$STRATEGY" \
        --pasc_throughput
[ "$DRYRUN" = "1" ] && exit 0

echo "=== bench_2node summary ($TAG) ==="
grep -h "replicas\|throughput:\|steps took" "${LOG}_training_0.log" | tail -5
