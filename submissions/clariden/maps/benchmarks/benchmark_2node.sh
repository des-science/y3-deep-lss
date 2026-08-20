#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --job-name=bench_2node
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/2node/slurm/slurm-%j.out

# Multi-node scaling benchmark for the v17 clustering maps+cls transformer recipe (global batch 80,
# 250k steps, 7.34 it/s avg on 1 node / 4 GPUs mirrored = 9h28m training): can 2 nodes fit the run
# into a single 12 h job? Runs run_training.py with --pasc_throughput (measures steps 200-1200) on
# a bench_2node config, no wandb, no eval/inference tail.
#
# Topology is set at submit time (CLI flags override the #SBATCH defaults above); the strategy,
# config and GPU binding come from the environment. The five benchmark points:
#
#   ctrl (mirrored, 1 node, known-good production launch geometry):
#     STRATEGY=mirrored CFG=ctrl_b20 TAG=ctrl_mirrored_b20 GPU_BIND=none \
#       sbatch --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --gpus-per-task=4 --cpus-per-task=288 benchmark_2node.sh
#   multi-worker (4 tasks/node x 1 GPU, Perlmutter launch geometry, --gpu-bind=single:1):
#     STRATEGY=multi_worker_mirrored CFG=b20 TAG=mwms_1node_b20 \
#       sbatch --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --gpus-per-task=1 --cpus-per-task=72 benchmark_2node.sh
#     STRATEGY=multi_worker_mirrored CFG=b10 TAG=mwms_2node_b10 \
#       sbatch --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --gpus-per-task=1 --cpus-per-task=72 benchmark_2node.sh
#     STRATEGY=multi_worker_mirrored CFG=b20 TAG=mwms_2node_b20 \
#       sbatch --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --gpus-per-task=1 --cpus-per-task=72 benchmark_2node.sh
#     STRATEGY=horovod CFG=b10 TAG=hvd_2node_b10 \
#       sbatch --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --gpus-per-task=1 --cpus-per-task=72 benchmark_2node.sh
#
# horovod comes from the NGC container (not the venv); if hvd.init() cannot rendezvous under srun
# on Clariden CE, the run reports "Running on 1 replicas" or crashes — either is a valid negative
# result, multi_worker_mirrored is the primary candidate.

# --- Runtime environment ---------------------------------------------------------------------

ulimit -c 0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

DEEP_LSS="$REPOS/y3-deep-lss"
MSFM="$REPOS/multiprobe-simulation-forward-model"

# --- Fixed settings ----------------------------------------------------------------------------

PROBE="clustering"
LOSS="vmim"
DATA="default"
SCALES="8wl,32gc"

# --- Overridable defaults (the benchmark point; see the table above) ---------------------------

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

STRATEGY="${STRATEGY:-multi_worker_mirrored}"
CFG="${CFG:-b10}"
TAG="${TAG:-${STRATEGY}_${CFG}}"
# mirrored (1 task, 4 GPUs) needs none; multi-worker (4 tasks x 1 GPU) needs single:1 so each
# task/replica sees exactly its own GPU (there is no in-code GPU pinning)
GPU_BIND="${GPU_BIND:-single:1}"

# --- Derived paths, configs and flags ----------------------------------------------------------

NET_CONFIG="$DEEP_LSS/configs/transformer/dev/clustering/bench_2node/${CFG}.yaml"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
MODEL_DIR="bench_2node_${TAG}_${SLURM_JOB_ID}"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

# maps+cls needs the rebinned-Cls asinh-calibration cache; run_training.py only reads it. It exists
# from the production clustering runs — fail fast instead of building it here.
CLS_CACHE="$INPUT/cls/rebinned_nb16_${SCALES}.h5"
if [ ! -f "$CLS_CACHE" ]; then
    echo "ERROR: Cls cache $CLS_CACHE missing — build it via" \
         "submissions/clariden/maps/training.sh or submissions/clariden/cls/cls_training.sh first."
    exit 1
fi

# --- Stage 1: Throughput-only training run -----------------------------------------------------

echo "bench_2node: TAG=$TAG STRATEGY=$STRATEGY CFG=$CFG nodes=$SLURM_JOB_NUM_NODES" \
     "tasks/node=$SLURM_NTASKS_PER_NODE gpu-bind=$GPU_BIND"

# horovod rendezvous goes through the container's OpenMPI, which needs srun to provide PMIx
# (without it: "OPAL ERROR: Unreachable in pmix3x_client.c", observed job 2795158)
SRUN_MPI=""
if [ "$STRATEGY" = "horovod" ]; then
    SRUN_MPI="--mpi=pmix"
fi

# %t = task rank, so the 4-8 multi-worker tasks don't interleave one file
srun $SRUN_MPI --environment=tensorflow --gpu-bind=$GPU_BIND --output="$LOG"_training_%t.log \
    python $DEEP_LSS/deep_lss/apps/run_training.py \
        --dir_base=$OUTPUT \
        --dir_model=$MODEL_DIR \
        --train_tfr_pattern=$TRAIN_TFR \
        --data_dir=$INPUT \
        --msfm_config="$MSFM/configs/$VERSION/$SUBVERSION.yaml" \
        --probes_config="$DEEP_LSS/configs/probes/${PROBE}.yaml" \
        --scales_config="$DEEP_LSS/configs/scales/${SCALES}.yaml" \
        --loss_config="$DEEP_LSS/configs/loss/${LOSS}.yaml" \
        --data_config="$DEEP_LSS/configs/data/${DATA}.yaml" \
        --net_config=$NET_CONFIG \
        --dist_strategy="$STRATEGY" \
        --pasc_throughput

echo "=== bench_2node summary ($TAG) ==="
grep -h "replicas\|throughput:\|steps took" "$LOG"_training_0.log | tail -5
