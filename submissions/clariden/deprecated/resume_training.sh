#!/bin/bash
# Superseded 2026-08-04 by maps/training.sh (RUN_NUM>1 already implies --restore_checkpoint)
# + maps/training_chainer.sh. Also carried a stale VERSION=v16/SUBVERSION=rot_in_place
# hardcode, never migrated to v17/baseline — do not resurrect without fixing that first.
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# disable core dumps: a crashing TF/Python task can otherwise write a 50+ GB
# core file into the cwd (core_pattern=core_%h_%p) and fill the /users quota
ulimit -c 0

# extract Weights & Biases API key from the host's .netrc file and pass it as an environment variable
# to accommodate containerized execution that might not inherit the host's home directory mounts properly.
export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)

# Optimize OpenMP and TensorFlow thread pools for the 288 available CPU cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

RUN_NUM=${RUN_NUM:-2}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"

STRATEGY="mirrored"

# PROBE / MODEL may be overridden from the environment (e.g. sbatch --export=ALL,PROBE=lensing,MODEL=t2_default)
# PROBE options: lensing / clustering / combined
PROBE="${PROBE:-combined}"

# MODEL="v6"
MODEL="${MODEL:-v8_cls}"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

# net/probes/scales/loss/msfm configs are restored from $OUTPUT/$MODEL/configs.yaml
srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_training.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_training.py \
        --dir_base=$OUTPUT \
        --dir_model=$MODEL \
        --train_tfr_pattern=$TRAIN_TFR \
        --grid_vali_tfr_pattern=$TRAIN_TFR \
        --dist_strategy="$STRATEGY" \
        --wandb \
        --restore_checkpoint

sleep 30

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_evaluation.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_evaluation.py \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern=$TRAIN_TFR \
        --data_dir=$INPUT \
        --include_grid \
        --include_des \
        --include_mocks

sleep 30

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output=""$LOG"_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        --sample_posterior \
        --include_grid \
        --include_des \
        --include_mocks"
