#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=eval_inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Runs ONLY the evaluation + inference stages of training.sh against an already-trained
# model (restores the latest checkpoint). Use this when training has completed but the
# downstream stages still need to run — e.g. after fixing the run_evaluation.py transformer
# crash. The env-var block below is kept identical to training.sh so the same MODEL/paths
# are resolved.

# extract Weights & Biases API key from the host's .netrc file and pass it as an environment variable
export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)

# Optimize OpenMP and TensorFlow thread pools for the 288 available CPU cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

RUN_NUM=${RUN_NUM:-1}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"

STRATEGY="mirrored"
LOSS="vmim"
SCALES="8wl,32gc"
# SCALES="unsmoothed"
DATA="default"

PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"

# MAPS_PLUS_CLS="true"
MAPS_PLUS_CLS="false"

# network architecture: directory under configs/ holding the per-probe net configs
# ARCH="deepsphere"
ARCH="transformer"

PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml"
if [ "$MAPS_PLUS_CLS" = "true" ]; then
    NET_CONFIG="$REPOS/y3-deep-lss/configs/${ARCH}/${PROBE}/maps+cls.yaml"
else
    NET_CONFIG="$REPOS/y3-deep-lss/configs/${ARCH}/${PROBE}/maps.yaml"
fi

# MODEL="v8_cls"
MODEL="t1"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

# --dir_model is passed explicitly here (full path to the model directory). In training.sh
# the training stage writes ./.env_var/id_<JOB_ID>.txt and the eval stage of the SAME job
# reads it; a standalone eval job has no such file, so we point at the model directory directly.
srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_evaluation.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_evaluation.py \
        --dir_model="$OUTPUT/$MODEL" \
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
