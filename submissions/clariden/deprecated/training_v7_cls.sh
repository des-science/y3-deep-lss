#!/bin/bash
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

# extract Weights & Biases API key from the host's .netrc file and pass it as an environment variable
# to accommodate containerized execution that might not inherit the host's home directory mounts properly.
export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)

# Optimize OpenMP and TensorFlow thread pools for the 288 available CPU cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

# v7_cls always continues training from a seeded v6_cls checkpoint
RESTORE_FLAG="--restore_checkpoint"

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"

STRATEGY="mirrored"
LOSS="vmim"
SCALES="8wl,32gc"
# SCALES="unsmoothed"
DATA="default"

PROBE="${PROBE:?set PROBE=lensing|clustering|combined}"

MAPS_PLUS_CLS="true"
# MAPS_PLUS_CLS="false"

PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml"
if [ "$MAPS_PLUS_CLS" = "true" ]; then
    NET_CONFIG="$REPOS/y3-deep-lss/configs/deepsphere/${PROBE}/maps+cls.yaml"
else
    NET_CONFIG="$REPOS/y3-deep-lss/configs/deepsphere/${PROBE}/maps.yaml"
fi

MODEL="v7_cls"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL/logs/"$STRATEGY"_"$SLURM_JOB_ID""

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_training.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_training.py \
        --dir_base=$OUTPUT \
        --dir_model=$MODEL \
        --train_tfr_pattern=$TRAIN_TFR \
        --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --probes_config=$PROBES_CONFIG \
        --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
        --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
        --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
        --net_config=$NET_CONFIG \
        --dist_strategy="$STRATEGY" \
        --wandb \
        --wandb_tags "$VERSION" "$SUBVERSION" "$PROBE" "$LOSS" "$STRATEGY" "resnet" \
        $RESTORE_FLAG

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

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/default.yaml"

srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output=""$LOG"_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --include_grid \
        --include_des \
        --include_mocks"
