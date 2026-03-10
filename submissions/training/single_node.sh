#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=training

VERSION="v16"
SUBVERSION="default"
MODEL="v1"

# PROBE="lensing"
PROBE="clustering"
# PROBE="combined"

RUN_NUM=${RUN_NUM:-1}
STRATEGY="mirrored"
LOSS="mutual_info"

BASE="/pscratch/sd/a/athomsen/deep_lss/$VERSION/$SUBVERSION/maps/$PROBE"
OUTPUT="/$BASE/$MODEL/logs/"$RUN_NUM"_"$STRATEGY"_"$SLURM_JOB_ID""

TRAIN_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"
GRID_EVAL_TFR=$TRAIN_TFR

# Add --restore_checkpoint only for RUN_NUM > 1
RESTORE_FLAG=""
if [ "$RUN_NUM" -gt 1 ]; then
    RESTORE_FLAG="--restore_checkpoint"
fi

srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_training.log" \
    python ../../deep_lss/apps/run_training.py \
        --dir_base=$BASE \
        --dir_model=$MODEL \
        --loss_function=$LOSS \
        --train_tfr_pattern=$TRAIN_TFR \
        --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
        --dlss_config="configs/$VERSION/default/$PROBE/dlss.yaml" \
        --net_config="configs/$VERSION/deepsphere_default.yaml" \
        --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --dist_strategy="$STRATEGY" \
        --wandb \
        --wandb_tags "$VERSION" "$SUBVERSION" "$PROBE" "$LOSS" "$STRATEGY" "resnet" \
        --wandb_notes="single $PROBE node run $RUN_NUM" \
        $RESTORE_FLAG \

# evaluate all the network checkpoints in a separate script after training has completed to avoid CPU OOM errors
srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_inference.log" \
    python ../../deep_lss/apps/run_evaluation.py \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern=$GRID_EVAL_TFR
