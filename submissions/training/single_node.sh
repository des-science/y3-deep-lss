#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=training
#SBATCH --output="./logs/v8/training_%j.log"

STRATEGY="mirrored"
VERSION="v8"
# lensing, clustering, combined
PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"
# linear_bias, quadratic_bias
BIAS="linear_bias"
# delta, likelihood
LOSS="delta"
# LOSS="likelihood"

OUTPUT="./logs/$VERSION/$PROBE/$LOSS/"$STRATEGY"_"$SLURM_JOB_ID".log"

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

TRAIN_TFR="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord"
FIDU_VALI_TFR="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord"
GRID_VALI_TFR="/pscratch/sd/a/athomsen/DESY3/v7/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=none --output="$OUTPUT" \
    python ../../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern=$TRAIN_TFR \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --grid_vali_tfr_pattern=$GRID_VALI_TFR \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
    --net_config="configs/$VERSION/$PROBE/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
    --slurm_output="$OUTPUT" \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "resnet" "CV1.1" \
    --wandb_notes="like glad-cloud-1011, but with double the number of channels"

# evaluate all the network checkpoints in a separate script after training has completed to avoid CPU OOM errors
srun --cpu-bind=threads --gpu-bind=none \
    python ../../../deep_lss/apps/run_evaluation.py \
    --evaluate_all_checkpoints \
    --dist_strategy="$STRATEGY" \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --grid_vali_tfr_pattern=$GRID_VALI_TFR

