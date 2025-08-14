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
VERSION="v10"

PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"

BIAS="linear_bias"
# BIAS="quadratic_bias"

# LOSS="delta"
LOSS="mutual_info"
# LOSS="likelihood"

OUTPUT="./logs/$VERSION/$PROBE/$LOSS/"$STRATEGY"_"$SLURM_JOB_ID".log"

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

TRAIN_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord"
FIDU_VALI_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord"
GRID_VALI_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=none --output="$OUTPUT" \
    python ../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --train_tfr_pattern=$TRAIN_TFR \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --dist_strategy="$STRATEGY" \
    --slurm_output="$OUTPUT" \
    --dir_model="/pscratch/sd/a/athomsen/run_files/v10/clustering/mutual_info/2024-08-28_00-40-26_deepsphere_default" \
    --restore_checkpoint \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "resnet" "CosmoGridV1.1" \
    --wandb_notes="Continued training of deft-puddle-1049 10"

# evaluate all the network checkpoints in a separate script after training has completed to avoid CPU OOM errors
srun --cpu-bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="$STRATEGY" \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --grid_vali_tfr_pattern=$GRID_VALI_TFR

# TRAIN_TFR="/pscratch/sd/a/athomsen/v11desy3/debug/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord"

# TRAIN_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord"

# python deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern=$TRAIN_TFR \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/dlss_config.yaml" \
#     --net_config="configs/$VERSION/$PROBE/deepsphere_default.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
#     --dist_strategy="$STRATEGY"

# python ../../deep_lss/apps/run_evaluation.py \
#     --dist_strategy="$STRATEGY" \
#     --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
#     --grid_vali_tfr_pattern=$GRID_VALI_TFR \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v10/clustering/delta/2024-08-28_02-52-15_deepsphere_default"
# --dir_model="/pscratch/sd/a/athomsen/run_files/v10/lensing/mutual_info/2024-08-28_08-51-37_deepsphere_default"
# --dir_model="/pscratch/sd/a/athomsen/run_files/v10/clustering/mutual_info/2024-08-28_00-40-26_deepsphere_default"
# --dir_model="/pscratch/sd/a/athomsen/run_files/v10/lensing/mutual_info/2024-08-26_06-07-27_deepsphere_default"
