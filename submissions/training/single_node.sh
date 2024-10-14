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
#SBATCH --output="./logs/v11/training_%j.log"

STRATEGY="mirrored"
VERSION="v11"

# PROBE="lensing"
PROBE="clustering"
# PROBE="combined"

BIAS="extended"

SUBVERSION=$BIAS
# SUBVERSION=""$BIAS"_octant"
# SUBVERSION=""$BIAS"_hard_cut"

# LOSS="delta"
LOSS="mutual_info"
# LOSS="likelihood"

OUTPUT="./logs/$VERSION/$PROBE/$LOSS/"$STRATEGY"_"$SLURM_JOB_ID""

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

TRAIN_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord"
FIDU_VALI_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord"
GRID_VALI_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_training.log" \
    python ../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --train_tfr_pattern=$TRAIN_TFR \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/$PROBE/soft_vs_hard_cut/dlss_soft.yaml" \
    --net_config="configs/$VERSION/$PROBE/deepsphere_default.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
    --dist_strategy="$STRATEGY" \
    --slurm_output=""$OUTPUT"_training" \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "$SUBVERSION" "resnet" "CosmoGridV1.1" \
    --wandb_notes="map-level 24mpc/h soft-cut run to check the mocks and compare to a hard cut"

# evaluate all the network checkpoints in a separate script after training has completed to avoid CPU OOM errors
srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_inference.log" \
    python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="$STRATEGY" \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --grid_vali_tfr_pattern=$GRID_VALI_TFR

# srun --cpu-bind=threads --gpu-bind=none --output="{$OUTPUT}_training" \
#     python ../../deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern=$TRAIN_TFR \
#     --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v10/combined/mutual_info/2024-08-31_18-48-28_deepsphere_default" \
#     --restore_checkpoint \
#     --dist_strategy="$STRATEGY" \
#     --slurm_output="{$OUTPUT}_training" \
#     --wandb \
#     --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "resnet" "CosmoGridV1.1" \
#     --wandb_notes="Continuation of gallant-night-1058 for more steps"

# # evaluate all the network checkpoints in a separate script after training has completed to avoid CPU OOM errors
# srun --cpu-bind=threads --gpu-bind=none --output="{$OUTPUT}_inference" \
#     python ../../deep_lss/apps/run_evaluation.py \
#     --dist_strategy="$STRATEGY" \
#     --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
#     --grid_vali_tfr_pattern=$GRID_VALI_TFR

# python ../../deep_lss/apps/run_evaluation.py \
#     --dist_strategy="$STRATEGY" \
#     --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
#     --grid_vali_tfr_pattern=$GRID_VALI_TFR \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v11/extended_hard_cut/clustering/mutual_info/2024-10-12_04-01-53_deepsphere_default"
# --dir_model="/pscratch/sd/a/athomsen/run_files/v11/extended/clustering/mutual_info/2024-10-12_04-01-53_deepsphere_default"

# python ../../deep_lss/apps/run_evaluation.py \
#     --dist_strategy="$STRATEGY" \
#     --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
#     --grid_vali_tfr_pattern=$GRID_VALI_TFR \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v10/linear_bias/clustering/mutual_info/2024-10-04_08-12-55_deepsphere_default"

# python ../../deep_lss/apps/run_evaluation.py \
#     --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
#     --grid_vali_tfr_pattern=$GRID_VALI_TFR \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v10/linear_bias/clustering/mutual_info/2024-10-04_08-12-55_deepsphere_default"

# python ../../deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern=$TRAIN_TFR \
#     --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/l_max_comparison/dlss_smoothing<noise.yaml" \
#     --net_config="configs/$VERSION/$PROBE/deepsphere_default.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
#     --dist_strategy="$STRATEGY" \
#     --slurm_output=""$OUTPUT"_training"
