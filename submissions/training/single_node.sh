#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=training
#SBATCH --output="./logs/v15/training_%j.log"

STRATEGY="mirrored"
VERSION="v15"

PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"

# BIAS="nonlinear"
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
FIDU_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/fiducial/DESy3_fiducial_dmb_????.tfrecord"
GRID_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_training.log" \
    python ../../deep_lss/apps/run_training.py \
    --dist_strategy="$STRATEGY" \
    --loss_function="$LOSS" \
    --train_tfr_pattern=$TRAIN_TFR \
    --grid_vali_tfr_pattern=$TRAIN_TFR \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
    --slurm_output=""$OUTPUT"_training" \
    --dlss_config="configs/$VERSION/$PROBE/smoothing_fwhm/dlss_8mpc.yaml" \
    --net_config="configs/$VERSION/deepsphere_default.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "$SUBVERSION" "resnet" "CosmoGridV1.1" \
    --wandb_notes="single node lensing start"

# evaluate all the network checkpoints in a separate script after training has completed to avoid CPU OOM errors
srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_inference.log" \
    python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="$STRATEGY" \
    --fidu_vali_tfr_pattern=$FIDU_EVAL_TFR \
    --grid_vali_tfr_pattern=$GRID_EVAL_TFR
# --evaluate_all_checkpoints
# --fidu_vali_tfr_pattern=$FIDU_EVAL_TFR \

# python ../../deep_lss/apps/run_evaluation.py \
#     --dist_strategy="$STRATEGY" \
#     --fidu_vali_tfr_pattern=$FIDU_EVAL_TFR \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default"
# --dir_model="/pscratch/sd/a/athomsen/run_files/v14/extended/clustering/mutual_info/2025-05-14_23-10-45_deepsphere_default"
# --dir_model="/pscratch/sd/a/athomsen/run_files/v14/extended/lensing/mutual_info/2025-04-19_18-54-31_deepsphere_default"

# --grid_vali_tfr_pattern=$GRID_EVAL_TFR \
# python ../../deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern=$TRAIN_TFR \
#     --grid_vali_tfr_pattern=$TRAIN_TFR \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/smoothing_fwhm/dlss_8mpc.yaml" \
#     --net_config="configs/$VERSION/deepsphere_debug.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
#     --dist_strategy="$STRATEGY"

# python ../../deep_lss/apps/run_evaluation.py \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v13/extended/clustering/mutual_info/2025-02-01_01-16-48_deepsphere_default" \
#     --dist_strategy="$STRATEGY" \
#     --grid_vali_tfr_pattern=$GRID_EVAL_TFR
# --evaluate_all_checkpoints

# python ../../deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern=$TRAIN_TFR \
#     --grid_vali_tfr_pattern=$TRAIN_TFR \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/smoothing_fwhm/dlss_8mpc.yaml" \
#     --net_config="configs/$VERSION/deepsphere_debug.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
#     --dist_strategy="$STRATEGY" \
#     --slurm_output=""$OUTPUT"_training"
