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
#SBATCH --output="./logs/v15/training_%j.log"

RUN_NUM=${RUN_NUM:-1}
STRATEGY="mirrored"
VERSION="v15"

# PROBE="lensing"
# PROBE="clustering"
PROBE="combined"

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

# Add --restore_checkpoint only for RUN_NUM > 1
RESTORE_FLAG=""
if [ "$RUN_NUM" -gt 1 ]; then
    RESTORE_FLAG="--restore_checkpoint"
fi


srun --cpu-bind=threads --gpu-bind=none --output=""$OUTPUT"_training.log" \
    python ../../deep_lss/apps/run_training.py \
        --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
        --dir_model="default/v1" \
        --loss_function="$LOSS" \
        --train_tfr_pattern=$TRAIN_TFR \
        --grid_vali_tfr_pattern=$TRAIN_TFR \
        --dlss_config="configs/$VERSION/default/$PROBE/dlss.yaml" \
        --net_config="configs/$VERSION/deepsphere_default.yaml" \
        --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --slurm_output="${OUTPUT}_training" \
        --dist_strategy="$STRATEGY" \
        --wandb \
        --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "$SUBVERSION" "resnet" \
        --wandb_notes="single $PROBE node run $RUN_NUM" \
        --restore_checkpoint \
        # $RESTORE_FLAG \

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
#     --dir_model="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS/default/v1" \
#     --grid_vali_tfr_pattern=$GRID_EVAL_TFR

# python ../../deep_lss/apps/run_training.py \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
#     --dir_model="default/v1" \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern=$TRAIN_TFR \
#     --grid_vali_tfr_pattern=$TRAIN_TFR \
#     --dlss_config="configs/$VERSION/default/$PROBE/dlss.yaml" \
#     --net_config="configs/$VERSION/deepsphere_default.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
#     --slurm_output="${OUTPUT}_training" \
#     --dist_strategy="$STRATEGY" \
#     --restore_checkpoint \