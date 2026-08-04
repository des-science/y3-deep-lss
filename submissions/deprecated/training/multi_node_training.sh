#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=hvd_training
#SBATCH --output="./logs/v15/training_%j.log"

STRATEGY="horovod"
VERSION="v15"

# PROBE="lensing"
PROBE="clustering"
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

srun --cpu-bind=threads --gpu-bind=single:1 --output=""$OUTPUT"_training.log" \
    python ../../deep_lss/apps/run_training.py \
    --dist_strategy="$STRATEGY" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
    --dir_model="multi/v1" \
    --loss_function="$LOSS" \
    --train_tfr_pattern=$TRAIN_TFR \
    --grid_vali_tfr_pattern=$TRAIN_TFR \
    --dlss_config="configs/$VERSION/fiducial/$PROBE/dlss.yaml" \
    --net_config="configs/$VERSION/deepsphere_multi.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
    --slurm_output=""$OUTPUT"_training" \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "$SUBVERSION" "resnet" \
    --wandb_notes="multi-node start"


python ../../deep_lss/apps/run_training.py \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
    --dir_model="debug/v1" \
    --loss_function="$LOSS" \
    --train_tfr_pattern=$TRAIN_TFR \
    --grid_vali_tfr_pattern=$TRAIN_TFR \
    --dlss_config="configs/$VERSION/fiducial/$PROBE/dlss.yaml" \
    --net_config="configs/$VERSION/deepsphere_multi.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml"