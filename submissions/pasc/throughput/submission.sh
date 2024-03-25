#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=hvd_throughput
#SBATCH --output="./logs/hvd_troughput_%j.log"

STRATEGY="horovod"
VERSION="v7"
# lensing, clustering, combined
PROBE="combined"
# linear_bias, quadratic_bias
BIAS="linear_bias"
# delta, likelihood
LOSS=$1
WANDB_NOTES=$2

OUTPUT="./logs/$VERSION/$PROBE/$LOSS/"$STRATEGY"_"$SLURM_JOB_ID".log"

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

srun --cpu-bind=threads --gpu-bind=single:1 --output="$OUTPUT" \
    python ../../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_????.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_????.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/pasc/$PROBE/dlss_config.yaml" \
    --net_config="configs/$VERSION/pasc/resnet_hvd.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
    --slurm_output="$OUTPUT" \
    --pasc_throughput \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "pasc" "throughput" \
    --wandb_notes="$WANDB_NOTES"

# srun --nodes=1 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=threads --gpus-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
#     python deep_lss/apps/run_training.py \
#     --loss_function="delta" \
#     --dist_strategy="$STRATEGY" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/DESy3_fiducial_????.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_????.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/pasc/$PROBE/dlss_config.yaml" \
#     --net_config="configs/$VERSION/pasc/resnet_hvd.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
#     --pasc_throughput 
