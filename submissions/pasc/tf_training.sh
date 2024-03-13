#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=tf_training
#SBATCH --output="./logs/training_%j.log"

STRATEGY="mirrored"
VERSION="v7"
# lensing, clustering, combined
PROBE="combined"
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

srun --cpu-bind=threads --gpu-bind=none --output="$OUTPUT" \
    python ../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_????.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_????.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/pasc/dlss_config.yaml" \
    --net_config="configs/$VERSION/pasc/resnet_tf.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
    --slurm_output="$OUTPUT" \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "pasc" "tf"
    # --wandb_notes="Run with dropout, the new and improved validation tracking and reduced jacobian weight"
