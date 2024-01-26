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
#SBATCH --output="./logs/v6/training_%j.log"

VERSION="v6"
# lensing, clustering, combined
PROBE="combined"
# delta, likelihood
LOSS="delta"
# mirrored
STRATEGY="mirrored"
# linear_bias, quadratic_bias
BIAS="linear_bias"

OUTPUT="./logs/$VERSION/$PROBE/$LOSS/"$STRATEGY"_"$SLURM_JOB_ID".log"

srun --cpu-bind=threads --gpu-bind=none --output="$OUTPUT" \
    python ../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/DESy3_fiducial_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
    --net_config="configs/$VERSION/$PROBE/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
    --slurm_output="$OUTPUT" \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "100k"
