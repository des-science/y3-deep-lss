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

# srun --cpu-bind=threads --gpu-bind=none --output="$OUTPUT" \
#     python ../../deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --dist_strategy="$STRATEGY" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_????.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_????.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
#     --net_config="configs/$VERSION/$PROBE/vit/vit_bigger_patches.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
#     --slurm_output="$OUTPUT" \
#     --wandb \
#     --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "ViT" \
#     --wandb_notes="ViT run with bigger patches, but a larger embedding dimension to compare to convolutions"
    # --dir_model="/pscratch/sd/a/athomsen/run_files/v6/combined/likelihood/2024-02-01_09-02-59_resnet_vanilla" \
    # --restore_checkpoint

srun --cpu-bind=threads --gpu-bind=none --output="$OUTPUT" \
    python ../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v7/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
    --net_config="configs/$VERSION/$PROBE/resnet_no_second_to_last.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
    --slurm_output="$OUTPUT" \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "resnet" "CV1.1" \
    --wandb_notes="like glad-cloud-1011, but with double the number of channels"


# --dir_model="/pscratch/sd/a/athomsen/run_files/v6/lensing/delta/2024-02-13_07-42-20_vit_vanilla" \
# --restore_checkpoint
# --dir_model="/pscratch/sd/a/athomsen/run_files/v6/combined/delta/2024-02-02_00-28-48_resnet_vanilla" \
# --restore_checkpoint
# --dir_model="/pscratch/sd/a/athomsen/run_files/v6/lensing_only/delta/2024-01-12_19-35-59_resnet_vanilla" \
# --dlss_config="configs/$VERSION/$PROBE/smaller_scales/dlss_config.yaml" \

# srun --cpu-bind=threads --gpu-bind=none --output="$OUTPUT" \
#     python ../../deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --dist_strategy="$STRATEGY" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_$TRAINSET_*.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
#     --net_config="configs/$VERSION/$PROBE/resnet.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
#     --slurm_output="$OUTPUT" \
#     --wandb \
#     --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "100k"

# --wandb_sweep_id="b854ginf" \
# --dir_model="/pscratch/sd/a/athomsen/run_files/v6/combined/delta/2024-02-02_00-28-48_resnet_vanilla" \
# --restore_checkpoint \

# python deep_lss/apps/run_training.py \
#     --dist_strategy="$STRATEGY" \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_$TRAINSET_*.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/debug/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
#     --net_config="configs/$VERSION/$PROBE/resnet.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml"
# --debug
# --wandb \
# --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "debug" "hyper"

# python deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --dist_strategy="$STRATEGY" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_*.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
#     --slurm_output="$OUTPUT" \
#     --wandb \
#     --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "debug" \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v6/lensing_only/delta/2024-01-12_19-35-59_resnet_vanilla" \
#     --restore_checkpoint

# python deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_*.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
#     --net_config="configs/$VERSION/$PROBE/one_d_conv.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
#     # --dist_strategy="$STRATEGY" \
#     # --slurm_output="$OUTPUT" \
#     # --wandb \
#     # --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "1d_conv" "debug"

# python deep_lss/apps/run_training.py \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_*.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
#     --net_config="configs/$VERSION/$PROBE/grapht.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
#     --wandb \
#     --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "graph_transformer" "debug"

# python deep_lss/apps/run_training.py \
#     --dist_strategy="$STRATEGY" \
#     --loss_function="$LOSS" \
#     --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_????.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_????.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$PROBE/$LOSS" \
#     --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
#     --net_config="configs/$VERSION/$PROBE/resnet.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml"
#     # --wandb \
#     # --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "validation" "debug"


python ../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v7/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
    --net_config="configs/$VERSION/$PROBE/debug/resnet_debug.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
    --slurm_output="$OUTPUT"