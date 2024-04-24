#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=mirrored_eval
#SBATCH --output=./logs/mirrored_eval.%j.log

VERSION="v8"
# linear_bias, quadratic_bias
BIAS="linear_bias"

srun --cpu-bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="mirrored" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v7/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
    --dir_model="/pscratch/sd/a/athomsen/run_files/v8/lensing/delta/2024-04-22_06-50-39_resnet_vanilla" \
    --file_label="sbatch"

    # --fidu_train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/DESy3_fiducial_dmb_????.tfrecord" \
    # --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
# python deep_lss/apps/run_evaluation.py \
#     --fidu_train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/DESy3_fiducial_*.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v7/lensing/delta/2024-03-14_10-10-52_resnet_vanilla"

# srun --cpu-bind=threads --gpu-bind=none \
#     python ../../deep_lss/apps/run_evaluation.py \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v7/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v8/lensing/delta/2024-04-22_06-50-39_resnet_vanilla"

python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="mirrored" \
    --evaluate_all_checkpoints \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord" \
    --dir_model="/pscratch/sd/a/athomsen/run_files/v8/lensing/delta/2024-04-22_06-50-39_resnet_vanilla" \
    --file_label="test"

# python ../../deep_lss/apps/run_evaluation.py \
#     --dist_strategy="mirrored" \
#     --evaluate_all_checkpoints \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v7/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v8/lensing/delta/2024-04-22_06-50-39_resnet_vanilla" 

python ../../deep_lss/apps/run_evaluation.py \
    --dist_strategy="mirrored" \
    --evaluate_all_checkpoints \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_dmb_????.tfrecord" \
    --file_label="test"
