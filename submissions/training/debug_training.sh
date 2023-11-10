#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=debug_training
#SBATCH --output=./logs/debug_training.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=128
# export OMP_PLACES=threads
# export OMP_PROC_BIND=close

#run the application:
srun --cpus-per-task=128 --cpu_bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_training.py \
    --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/fiducial/DESy3_fiducial_???.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/fiducial/validation/DESy3_fiducial_???.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/grid/DESy3_grid_???.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v5/debug" \
    --net_config="configs/v5/clustering_only/resnet_vanilla.yaml" \
    --dlss_config="configs/v5/clustering_only/quadratic_bias/dlss_config.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v5/quadratic_bias.yaml"
