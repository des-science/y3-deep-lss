#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=trainingx
#SBATCH --output=./logs/training.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

srun --cpus-per-task=128 --cpu_bind=threads --gpu-bind=none \
    python ../../deep_lss/apps/run_training.py \
    --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/fiducial/DESy3_fiducial_???.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/fiducial/validation/DESy3_fiducial_???.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/grid/DESy3_grid_???.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v5/quadratic_bias_v2/debug" \
    --net_config="configs/v5/combined_probes/resnet_vanilla.yaml" \
    --dlss_config="configs/v5/combined_probes/quadratic_bias/dlss_config.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v5/quadratic_bias.yaml"


# dataset = "linear_bias"

# srun --cpus-per-task=128 --cpu_bind=threads --gpu-bind=none \
#     python ../../deep_lss/apps/run_training.py \
#     --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/"$dataset"/tfrecords/fiducial/DESy3_fiducial_???.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/"$dataset"/tfrecords/fiducial/validation/DESy3_fiducial_???.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/"$dataset"/tfrecords/grid/DESy3_grid_???.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/v5/"$dataset"" \
#     --net_config="configs/v5/clustering_only/resnet_vanilla.yaml" \
#     --dlss_config="configs/v5/clustering_only/"$dataset"/dlss_config.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v5/"$dataset".yaml"

# linear bias
# srun --cpus-per-task=128 --cpu_bind=threads --gpu-bind=none \
#     python ../../deep_lss/apps/run_training.py \
#     --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/fiducial/DESy3_fiducial_???.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_???.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/grid/DESy3_grid_???.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/v5/linear_bias" \
#     --net_config="configs/v5/clustering_only/resnet_vanilla.yaml" \
#     --dlss_config="configs/v5/clustering_only/linear_bias/dlss_config.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v5/linear_bias.yaml"

# quadratic_bias_v2
# srun --cpus-per-task=128 --cpu_bind=threads --gpu-bind=none \
#     python ../../deep_lss/apps/run_training.py \
#     --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/fiducial/DESy3_fiducial_???.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/fiducial/validation/DESy3_fiducial_???.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/quadratic_bias_v2/tfrecords/grid/DESy3_grid_???.tfrecord" \
#     --dir_model="/pscratch/sd/a/athomsen/run_files/v5/quadratic_bias_v2/2023-10-03_00-16-59_resnet_vanilla" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/v5/quadratic_bias_v2" \
#     --net_config="configs/v5/clustering_only/resnet_vanilla.yaml" \
#     --dlss_config="configs/v5/clustering_only/quadratic_bias/dlss_config.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v5/quadratic_bias.yaml"
    # --restore_checkpoint

# stochasticity
# srun --cpus-per-task=128 --cpu_bind=threads --gpu-bind=none \
#     python ../../deep_lss/apps/run_training.py \
#     --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/stochasticity/tfrecords/fiducial/DESy3_fiducial_???.tfrecord" \
#     --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/stochasticity/tfrecords/fiducial/validation/DESy3_fiducial_???.tfrecord" \
#     --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v5/stochasticity/tfrecords/grid/DESy3_grid_???.tfrecord" \
#     --dir_base="/pscratch/sd/a/athomsen/run_files/v5/stochasticity" \
#     --net_config="configs/v5/clustering_only/resnet_vanilla.yaml" \
#     --dlss_config="configs/v5/clustering_only/linear_bias/dlss_config.yaml" \
#     --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v5/stochasticity.yaml"
