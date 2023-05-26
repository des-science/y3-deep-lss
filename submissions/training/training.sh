#!/bin/bash
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=training
#SBATCH --output=./logs/training.%j.log

#OpenMP settings:
# export OMP_NUM_THREADS=128
# export OMP_PLACES=threads
# export OMP_PROC_BIND=close

#run the application:
srun python ../../deep_lss/apps/run_training.py \
    --fidu_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/fiducial/DESy3_fiducial_???.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/fiducial/validation/DESy3_fiducial_???.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/grid/DESy3_grid_???.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v3" \
    --restore_checkpoint \
    --dir_model="2023-05-24_05-18-42_resnet_vanilla"
# --net_config="configs/resnet_vanilla.yaml" \

# srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/fiducial/DESy3_fiducial_???.tfrecord" --grid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/grid/DESy3_grid_???.tfrecord" --net_config="/global/u2/a/athomsen/y3-deep-lss/configs/resnet_vanilla.yaml" --dir_base="/pscratch/sd/a/athomsen/run_files/v3/"
# srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/fiducial/DESy3_fiducial_???.tfrecord" --grid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/grid/DESy3_grid_???.tfrecord" --net_config="/global/u2/a/athomsen/y3-deep-lss/configs/resnet_vanilla.yaml" --dir_model="/pscratch/sd/a/athomsen/run_files/v3/2023-05-24_05-26-55_resnet_vanilla" --restore_checkpoint
# srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern=${1} --net_config=${2} --dir_base=${3}
# srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/fiducial/DESy3_fiducial_???.tfrecord" --net_config="/global/u2/a/athomsen/y3-deep-lss/configs/resnet_vanilla.yaml" --dir_model="/pscratch/sd/a/athomsen/run_files/v3/2023-05-15_07-04-57_resnet_vanilla" --restore_checkpoint
# srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v3/DESy3_fiducial_???.tfrecord" --net_config="/global/u2/a/athomsen/y3-deep-lss/configs/probe_combination/resnet_vanilla.yaml" --dlss_config="/global/u2/a/athomsen/y3-deep-lss/configs/probe_combination/dlss_config.yaml" --dir_base="/pscratch/sd/a/athomsen/run_files/probe_combination"

# recommendation from https://my.nersc.gov/script_generator.php. This can be simplified because there is only a single task
# srun --ntasks=1 --cpus-per-task=128 --cpu_bind=cores --gpus 4 --gpu-bind=single:1 python ../deep_lss/apps/run_training.py --tfr_pattern=${1} --net_config=${2} --dir_base=${3}
