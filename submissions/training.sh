#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --qos=regular
#SBATCH --job-name=training
#SBATCH --time=00:10:00
#SBATCH --account=des_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=./training.%j.log

#OpenMP settings:
export OMP_NUM_THREADS=1

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1 
srun --cpu_bind=cores --gpus 4 --gpu-bind=single:1  ../deep_lss/apps/run_training.py --tfr_pattern=/pscratch/sd/a/athomsen/DESY3/v2/fiducial/DESy3_fiducial_???.tfrecord --net_config=../configs/resnet_debug.yaml --dir_base=/pscratch/sd/a/athomsen/run_files
