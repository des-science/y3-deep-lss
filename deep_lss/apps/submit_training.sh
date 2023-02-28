#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J training
#SBATCH -t 00:30:00
#SBATCH -A des_g

#OpenMP settings:
export OMP_NUM_THREADS=1

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1  deep_lss/apps/run_training.py --tfr_pattern=/pscratch/sd/a/athomsen/DESY3/v2/fiducial/DESy3_fiducial_???.tfrecord --net_config=configs/resnet_small.yaml --dir_base=/pscratch/sd/a/athomsen/run_files
