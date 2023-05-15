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
export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=close

#run the application:
srun python ../../deep_lss/apps/run_training.py --fid_tfr_pattern=${1} --net_config=${2} --dir_base=${3}

# recommendation from https://my.nersc.gov/script_generator.php. This can be simplified because there is only a single task
# srun --ntasks=1 --cpus-per-task=128 --cpu_bind=cores --gpus 4 --gpu-bind=single:1 python ../deep_lss/apps/run_training.py --tfr_pattern=${1} --net_config=${2} --dir_base=${3}
