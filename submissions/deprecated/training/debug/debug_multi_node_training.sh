#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=hvd_training_debug
#SBATCH --output="./logs/v15/training_debug_%j.log"

STRATEGY="horovod"
VERSION="v15"

# PROBE="lensing"
# PROBE="clustering"
PROBE="combined"

# BIAS="nonlinear"
BIAS="extended"

SUBVERSION=$BIAS
# SUBVERSION=""$BIAS"_octant"
# SUBVERSION=""$BIAS"_hard_cut"

# LOSS="delta"
LOSS="mutual_info"
# LOSS="likelihood"

OUTPUT="./logs/$VERSION/$PROBE/$LOSS/"$STRATEGY"_"$SLURM_JOB_ID""

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

TRAIN_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord"
FIDU_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/fiducial/DESy3_fiducial_dmb_????.tfrecord"
GRID_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

srun --cpu-bind=threads --gpu-bind=single:1 --output=""$OUTPUT"_training.log" \
    python ../../deep_lss/apps/run_training.py \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/debug/$PROBE/$LOSS" \
    --loss_function="$LOSS" \
    --train_tfr_pattern=$TRAIN_TFR \
    --grid_vali_tfr_pattern=$TRAIN_TFR \
    --dlss_config="configs/$VERSION/fiducial/$PROBE/dlss.yaml" \
    --net_config="configs/$VERSION/deepsphere_debug.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
    --dist_strategy="$STRATEGY" \
    --slurm_output=""$OUTPUT"_training"