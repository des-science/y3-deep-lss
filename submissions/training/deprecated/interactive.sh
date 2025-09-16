STRATEGY="mirrored"
VERSION="v7"
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

python deep_lss/apps/run_training.py \
    --dist_strategy="$STRATEGY" \
    --loss_function="$LOSS" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_????.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_????.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
    --net_config="configs/$VERSION/$PROBE/resnet_debug.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" 
    
    
    
    \
    --mixed_precision \
    --xla
    # --net_config="configs/$VERSION/$PROBE/vit_debug.yaml" \





STRATEGY="horovod"
VERSION="v7"
# lensing, clustering, combined
PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"
# linear_bias, quadratic_bias
BIAS="linear_bias"
# delta, likelihood
# LOSS="delta"
LOSS="likelihood"

OUTPUT="./logs/$VERSION/$PROBE/$LOSS/"$STRATEGY"_"$SLURM_JOB_ID".log"

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

export HOROVOD_ENABLE_XLA_OPS=1

srun --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=threads --gpus-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
    python deep_lss/apps/run_training.py \
        --dist_strategy="$STRATEGY" \
        --loss_function="$LOSS" \
        --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_${TRAINSET}_????.tfrecord" \
        --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_????.tfrecord" \
        --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_????.tfrecord" \
        --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$PROBE/$LOSS" \
        --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
        --net_config="configs/$VERSION/$PROBE/vit_debug.yaml" \
        --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml" \
        --mixed_precision \
        --xla
    # --net_config="configs/$VERSION/$PROBE/vit_debug.yaml" \




