STRATEGY="mirrored"
VERSION="v7"
# lensing, clustering, combined
PROBE="combined"
# linear_bias, quadratic_bias
BIAS="linear_bias"
# delta, likelihood
LOSS="delta"
# LOSS="likelihood"

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

srun --nodes=1 --ntasks-per-node=1 --cpu-bind=threads --cpus-per-task=128 --gpu-bind=none --gpus-per-node=4 --gpus-per-task=4  \
    python deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_$TRAINSET_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/PASC/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/pasc/$PROBE/dlss_config.yaml" \
    --net_config="configs/$VERSION/pasc/resnet_tf.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml"