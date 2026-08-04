STRATEGY="horovod"
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

srun --ntasks=1 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=threads --gpus-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
    python deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/$TRAINSET/DESy3_$TRAINSET_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/$VERSION/$BIAS/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/pasc/debug/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/pasc/$PROBE/dlss_config.yaml" \
    --net_config="configs/$VERSION/pasc/resnet_hvd.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml"


sbatch --nodes=1 --ntasks=2 hvd_training.sh "1 node, 2 GPUs"
sbatch --nodes=1 --ntasks=4 hvd_training.sh "1 node, 4 GPUs"
sbatch --nodes=2 --ntasks=8 hvd_training.sh "2 nodes, 8 GPUs"
sbatch --nodes=8 --ntasks=32 --time=00:10:00 hvd_training.sh "8 nodes, 32 GPUs"