python ../../deep_lss/apps/run_training.py \
    --loss_function="$LOSS" \
    --dist_strategy="$STRATEGY" \
    --train_tfr_pattern=$TRAIN_TFR \
    --fidu_vali_tfr_pattern=$FIDU_VALI_TFR \
    --dir_base="/pscratch/sd/a/athomsen/run_files/debug/$VERSION/$PROBE/$LOSS" \
    --dlss_config="configs/$VERSION/$PROBE/$BIAS/dlss_config.yaml" \
    --net_config="configs/$VERSION/$PROBE/resnet_vanilla.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$BIAS.yaml"

python msfm/apps/perlmutter/merge_fiducial_cls.py \
    --dir_out="/pscratch/sd/a/athomsen/DESY3/v8/linear_bias/tfrecords/fiducial" \
    --config="/global/u2/a/athomsen/multiprobe-simulation-forward-model/configs/v8/linear_bias.yaml" \
    --file_suffix=""

python msfm/apps/perlmutter/merge_grid_cls.py \
    --dir_out="/pscratch/sd/a/athomsen/DESY3/v8/linear_bias/tfrecords/grid" \
    --config="/global/u2/a/athomsen/multiprobe-simulation-forward-model/configs/v8/linear_bias.yaml" \
    --file_suffix=""

tfr_pattern = filenames.get_filename_tfrecords(
    "/pscratch/sd/a/athomsen/v11desy3/v14/extended/tfrecords/grid",
    tag=conf["survey"]["name"],
    with_bary=conf["analysis"]["modelling"]["baryonified"],
    index=None,
    simset="grid",
    return_pattern=True,
)