python deep_lss/apps/run_evaluation.py \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --grid_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --dir_model="/pscratch/sd/a/athomsen/run_files/v6/debug/2024-01-08_05-08-16_resnet_debug"

srun --output="/global/homes/a/athomsen/y3-deep-lss/run_files/mirrored_training.log" \
    python deep_lss/apps/run_training.py \
    --verbosity=debug \
    --loss_function="likelihood" \
    --dist_strategy="mirrored" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/debug" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet_debug.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml"

    python deep_lss/apps/run_training.py \
    --loss_function="likelihood" \
    --dist_strategy="mirrored" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/debug" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml" \
    --wandb \
    --wandb_tags v6 mirrored lensing debug likelihood \
    --wandb_sweep_id="wopz2i4w"

python deep_lss/apps/run_training.py \
    --loss_function="likelihood" \
    --dist_strategy="mirrored" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/debug" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml" \
    --wandb \
    --wandb_tags v6 mirrored lensing debug likelihood \
    --wandb_sweep_id="lzh3t80e"

python deep_lss/apps/run_training.py \
    --loss_function="likelihood" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/debug" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml" \
    --wandb \
    --wandb_tags v6 lensing debug likelihood


srun --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=threads --gpu-bind=single:1 \
    python deep_lss/apps/run_training.py \
    --loss_function="likelihood" \
    --dist_strategy="horovod" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/debug" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml" \
    --wandb \
    --wandb_tags v6 horovod lensing debug likelihood \
    --wandb_sweep_id="sne7gupc"

srun --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32 --cpu-bind=threads --gpu-bind=single:1 \
    python deep_lss/apps/run_training.py \
    --loss_function="delta" \
    --dist_strategy="horovod" \
    --train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/DESy3_fiducial_*.tfrecord" \
    --fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/validation/DESy3_fiducial_*.tfrecord" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/v6/debug" \
    --dlss_config="configs/v6/lensing_only/dlss_config.yaml" \
    --net_config="configs/v6/lensing_only/resnet.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/config.yaml" \
    --wandb \
    --wandb_tags v6 horovod lensing debug delta \
    --wandb_sweep_id="sne7gupc"

rsync -ahv --prune-empty-dirs \
    /pscratch/sd/a/athomsen/run_files/v6 \
    /global/cfs/cdirs/des/athomsen/deep_lss/run_files


/global/common/software/des/athomsen/dlss/bin/python -m black --line-length 119 /global/homes/a/athomsen/y3-deep-lss/deep_lss/apps/run_training.py

/global/common/software/des/athomsen/dlss/bin/python -m black --line-length 119 /global/homes/a/athomsen/y3-deep-lss/deep_lss/utils/run_training.py