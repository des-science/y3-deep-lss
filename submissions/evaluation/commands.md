```
python deep_lss/apps/run_evaluation.py \
--verbosity="debug" \
--fidu_train_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v4/large_scales/fiducial/DESy3_fiducial_???.tfrecord" \
--fidu_vali_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v4/large_scales/fiducial/validation/DESy3_fiducial_???.tfrecord" \
--grid_tfr_pattern="/pscratch/sd/a/athomsen/DESY3/v4/large_scales/grid/DESy3_grid_???.tfrecord" \
--dir_model="/pscratch/sd/a/athomsen/run_files/v4/2023-08-25_05-40-02_resnet_vanilla" \
--file_label="large_scales"
```
