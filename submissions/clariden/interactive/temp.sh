REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"

STRATEGY="mirrored"
LOSS="vmim"
SCALES="8wl,32gc"
# SCALES="unsmoothed"
DATA="default"

PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"
PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml"

# NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/bench/wide.yaml"
# NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/maps.yaml"

# NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/debug/default.yaml"
# NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/debug/dropout.yaml"
# NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/debug/layerscale.yaml"
NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/bench_t3/default.yaml"
# NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/bench_t3/dropout.yaml"
# NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/${PROBE}/bench_t3/layerscale.yaml"

MODEL="debug/t3"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

python $REPOS/y3-deep-lss/deep_lss/apps/run_training.py \
    --dir_base=$OUTPUT \
    --dir_model=$MODEL \
    --train_tfr_pattern=$TRAIN_TFR \
    --data_dir=$INPUT \
    --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
    --probes_config=$PROBES_CONFIG \
    --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
    --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
    --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
    --net_config=$NET_CONFIG \
    --dist_strategy="$STRATEGY"
