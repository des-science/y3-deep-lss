#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# disable core dumps: a crashing TF/Python task can otherwise write a 50+ GB
# core file into the cwd (core_pattern=core_%h_%p) and fill the /users quota
ulimit -c 0

# extract Weights & Biases API key from the host's .netrc file and pass it as an environment variable
# to accommodate containerized execution that might not inherit the host's home directory mounts properly.
export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc)

# Optimize OpenMP and TensorFlow thread pools for the 288 available CPU cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

RUN_NUM=${RUN_NUM:-1}
# add --restore_checkpoint only for RUN_NUM > 1
RESTORE_FLAG=""
if [ "$RUN_NUM" -gt 1 ]; then
    RESTORE_FLAG="--restore_checkpoint"
fi

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"

STRATEGY="mirrored"
LOSS="vmim"
DATA="default"

# SCALES may be overridden from the environment (e.g. SCALES=unsmoothed, SCALES=lmax_1024)
SCALES="${SCALES:-8wl,32gc}"

# PROBE may be overridden from the environment. Options: lensing / clustering / combined
PROBE="${PROBE:-lensing}"

ARCH="transformer"

# MODEL selects a bench_t7 config: the bench_t6/default.yaml base (the winning architecture) plus
# one feature toggle. Options: dropout / masked / multiscale / pool / symmetric, plus the symmetric
# follow-ups symmetric_init (non-zero RBF init of the scalar distance kernel) and symmetric_binned
# (distance-binned learnable bias), and their concat counterparts geodesic / geodesic_binned (the same
# RBF-init distance biases but ON TOP of the order-sensitive concat merge, completing a 2x2
# {concat,deepsets} x {geodesic,geodesic_binned} factorial). Each config carries its own
# local_batch_size and n_steps (sized for ~11 h; the symmetric*/geodesic* variants run at batch 18
# because the distance-bias pathway pushes batch 20 into the ~89.6 GB NCCL time-bomb zone, the rest at 20).
# Override MODEL from the environment to sweep the variants.
MODEL="${MODEL:-dropout}"

PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml"
# NET_CONFIG and MODEL_DIR default to the bench_t7 sweep but may be overridden from the environment,
# e.g. to run the winning base (bench_t6/default.yaml) under a different scale cut:
#   NET_CONFIG=.../bench_t6/default.yaml SCALES=unsmoothed MODEL_DIR=default_unsmoothed
NET_CONFIG="${NET_CONFIG:-$REPOS/y3-deep-lss/configs/transformer/lensing/bench_t7/${MODEL}.yaml}"

MODEL_DIR="${MODEL_DIR:-t7_${MODEL}}"


INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

# maps+cls runs need a rebinned-Cls asinh-calibration cache keyed on the scales name. Unlike
# cls_training.sh, run_training.py only READS this cache — it aborts if it is missing rather than
# building it. Build it here on the full node before training when the net_config carries a cls block
# and the cache for this SCALES is absent. Idempotent: the app returns immediately if the file exists.
CLS_N_BINS=$(grep -E '^\s*n_bins:' "$NET_CONFIG" | head -1 | grep -oE '[0-9]+')
CLS_N_BINS=${CLS_N_BINS:-16}
CLS_CACHE="$INPUT/cls/rebinned_nb${CLS_N_BINS}_${SCALES}.h5"
if grep -qE '^\s*cls:' "$NET_CONFIG" && [ ! -f "$CLS_CACHE" ]; then
    echo "maps+cls run: Cls cache $CLS_CACHE missing — building it before training."
    srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_precache.log" \
        python $REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py \
            --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
            --probes_config="$REPOS/y3-deep-lss/configs/probes/combined.yaml" \
            --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
            --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
            --net_config="$REPOS/y3-deep-lss/configs/cls/mlp.yaml" \
            --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
            --data_dir="$INPUT" \
            --out_dir="$INPUT" \
            --model_name="precache" \
            --precache_only
    sleep 10
fi

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_training.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_training.py \
        --dir_base=$OUTPUT \
        --dir_model=$MODEL_DIR \
        --train_tfr_pattern=$TRAIN_TFR \
        --data_dir=$INPUT \
        --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --probes_config=$PROBES_CONFIG \
        --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
        --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
        --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
        --net_config=$NET_CONFIG \
        --dist_strategy="$STRATEGY" \
        --wandb \
        --wandb_tags "$VERSION" "$SUBVERSION" "$PROBE" "$LOSS" "$STRATEGY" "$ARCH" "$MODEL" "$SCALES" \
        $RESTORE_FLAG

sleep 30

srun --environment=tensorflow --gpu-bind=none --output=""$LOG"_evaluation.log" \
    python $REPOS/y3-deep-lss/deep_lss/apps/run_evaluation.py \
        --dist_strategy="$STRATEGY" \
        --grid_vali_tfr_pattern=$TRAIN_TFR \
        --data_dir=$INPUT \
        --include_grid \
        --include_des \
        --include_mocks

sleep 30

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output=""$LOG"_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL_DIR\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        --sample_posterior \
        --include_grid \
        --include_des \
        --include_mocks"