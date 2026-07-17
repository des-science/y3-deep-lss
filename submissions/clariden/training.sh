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

# PROFILE=1 traces steps 800->805 into the run's summary dir (run_training.py --profile). Use it for
# one-off step-time diagnostics only; leave it unset for production runs (it perturbs timing and the
# run may be short-circuited manually once the trace is captured).
PROFILE_FLAG=""
if [ "${PROFILE:-0}" = "1" ]; then
    PROFILE_FLAG="--profile"
fi

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

# VERSION/SUBVERSION may be overridden from the environment to point at a different dataset,
# e.g. VERSION=v17 SUBVERSION=baseline (the standard-NLA, bta-free dataset — pair it with the
# *_nla probes configs via PROBE_CONFIG=lensing_nla / combined_nla, and
# CLS_PROBES_CONFIG=.../combined_nla.yaml).
VERSION="${VERSION:-v16}"
SUBVERSION="${SUBVERSION:-rot_in_place}"

STRATEGY="mirrored"
# LOSS may be overridden from the environment (e.g. LOSS=vmim_gmm for the GMM-head control)
LOSS="${LOSS:-vmim}"
DATA="default"

# SCALES may be overridden from the environment (e.g. SCALES=unsmoothed, SCALES=lmax_1024)
SCALES="${SCALES:-8wl,32gc}"

# PROBE may be overridden from the environment. Options: lensing / clustering / combined
# PROBE names the physical probe and is what the run directory and the wandb tag are keyed on.
# PROBE_CONFIG selects which configs/probes/*.yaml supplies it, defaulting to the probe name. Split
# the two when a dataset needs a variant config but should keep the standard run layout -- e.g. on
# the bta-free v17+ data (msfm extended_nla: False):
#   VERSION=v17 SUBVERSION=baseline PROBE=lensing PROBE_CONFIG=lensing_nla
# trains from lensing_nla.yaml but still writes to maps/lensing.
PROBE="${PROBE:-lensing}"
PROBE_CONFIG="${PROBE_CONFIG:-$PROBE}"

ARCH="transformer"

# MODEL selects a bench_t8 config: the bench_t6/default.yaml winning architecture, now with the
# cosine LR schedule (cosine did well back in bench_t4; bench_t6/t7 used constant LR). Options:
#   flat        — the FLAT-schedule control (scheduler Null); the t8_flat run dir is a copy of the
#                 finished t7_default run, so no retrain is needed (run MODEL=flat only to reproduce).
#   default     — flat's architecture + cosine schedule.
#   geodesic    — default arch + scalar geodesic distance-kernel attention bias (concat merge), + cosine.
#   masked      — default arch + masked attention, + cosine.
#   multiscale  — default arch + per-stage multi-scale readout, + cosine.
# Each config carries its own local_batch_size and n_steps (sized for ~11 h; geodesic runs at batch 18
# because the distance-bias pathway pushes batch 20 into the ~89.6 GB NCCL time-bomb zone, the rest at 20).
# Override MODEL from the environment to sweep the variants.
MODEL="${MODEL:-default}"

PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/${PROBE_CONFIG}.yaml"

# The rebinned-Cls precache below spans ALL probe pairs, so it uses the combined probes config
# regardless of PROBE. Override for bta-free datasets: CLS_PROBES_CONFIG=.../combined_nla.yaml (v17+).
CLS_PROBES_CONFIG="${CLS_PROBES_CONFIG:-$REPOS/y3-deep-lss/configs/probes/combined.yaml}"
# NET_CONFIG and MODEL_DIR default to the bench_t8 sweep but may be overridden from the environment,
# e.g. to run the winning base under a different scale cut, or the combined multi-res maps+cls config:
#   NET_CONFIG=.../bench_t6/default.yaml SCALES=unsmoothed MODEL_DIR=default_unsmoothed
#   PROBE=combined NET_CONFIG=.../combined/maps+cls.yaml MODEL_DIR=combined_maps+cls
NET_CONFIG="${NET_CONFIG:-$REPOS/y3-deep-lss/configs/transformer/lensing/bench_t8/${MODEL}.yaml}"

MODEL_DIR="${MODEL_DIR:-t8_${MODEL}}"


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
            --probes_config="$CLS_PROBES_CONFIG" \
            --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
            --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
            --net_config="$REPOS/y3-deep-lss/configs/cls/mlp/default.yaml" \
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
        $RESTORE_FLAG $PROFILE_FLAG

# SKIP_EVAL=1 stops after training and skips the evaluation + inference tail. Use it for short
# benchmark/profiling jobs where the model is undertrained and eval/inference would only waste the
# allocation. Leave unset for production runs.
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    echo "SKIP_EVAL=1: skipping evaluation and inference."
    exit 0
fi

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

# --cpu-bind=none: this is the only step that reshapes the task to a 1-GPU / 72-CPU sub-allocation.
# Without it SLURM maps the 72 CPUs to a single socket's mask and the default bind fails to launch
# ("Unable to satisfy cpu bind request" -> step CANCELLED at 00:00:00, empty inference log), the same
# way the training/eval steps pass --gpu-bind=none. Surfaced on jobs submitted with
# --uenv-passthrough=ignore (uenv session), where the step also requests --uenv=pytorch.
srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G --cpu-bind=none \
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