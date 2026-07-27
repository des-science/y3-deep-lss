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

# VERSION/SUBVERSION select the dataset. Default v17/baseline: the standard-NLA, bta-free data that
# all the per-probe defaults below (PROBE_CONFIG, CLS_PROBES_CONFIG, NET_CONFIG) are set up for.
# To use an older dataset, override these and the non-nla probe configs, e.g.:
#   VERSION=v16 SUBVERSION=rot_in_place PROBE_CONFIG=lensing CLS_PROBES_CONFIG=.../combined.yaml
VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

STRATEGY="mirrored"
DATA="default"

# SCALES may be overridden from the environment (e.g. SCALES=unsmoothed, SCALES=lmax_1024)
SCALES="${SCALES:-8wl,32gc}"

# PROBE may be overridden from the environment. Options: lensing / clustering / combined.
# PROBE names the physical probe and keys the run directory and the wandb tag.
# PROBE_CONFIG selects which configs/probes/*.yaml supplies it. Default is auto-derived for the v17
# NLA data: lensing/combined -> *_nla (bta-free, msfm extended_nla: False), clustering -> plain (no
# IA params, no _nla variant). The run still writes to maps/<PROBE> regardless. On the v16 data set
# PROBE_CONFIG=<probe> explicitly to pick the non-nla config.
PROBE="${PROBE:-lensing}"
if [ -z "${PROBE_CONFIG:-}" ]; then
    case "$PROBE" in
        lensing|combined) PROBE_CONFIG="${PROBE}_nla" ;;
        *)                PROBE_CONFIG="$PROBE" ;;
    esac
fi

# LOSS selects the VMIM variational head via configs/loss/<LOSS>.yaml (vmim = RealNVP flow,
# vmim_gmm = Gaussian-mixture). The head is NOT carried in the per-probe net config — it lives in the
# loss config, so the per-probe default is set here (2026-07-21): clustering defaults to the GMM head,
# lensing/combined to the flow. Rationale: the GMM beats the flow on clustering DES FoM (seen on both
# DeepSphere and the transformer); the flow wins or ties on the other probes. Applies to every ARCH
# (the head is orthogonal to the network). Override LOSS to force either head on any probe.
if [ -z "${LOSS:-}" ]; then
    case "$PROBE" in
        clustering) LOSS="vmim_gmm" ;;
        *)          LOSS="vmim" ;;
    esac
fi

ARCH="${ARCH:-transformer}"

PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/${PROBE_CONFIG}.yaml"

# The rebinned-Cls precache spans ALL probe pairs, so it always uses the combined probes config
# regardless of PROBE. Default is the v17 NLA variant; on v16 set CLS_PROBES_CONFIG=.../combined.yaml.
CLS_PROBES_CONFIG="${CLS_PROBES_CONFIG:-$REPOS/y3-deep-lss/configs/probes/combined_nla.yaml}"
# NET_CONFIG points at the network + training config. Default: the finalized per-probe unified recipe
# (configs/transformer/<probe>/maps+cls.yaml — flow VMIM head, per-probe 12 h budget). Override to run
# the maps-only variant or an archived bench config, e.g.:
#   NET_CONFIG=$REPOS/y3-deep-lss/configs/transformer/lensing/maps.yaml
#   NET_CONFIG=$REPOS/y3-deep-lss/configs/transformer/lensing/bench_t8/masked.yaml
NET_CONFIG="${NET_CONFIG:-$REPOS/y3-deep-lss/configs/transformer/${PROBE}/maps+cls.yaml}"

# NET_NAME (config basename, e.g. "maps+cls") tags the run in wandb and names the default run dir.
NET_NAME="$(basename "${NET_CONFIG%.yaml}")"
# MODEL_DIR is the run directory under maps/<probe>/. Defaults to the config name so a bare run is
# self-describing and will not clobber the curated t2_* runs — set it per run for anything to keep
# (e.g. MODEL_DIR=t2_cls).
MODEL_DIR="${MODEL_DIR:-$NET_NAME}"


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
        --wandb_tags "$VERSION" "$SUBVERSION" "$PROBE" "$LOSS" "$STRATEGY" "$ARCH" "$NET_NAME" "$SCALES" \
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