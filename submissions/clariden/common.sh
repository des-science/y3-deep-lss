# Sourced by every script in this tree: roots, the dataset and its paths, shared stages.
# Source it by absolute path; sbatch runs a spool copy of the calling script.

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"  # the benchmarks' claude/bench outputs; datasets use their locations
DEEP_LSS="$REPOS/y3-deep-lss"
MSFM="$REPOS/multiprobe-simulation-forward-model"
MSI="$REPOS/multiprobe-simulation-inference"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc 2>/dev/null)

# DRYRUN=1 prints every srun/sbatch instead of running it (works on the login node).
DRYRUN="${DRYRUN:-0}"
SRUN="srun"
[ "$DRYRUN" = "1" ] && SRUN="echo srun"
SLURM_JOB_ID="${SLURM_JOB_ID:-dryrun}"

# --- Dataset -----------------------------------------------------------------------------------

# The dataset is msfm's configs/<VERSION>/<version>.yaml, e.g. VERSION=desy6/v2 for configs/desy6/v2/v2.yaml. VERSION
# has no default, so a run never falls back to another survey's data. DES Y3 v18 is VERSION=desy3/v18; v18 and v16
# are extended-NLA (plain configs/probes/desy3/*.yaml), v17 needs the *_nla probe configs.
SCALES="${SCALES:-8wl,32gc}"  # configs/scales/<VERSION>/
LOSS="${LOSS:-vmim}"          # configs/loss/
DATA="${DATA:-default}"       # configs/data/

datasets() { ls "$MSFM"/configs/*/*/locations.yaml 2>/dev/null | sed -E "s#^$MSFM/configs/(.*)/locations.yaml#\1#" | xargs; }
if [ -z "${VERSION:-}" ]; then
    echo "Set VERSION to a dataset of msfm's configs/: $(datasets)" >&2
    exit 1
fi
MSFM_CONFIG="$MSFM/configs/$VERSION/$(basename "$VERSION").yaml"
if [ ! -f "$MSFM_CONFIG" ]; then
    echo "No msfm config $MSFM_CONFIG; the datasets are: $(datasets)" >&2
    exit 1
fi
SURVEY=$(awk '/^survey:/{s=1} s&&/^  name:/{print $2; exit}' "$MSFM_CONFIG")
[ -n "$SURVEY" ] || { echo "$MSFM_CONFIG has no survey.name" >&2; exit 1; }

# The dataset's paths come from the locations.yaml next to $MSFM_CONFIG (msfm.utils.locations); an environment
# variable overrides one. INPUT holds tfrecords/ and cls/, OBS_DIR the *_obs_maps.h5 mocks, RUNS_ROOT the runs.
if [ -n "${OBS_INPUT:-}" ]; then
    echo "OBS_INPUT is replaced by OBS_DIR, the directory of the *_obs_maps.h5 files itself." >&2
    exit 1
fi
LOCATIONS_FILE=$(mktemp)
if [ "$DRYRUN" = "1" ]; then
    uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python \
        -m msfm.utils.locations "$MSFM_CONFIG" > "$LOCATIONS_FILE"
else
    srun -N1 -n1 --environment=tensorflow --gpu-bind=none --cpu-bind=none --output="$LOCATIONS_FILE" \
        python -m msfm.utils.locations "$MSFM_CONFIG"
fi || { cat "$LOCATIONS_FILE" >&2; echo "Cannot resolve the locations of $MSFM_CONFIG." >&2; exit 1; }
location() { sed -n "s|^$1=||p" "$LOCATIONS_FILE"; }
INPUT="${INPUT:-$(location data_dir)}"
OBS_DIR="${OBS_DIR:-$(location obs_dir)}"
RUNS_ROOT="${RUNS_ROOT:-$(location runs_dir)}"
rm -f "$LOCATIONS_FILE"
if [ -z "$INPUT" ] || [ -z "$OBS_DIR" ] || [ -z "$RUNS_ROOT" ]; then
    echo "The locations of $MSFM_CONFIG give no data, obs or runs; set INPUT, OBS_DIR and RUNS_ROOT." >&2
    exit 1
fi
TRAIN_TFR="${TRAIN_TFR:-$INPUT/tfrecords/grid/${SURVEY}_grid_dmb_????.tfrecord}"

# the probes configs are per survey (they name its bias parameters), the scale cuts per dataset version
PROBES_DIR="$DEEP_LSS/configs/probes/${SURVEY,,}/"
SCALES_CONFIG="$DEEP_LSS/configs/scales/$VERSION/$SCALES.yaml"
LOSS_CONFIG="$DEEP_LSS/configs/loss/$LOSS.yaml"
DATA_CONFIG="$DEEP_LSS/configs/data/$DATA.yaml"

# --- Helpers -----------------------------------------------------------------------------------

# A dry run must not create run directories.
make_log_dir() { [ "$DRYRUN" = "1" ] || mkdir -p "$1"; }

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# Pause between two stages of one job, so the previous step has released its GPUs.
settle() { [ "$DRYRUN" = "1" ] || sleep 30; }

# refuse_existing_preds <run dir>...: eval and infer change a run's predictions and chains, so a run that already holds
# predictions (the frozen production runs among them) needs OVERWRITE=1.
refuse_existing_preds() {
    [ "${OVERWRITE:-0}" = "1" ] && return 0
    local dir
    for dir in "$@"; do
        if compgen -G "$dir/preds_*.h5" > /dev/null; then
            echo "$dir already holds predictions, which eval and infer would change; set OVERWRITE=1." >&2
            exit 1
        fi
    done
}

# fresh_run <run dir>...: training from scratch. An existing run needs OVERWRITE=1 and is then deleted whole; its
# checkpoints would make the new network refuse to save, and its budget record would shorten the new run.
fresh_run() {
    local dir
    for dir in "$@"; do
        if [ "${OVERWRITE:-0}" = "1" ]; then
            [ -d "$dir" ] || continue
            echo "OVERWRITE=1: deleting the old run $dir"
            [ "$DRYRUN" = "1" ] || rm -rf "$dir"
        elif [ -f "$dir/configs.yaml" ]; then
            echo "$dir already holds a run; choose another RUN, or set OVERWRITE=1." >&2
            exit 1
        fi
    done
}

# Call directly after the srun it checks, while $? is still that step's status.
check_stage() {
    [ "$1" -eq 0 ] && return 0
    echo "$2 failed (exit $1), see $3. Aborting." >&2
    exit "$1"
}

# resolve_net <config> <out>: `dotted.key=value` dump with extends: resolved, path in RESOLVED.
# Never grep a net config itself; inherited keys are not in the file.
resolve_net() {
    RESOLVED=$2
    if [ "$DRYRUN" = "1" ]; then
        RESOLVED=$(mktemp)
        uenv run --view=default pytorch/v2.9.1:v2 -- ~/dlss/torch_env/bin/python \
            -m deep_lss.utils.config_check resolve "$1" --flat > "$RESOLVED"
    else
        srun -N1 -n1 --environment=tensorflow --gpu-bind=none --cpu-bind=none --output="$RESOLVED" \
            python -m deep_lss.utils.config_check resolve "$1" --flat
    fi
}

# Net config the rebinned-Cls cache is built from when a maps+cls config needs it.
CLS_CACHE_CONFIG="$DEEP_LSS/configs/cls/branch/default.yaml"

# ensure_cls_cache <net config> <log>: build the rebinned-Cls cache if the net (resolved in
# RESOLVED) needs one that is missing. Maps+cls nets always read it, Cls nets for hard_rebinned.
ensure_cls_cache() {
    local resolved=$RESOLVED log=$2 n_bins build_config
    if grep -qE '^network\.cls\.' "$resolved"; then
        n_bins=$(grep -E '^network\.cls\.n_bins=' "$resolved" | cut -d= -f2)
        build_config=$CLS_CACHE_CONFIG
        if ! grep -qE "^cls_n_bins: *${n_bins:-none}( |$)" "$build_config"; then
            echo "Net wants a Cls cache with n_bins=${n_bins:-unset}, which $build_config does not build." >&2
            return 1
        fi
    elif grep -qx 'scale_cut=hard_rebinned' "$resolved"; then
        n_bins=$(grep -E '^cls_n_bins=' "$resolved" | cut -d= -f2)
        build_config=$1
    else
        return 0
    fi

    CLS_CACHE="$INPUT/cls/rebinned_nb${n_bins}_${SCALES}.h5"
    [ -f "$CLS_CACHE" ] && return 0
    echo "Building missing Cls cache $CLS_CACHE"
    # The cache spans all probe pairs, hence combined.yaml; the _nla variants only differ in params.
    $SRUN -N1 -n1 --cpus-per-task=288 --mem=450G --cpu-bind=none --environment=tensorflow \
        --output="$log" \
        python "$DEEP_LSS/deep_lss/apps/run_cls_training+evaluation.py" \
            --msfm_config="$MSFM_CONFIG" \
            --probes_config="${PROBES_DIR}combined.yaml" \
            --scales_config="$SCALES_CONFIG" \
            --loss_config="$LOSS_CONFIG" \
            --net_config="$build_config" \
            --data_config="$DATA_CONFIG" \
            --data_dir="$INPUT" \
            --out_dir="$INPUT" \
            --model_name="precache" \
            --precache_only
}

# resolve_glob <glob>: absolute config paths into CONFIGS; a relative glob is taken from $DEEP_LSS.
resolve_glob() {
    local glob=$1
    case "$glob" in /*) ;; *) glob="$DEEP_LSS/$glob" ;; esac
    shopt -s nullglob
    CONFIGS=($glob)
    shopt -u nullglob
    [ ${#CONFIGS[@]} -gt 0 ] || { echo "No configs matched $1" >&2; exit 1; }
}

# run_inference <runs root> <run>...: msi's inference.sh inside this allocation, one run per GPU.
# The flow settings (N_FLOWS, FLOW_CONFIG, LOAD_FLOW, ...) pass through from the environment.
run_inference() {
    local root=$1
    shift
    RUNS="$*" RUNS_ROOT="$root" RUN_NUM="${RUN_NUM:-1}" DRYRUN="$DRYRUN" \
        bash "$MSI/submissions/clariden/inference.sh"
}
