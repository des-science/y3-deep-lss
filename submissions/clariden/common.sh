# Sourced by every script in this tree: roots, production data defaults, shared stages.
# Source it by absolute path; sbatch runs a spool copy of the calling script.

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"
STORE="/capstor/store/cscs/swissai/a0158/athomsen"
DEEP_LSS="$REPOS/y3-deep-lss"
MSFM="$REPOS/multiprobe-simulation-forward-model"
MSI="$REPOS/multiprobe-simulation-inference"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_API_KEY=$(awk '/password/ {print $2}' ~/.netrc 2>/dev/null)

# --- Production data defaults ------------------------------------------------------------------

# v18 and v16 are extended-NLA (plain configs/probes/*.yaml); v17 needs the *_nla probe configs.
VERSION="${VERSION:-v18}"
SUBVERSION="${SUBVERSION:-default}"
SCALES="${SCALES:-8wl,32gc}"  # configs/scales/
LOSS="${LOSS:-vmim}"          # configs/loss/
DATA="${DATA:-default}"       # configs/data/

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OBS_INPUT="${OBS_INPUT:-$STORE/deep_lss/data/$VERSION/$SUBVERSION}"  # mock observations live on the store
RUNS_ROOT="${RUNS_ROOT:-$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION}"
TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

MSFM_CONFIG="$MSFM/configs/$VERSION/$SUBVERSION.yaml"
SCALES_CONFIG="$DEEP_LSS/configs/scales/$SCALES.yaml"
LOSS_CONFIG="$DEEP_LSS/configs/loss/$LOSS.yaml"
DATA_CONFIG="$DEEP_LSS/configs/data/$DATA.yaml"

# DRYRUN=1 prints every srun/sbatch instead of running it (works on the login node).
DRYRUN="${DRYRUN:-0}"
SRUN="srun"
[ "$DRYRUN" = "1" ] && SRUN="echo srun"
SLURM_JOB_ID="${SLURM_JOB_ID:-dryrun}"

# --- Helpers -----------------------------------------------------------------------------------

# A dry run must not create run directories.
make_log_dir() { [ "$DRYRUN" = "1" ] || mkdir -p "$1"; }

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# Pause between two stages of one job, so the previous step has released its GPUs.
settle() { [ "$DRYRUN" = "1" ] || sleep 30; }

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
            --probes_config="$DEEP_LSS/configs/probes/combined.yaml" \
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
