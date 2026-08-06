#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Cls-summary train+eval+infer, one probe per GPU in parallel. Unlike maps/, there is no
# eval-only/infer-only recovery counterpart (no cls/rerun/) -- a failed stage means rerunning the
# whole script. One-off Cls ablations go through experiments/cls_experiment.sh.

# --- Runtime environment ---------------------------------------------------------------------

# Each probe runs as its own 1-GPU/72-CPU srun step, so the thread pools are sized per step rather
# than for the whole 288-core node.
export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

DEEP_LSS="$REPOS/y3-deep-lss"
MSFM="$REPOS/multiprobe-simulation-forward-model"
MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults ----------------------------------------------------------------------

VERSION="${VERSION:-v18}"
SUBVERSION="${SUBVERSION:-default}"

# Space-separated; the 4 defaults fill the node's 4 GPUs. Each entry is "probe" or
# "probe:probes_config" -- the probe keys the run dir, the config selects configs/probes/*.yaml
# (defaulting to the probe name). The defaults are the plain (bta-carrying) configs, which is what
# extended-NLA data wants: v18/default and v16. v17 is standard-NLA, so there it is
#   PROBES="lensing:lensing_nla 2x2pt:2x2pt_nla combined:combined_nla clustering"
read -r -a PROBES <<< "${PROBES:-lensing clustering 2x2pt combined}"

# Summary architecture: configs/cls/<NET>/<CLS_CONFIG>.yaml sets network.name (mlp | cls_cnn |
# cls_transformer) -- see run_cls_training+evaluation.py for the switch.
NET="${NET:-mlp}"                 # mlp | cnn | transformer
CLS_CONFIG="${CLS_CONFIG:-default}"

LOSS="${LOSS:-vmim}"              # configs/loss/: also vmim_vicreg, vmim_vicreg_inv, vmim_vicreg_inv_10
SCALES="${SCALES:-8wl,32gc}"      # configs/scales/: also 8wl,40gc, unsmoothed, lmax_1024
DATA="${DATA:-default}"           # configs/data/<DATA>.yaml
MODEL_NAME="${MODEL_NAME:-v1}"    # run dir under cls/<probe>/

# 1 builds the rebinned-Cls cache for (cls_n_bins, SCALES) and exits before any training. Use this
# on a dataset whose cache does not exist yet: the build reads the full raw grid Cls into memory
# (~410 GiB for 2500x400x1536x36 float32, twice over -- see build_rebinned_cls_cache) and takes
# tens of minutes, which does not fit inside this script's 1 h budget alongside four trainings.
# Every later job then no-ops on the cache in ~40 s. Give it more wall clock than the default:
#   VERSION=v18 SUBVERSION=default PRECACHE_ONLY=1 sbatch --time=02:00:00 cls/cls_training.sh
PRECACHE_ONLY="${PRECACHE_ONLY:-0}"

# --- Derived paths, configs and flags ----------------------------------------------------------

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
RUNS="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls"

MSFM_CONFIG="$MSFM/configs/$VERSION/$SUBVERSION.yaml"
SCALES_CONFIG="$DEEP_LSS/configs/scales/${SCALES}.yaml"
LOSS_CONFIG="$DEEP_LSS/configs/loss/${LOSS}.yaml"
DATA_CONFIG="$DEEP_LSS/configs/data/${DATA}.yaml"
NET_CONFIG="$DEEP_LSS/configs/cls/${NET}/${CLS_CONFIG}.yaml"
FLOW_CONFIG="$MSI/configs/flow/maf.yaml"

# Aborts the per-probe subshell instead of letting inference silently run against a stale
# preds_*.h5 if the training+eval stage fails. Inherited by the "( ... ) &" subshells below.
check_stage() {
    local status=$1 stage=$2 log=$3
    if [ "$status" -ne 0 ]; then
        echo "$stage failed (exit $status) — see $log. Aborting before the next stage." >&2
        exit "$status"
    fi
}

# --- Stage 0: Cls precache (hard_rebinned scale cut only) --------------------------------------

# The rebinned-Cls cache is probe-independent (it covers all probe pairs), so it is built once up
# front rather than raced for by the parallel probes below. Only hard_rebinned reads it.
SCALE_CUT=$(srun -N1 --ntasks-per-node=1 --environment=tensorflow python -c "
import yaml
with open('$NET_CONFIG') as f:
    print(yaml.safe_load(f).get('scale_cut', 'soft_pruned'))
")

if [ "$SCALE_CUT" = "hard_rebinned" ]; then
    # combined.yaml, not combined_nla.yaml: the two differ only in their `params` list, which does
    # not reach the cache -- the probe/channel structure that does is identical.
    # A PRECACHE_ONLY job has no run to speak of, so its log goes next to the cache instead of
    # creating an empty run dir under cls/lensing/.
    if [ "$PRECACHE_ONLY" = "1" ]; then
        LOG_PRECACHE="$INPUT/precache/logs/${SLURM_JOB_ID}_precache"
    else
        LOG_PRECACHE="$RUNS/lensing/$MODEL_NAME/logs/${SLURM_JOB_ID}_precache"
    fi
    mkdir -p "$(dirname "$LOG_PRECACHE")"
    srun -N1 --ntasks-per-node=1 --exclusive --cpus-per-task=288 --mem=450G \
        --environment=tensorflow \
        --output="${LOG_PRECACHE}.log" \
        python "$DEEP_LSS/deep_lss/apps/run_cls_training+evaluation.py" \
            --msfm_config="$MSFM_CONFIG" \
            --probes_config="$DEEP_LSS/configs/probes/combined.yaml" \
            --scales_config="$SCALES_CONFIG" \
            --loss_config="$LOSS_CONFIG" \
            --net_config="$NET_CONFIG" \
            --data_config="$DATA_CONFIG" \
            --data_dir="$INPUT" \
            --out_dir="$INPUT" \
            --model_name="precache" \
            --precache_only
    check_stage $? "Precache" "${LOG_PRECACHE}.log"
elif [ "$PRECACHE_ONLY" = "1" ]; then
    echo "PRECACHE_ONLY=1 but $NET_CONFIG has scale_cut=$SCALE_CUT, not hard_rebinned — no cache to"\
         "build. Nothing was done." >&2
    exit 1
fi

if [ "$PRECACHE_ONLY" = "1" ]; then
    echo "PRECACHE_ONLY=1: cache for (nb$(grep -E '^\s*cls_n_bins:' "$NET_CONFIG" | grep -oE '[0-9]+'), $SCALES) is in place — skipping training."
    exit 0
fi

# --- Stage 1: Per-probe train+eval, then inference (one probe per GPU, in parallel) ------------

for ENTRY in "${PROBES[@]}"; do
    PROBE="${ENTRY%%:*}"
    PROBE_CONFIG="${ENTRY##*:}"

    OUTPUT="$RUNS/$PROBE"
    LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$LOG")"

    (
        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --environment=tensorflow \
            --output="${LOG}_training.log" \
            python "$DEEP_LSS/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$MSFM_CONFIG" \
                --probes_config="$DEEP_LSS/configs/probes/${PROBE_CONFIG}.yaml" \
                --scales_config="$SCALES_CONFIG" \
                --loss_config="$LOSS_CONFIG" \
                --net_config="$NET_CONFIG" \
                --data_config="$DATA_CONFIG" \
                --data_dir="$INPUT" \
                --out_dir="$OUTPUT" \
                --model_name="$MODEL_NAME" \
                --include_grid \
                --include_des \
                --include_mocks
        check_stage $? "Training+evaluation ($PROBE)" "${LOG}_training.log"

        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="${LOG}_inference.log" \
            bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$MODEL_NAME\" \
                --flow_config=\"$FLOW_CONFIG\" \
                --n_flows=4 \
                --sample_posterior \
                --include_grid \
                --include_des \
                --include_mocks"
    ) &
done

wait
