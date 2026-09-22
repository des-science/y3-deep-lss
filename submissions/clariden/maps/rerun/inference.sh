#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Standalone re-run of the inference tail of ../training.sh against an existing preds_*.h5 (no
# retrain) -- to recover a run whose inference step failed to launch, or to re-infer finished runs
# after a change to the flow config. Needs eval too? Use eval_inference.sh instead.
#
# RUNS (below) fans several run dirs out over the node's 4 GPUs, one run per GPU, the same way
# cls/cls_training.sh fans out its probes -- inference is a single-GPU pytorch job, so one run per
# node would leave three GPUs idle. Nothing here is maps-specific (run_inference.py reads the run's
# own configs.yaml), so a `cls/<probe>/v1` entry is as valid as a `maps_gcnn/<probe>/v1` one.
#
# EXTEND_PARAMS / LOAD_FLOW (below) also make this the entry point for an ALTERNATIVE conditioning
# vector (production's lives in the flow config) and for re-sampling an already-trained flow.
# From inside a uenv session, submit as: env -u LD_LIBRARY_PATH sbatch --uenv-passthrough=ignore ...

# --- Runtime environment ---------------------------------------------------------------------

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

# Each run is its own 1-GPU/72-CPU srun step, so the per-step CPU count has to be stated here too.
# Left at the node's 288 (what --exclusive gives the batch step), every step asks to bind 288 CPUs
# inside its own 72-CPU allocation and dies instantly with "CPU binding outside of job step
# allocation" -- which is what cls/cls_training.sh's identical export is preventing.
export SLURM_CPUS_PER_TASK=72

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults ----------------------------------------------------------------------

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

# Run dirs to infer, space-separated, each "<representation>/<probe>/<run>" relative to RUNS_ROOT.
# Up to GPUS_PER_NODE run concurrently; a longer list is processed in waves, so raise --time for it.
#   RUNS="maps_gcnn/lensing/v1 maps_gcnn/clustering/v1 maps_gcnn/combined/v1 cls/lensing/v1" \
#       VERSION=v18 SUBVERSION=default env -u LD_LIBRARY_PATH sbatch inference.sh
# Empty keeps the original single-run behaviour: the PROBE/MODEL_DIR pair below, or OUTPUT directly.
RUNS="${RUNS:-}"
RUNS_ROOT="${RUNS_ROOT:-$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION}"

PROBE="${PROBE:-lensing}"         # run dir under maps/<probe>/; ignored if RUNS or OUTPUT is set
MODEL_DIR="${MODEL_DIR:-t1_cls}"  # the run to re-infer; ignored if RUNS is set
RUN_NUM="${RUN_NUM:-1}"           # names the log only; there is no chain here

# Extended conditioning vector. LEAVE THIS EMPTY for production: configs/flow/maf.yaml already sets
# extend_params: [ns, Ob, H0], so the flow conditions on the weakly constrained nuisance parameters
# by default and saves under the unprefixed <flow>_<steps>/ directory. See that config's header.
#
# Setting it here OVERRIDES the config and marks the run as an experiment: everything then saves
# under ext_<flow>_<steps>/ instead, so the production flow is untouched. Use it to try a different
# vector, or to train the unextended baseline for comparison:
#   EXTEND_PARAMS="--extend_params ns Ob H0 bary_Mc bary_nu" sbatch inference.sh
# There is no CLI way to ask for NO extension; point FLOW_CONFIG at a config with extend_params: [].
EXTEND_PARAMS="${EXTEND_PARAMS:-}"

# Rerun only the sampling stages against an already-trained flow (e.g. after a plotting fix):
#   LOAD_FLOW=--load_flow sbatch inference.sh
LOAD_FLOW="${LOAD_FLOW:-}"

# Per-member DES chains for the flow-ensemble convergence test (chain_DESy3_flow_{m}.npy, next to
# the ensemble chain, which is left alone). Needs --include_des and an ensemble flow.
#
# ON BY DEFAULT, matching eval_inference.sh: this is the blinding test, so every production run has
# to carry it, and a run that silently lacks it is only noticed when the figure is drawn. run_inference
# samples the unrestricted wCDM model for --flow_member_obs, which defaults to DESy3 alone -- the
# systematics variants would each multiply the cost by N_FLOWS and answer a different question.
# The ${VAR-default} form (one dash) means an explicitly EMPTY value switches the stage off:
#   FLOW_MEMBERS= sbatch inference.sh
FLOW_MEMBERS="${FLOW_MEMBERS---sample_flow_members}"

# Density estimator. FLOW_CONFIG names ONE architecture, replicated into N_FLOWS seed clones;
# FLOW_CONFIGS instead lists several and builds a HETEROGENEOUS ensemble of one member per
# (config, replica), so members disagree by architecture and not only by initialization. The two
# are mutually exclusive in run_inference.py. FLOW_LABEL prefixes the checkpoint directory
# (<label>_ensemble_flow_<steps>/), which is what keeps an experiment off the production flow:
#   FLOW_CONFIGS="$MSI/configs/flow/maf.yaml $MSI/configs/flow/sigmoid.yaml" \
#       N_FLOWS=4 FLOW_LABEL=hetero sbatch inference.sh
# A lipschitz member cannot be reloaded from its checkpoint, so do not mix one in if the flow is
# ever going to be re-sampled with LOAD_FLOW.
FLOW_CONFIG="${FLOW_CONFIG:-$MSI/configs/flow/maf.yaml}"
FLOW_CONFIGS="${FLOW_CONFIGS:-}"
FLOW_LABEL="${FLOW_LABEL:-}"
# Changing N_FLOWS rewrites the flow in place: the checkpoint dir is ensemble_flow_<n_steps>,
# which records the training steps but not the member count.
N_FLOWS="${N_FLOWS:-8}"

# Which stages the sampling tail runs. The defaults reproduce the training-tail behaviour; both use
# ${VAR-default}, so an explicitly EMPTY value switches a stage off. A targeted re-run that adds
# only the per-member chains to an existing flow, leaving mcmc_samples.h5 and the mock/grid chains
# untouched:
#   LOAD_FLOW=--load_flow FLOW_MEMBERS=--sample_flow_members SAMPLE_POSTERIOR= \
#       INCLUDE_OBS=--include_des sbatch inference.sh
SAMPLE_POSTERIOR="${SAMPLE_POSTERIOR---sample_posterior}"
INCLUDE_OBS="${INCLUDE_OBS---include_grid --include_des --include_mocks}"

# --- Fixed settings ----------------------------------------------------------------------------

STRATEGY="mirrored"  # names the logs only -- inference itself is single-GPU pytorch

# How many runs share a node. Lower it for memory-heavy runs and submit more jobs instead -- there
# is no prize for filling a node, and STEP_MEM below is a direct function of this number.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# Per-step memory. BUDGET THIS AGAINST HOST DRAM, WHICH IS 476 GB -- NOT the 870 GB SLURM reports.
# On GH200 each GPU's 95 GB of HBM is exposed as a CPU-less NUMA node, so `free` and RealMemory add
# 4 x 95 GB of GPU memory to the 4 x 119 GB of Grace DRAM and report 856 / 870 GB. --mem is accounted
# against that inflated figure, so SLURM will cheerfully admit a job that cannot fit in DRAM and let
# the kernel OOM-killer sort it out at node level -- a far worse failure than a clean cgroup kill.
# Real budget at 4-way packing: 476 / 4 = 119 GB per step.
#
# The coverage stage holds the whole (n_obs x n_samples x n_params) chain: 1000 mocks x 1024000
# samples is 41 GB at ten parameters but 53 GB under the extended ns/Ob/H0 vector, and it holds a
# host copy alongside the device one. Measured peaks: lensing/clustering ~62 GB, so four fit
# comfortably; `combined` exceeds 110 GB and was OOM-killed there at the old limit, AFTER sampling
# finished, so its chains were written and only mcmc_samples.h5 and the 2_* plots were missing.
# SO: combined packs at most 3 per node:  GPUS_PER_NODE=3 STEP_MEM=150G, three entries in RUNS.
STEP_MEM="${STEP_MEM:-110G}"

# --- Derived paths, configs and flags ----------------------------------------------------------

OUTPUT="${OUTPUT:-$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE}"

# One "<parent dir>|<run dir name>" pair per run to infer, since run_inference.py takes the two
# separately (--out_dir / --model_name).
PAIRS=()
if [ -n "$RUNS" ]; then
    for ENTRY in $RUNS; do
        PAIRS+=("$RUNS_ROOT/$(dirname "$ENTRY")|$(basename "$ENTRY")")
    done
else
    PAIRS=("$OUTPUT|$MODEL_DIR")
fi

# --flow_configs takes a bare list, --flow_config a single "=" argument; unquoted on purpose below
# so the list splits into separate argv entries.
if [ -n "$FLOW_CONFIGS" ]; then
    FLOW_CONFIG_FLAGS="--flow_configs $FLOW_CONFIGS"
else
    FLOW_CONFIG_FLAGS="--flow_config=$FLOW_CONFIG"
fi
LABEL_FLAG=""; [ -n "$FLOW_LABEL" ] && LABEL_FLAG="--flow_label=$FLOW_LABEL"

# --- Stage 1: Inference, one run per GPU -------------------------------------------------------

# Step flags copied from cls/cls_training.sh's inference step, which is the same run_inference.py
# under the same uenv and is what proves 4 of these coexist on one node: --exclusive is what makes
# SLURM hand each step its own GPU and CPU set instead of overlaying them all on the first, and
# --cpu-bind=none is the sub-allocation rule this script has always needed.
infer_one() {
    local out_dir="$1" model="$2"
    local log="$out_dir/$model/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
    mkdir -p "$(dirname "$log")"
    echo "[$(date +%T)] inference: $out_dir/$model -> ${log}_inference.log"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem="$STEP_MEM" \
        --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${log}_inference.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_inference.py \
            --out_dir=\"$out_dir\" \
            --model_name=\"$model\" \
            $FLOW_CONFIG_FLAGS \
            $LABEL_FLAG \
            --n_flows=$N_FLOWS \
            $EXTEND_PARAMS \
            $LOAD_FLOW \
            $FLOW_MEMBERS \
            $SAMPLE_POSTERIOR \
            $INCLUDE_OBS"

    local status=$?
    if [ "$status" -ne 0 ]; then
        echo "FAILED (exit $status): $out_dir/$model — see ${log}_inference.log" >&2
    fi
    return $status
}

# A bare `wait` returns only the LAST background job's status, so a job with one failed run inside
# it used to exit 0 and be recorded COMPLETED -- which is how five OOM-killed runs were first
# reported as a clean sweep. Wait on each pid and propagate.
rc=0
pids=()
flush() {
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    pids=()
}

launched=0
for PAIR in "${PAIRS[@]}"; do
    infer_one "${PAIR%%|*}" "${PAIR##*|}" &
    pids+=($!)
    launched=$((launched + 1))
    # a list longer than the node's GPU count is processed in waves rather than oversubscribed
    [ $((launched % GPUS_PER_NODE)) -eq 0 ] && flush
done
flush

[ "$rc" -ne 0 ] && echo "One or more runs FAILED -- see the FAILED lines above." >&2
exit "$rc"
