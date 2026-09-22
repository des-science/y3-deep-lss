#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Standalone re-run of the inference tail of ../training.sh against an existing preds_*.h5 (no
# retrain) -- use to recover a run whose inference step failed to launch. Override OUTPUT/MODEL_DIR
# to target a specific run directory. Needs eval too? Use eval_inference.sh instead.
# EXTEND_PARAMS / LOAD_FLOW (below) also make this the entry point for an ALTERNATIVE conditioning
# vector (production's lives in the flow config) and for re-sampling an already-trained flow.
# Submit with --uenv-passthrough=ignore from inside a uenv session.

# --- Runtime environment ---------------------------------------------------------------------

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

MSI="$REPOS/multiprobe-simulation-inference"

# --- Overridable defaults ----------------------------------------------------------------------

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

PROBE="${PROBE:-lensing}"         # run dir under maps/<probe>/; ignored if OUTPUT is set directly
MODEL_DIR="${MODEL_DIR:-t1_cls}"  # the run to re-infer
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
FLOW_MEMBERS="${FLOW_MEMBERS:-}"

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

# --- Derived paths, configs and flags ----------------------------------------------------------

OUTPUT="${OUTPUT:-$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE}"
LOG="$OUTPUT/$MODEL_DIR/logs/${SLURM_JOB_ID}_${RUN_NUM}_${STRATEGY}"
mkdir -p "$(dirname "$LOG")"

# --flow_configs takes a bare list, --flow_config a single "=" argument; unquoted on purpose below
# so the list splits into separate argv entries.
if [ -n "$FLOW_CONFIGS" ]; then
    FLOW_CONFIG_FLAGS="--flow_configs $FLOW_CONFIGS"
else
    FLOW_CONFIG_FLAGS="--flow_config=$FLOW_CONFIG"
fi
LABEL_FLAG=""; [ -n "$FLOW_LABEL" ] && LABEL_FLAG="--flow_label=$FLOW_LABEL"

# --- Stage 1: Inference ------------------------------------------------------------------------

# --cpu-bind=none: otherwise this 1-GPU/72-CPU sub-allocation fails to launch
srun -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G --cpu-bind=none \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $MSI/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL_DIR\" \
        $FLOW_CONFIG_FLAGS \
        $LABEL_FLAG \
        --n_flows=$N_FLOWS \
        $EXTEND_PARAMS \
        $LOAD_FLOW \
        $FLOW_MEMBERS \
        $SAMPLE_POSTERIOR \
        $INCLUDE_OBS"
