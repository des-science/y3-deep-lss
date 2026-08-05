#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_experiment
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# One-off Cls-summary train+infer experiment on a single GPU/node -- the generic form of what used
# to be several near-identical scripts per (probe, loss, net) combination. Everything lands in
# cls/$PROBE/$MODEL_NAME/, no existing run is touched.
#
# Usage:
#   MODEL_NAME=<name> [PROBE=lensing] [LOSS=vmim] [NET=mlp] [CLS_CONFIG=default] \
#       sbatch --job-name=<name> cls_experiment.sh
#   # or point LOSS_CONFIG / NET_CONFIG at an ad hoc .yaml not yet under configs/{loss,cls/<net>}/
#
# For a multi-config sweep (N of these across the node's 4 GPUs), copy the "( ... ) &" / "wait"
# loop from ../cls_training.sh rather than submitting N of these jobs.

# --- Runtime environment ---------------------------------------------------------------------

# The work runs as a single 1-GPU/72-CPU srun step, so size the thread pools for that, not the node.
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

: "${MODEL_NAME:?set MODEL_NAME, e.g. MODEL_NAME=my_experiment sbatch cls_experiment.sh}"

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"

PROBE="${PROBE:-lensing}"       # configs/probes/<PROBE>.yaml; v17 data wants the _nla variants
LOSS="${LOSS:-vmim}"            # configs/loss/<LOSS>.yaml
NET="${NET:-mlp}"               # mlp | cnn | transformer
CLS_CONFIG="${CLS_CONFIG:-default}"
SCALES="${SCALES:-8wl,32gc}"    # configs/scales/<SCALES>.yaml
DATA="${DATA:-default}"         # configs/data/<DATA>.yaml

# Set these directly to use an ad hoc yaml that has no home under configs/ yet.
LOSS_CONFIG="${LOSS_CONFIG:-$DEEP_LSS/configs/loss/${LOSS}.yaml}"
NET_CONFIG="${NET_CONFIG:-$DEEP_LSS/configs/cls/${NET}/${CLS_CONFIG}.yaml}"

# --- Derived paths, configs and flags ----------------------------------------------------------

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$LOG")"

MSFM_CONFIG="$MSFM/configs/$VERSION/$SUBVERSION.yaml"
PROBES_CONFIG="$DEEP_LSS/configs/probes/${PROBE}.yaml"
SCALES_CONFIG="$DEEP_LSS/configs/scales/${SCALES}.yaml"
DATA_CONFIG="$DEEP_LSS/configs/data/${DATA}.yaml"
FLOW_CONFIG="$MSI/configs/flow/maf.yaml"

# --- Stage 1: Training + evaluation ------------------------------------------------------------

srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --environment=tensorflow \
    --output="${LOG}_training.log" \
    python "$DEEP_LSS/deep_lss/apps/run_cls_training+evaluation.py" \
        --msfm_config="$MSFM_CONFIG" \
        --probes_config="$PROBES_CONFIG" \
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

# --- Stage 2: Inference ------------------------------------------------------------------------

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
