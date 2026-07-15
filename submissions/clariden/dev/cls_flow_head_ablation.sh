#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G

# Flow-head architecture ablation (2026-07-12): pick the SMALLEST RealNVP head that matches the
# best MI bound before blessing it as the vmim.yaml default. Grid: num_layers {4,6} x permute
# {on,off} x num_hidden_units {64,128}; reference cell = lmax_1024_flow_std (6 layers, 128 units,
# no permutation, best head vali NLL -2.2923). All runs use the new standardize_theta=true
# DEFAULT (key omitted in the configs on purpose, so this also end-to-end tests the default flip).
# Parameterized via env vars; submit like:
#   sbatch --job-name=<name> --output=<logdir>/slurm-%j.out \
#          --export=ALL,LOSS_CONFIG_PATH=<abs .yaml>,MODEL_NAME=<name> cls_flow_head_ablation.sh
# PROBE is optionally overridable (default lensing) -- used 2026-07-12 to repeat the target-
# dimensionality experiment (lensing_min 3p / lensing 6p / lensing_ext 11p) with the flow head.
# Everything lands in cls/lensing/$MODEL_NAME/, no existing run is touched; the lmax_1024 Cls
# cache is scale-dependent but probe/params-independent and already exists.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

PROBE="${PROBE:-lensing}"
SCALES="lmax_1024"
MLP="default"
DATA="default"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/lensing"
FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

: "${LOSS_CONFIG_PATH:?set LOSS_CONFIG_PATH via --export}"
: "${MODEL_NAME:?set MODEL_NAME via --export}"

LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$LOG")"

srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --environment=tensorflow \
    --output="${LOG}_training.log" \
    python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
        --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --probes_config="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml" \
        --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
        --loss_config="$LOSS_CONFIG_PATH" \
        --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${MLP}.yaml" \
        --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
        --data_dir="$INPUT" \
        --out_dir="$OUTPUT" \
        --model_name="$MODEL_NAME" \
        --include_grid \
        --include_des \
        --include_mocks

srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --uenv=pytorch/v2.9.1:v2 --view=default \
    --output="${LOG}_inference.log" \
    bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
        --out_dir=\"$OUTPUT\" \
        --model_name=\"$MODEL_NAME\" \
        --flow_config=\"$FLOW_CONFIG\" \
        --n_flows=4 \
        --sample_posterior \
        --include_grid \
        --include_des \
        --include_mocks"
