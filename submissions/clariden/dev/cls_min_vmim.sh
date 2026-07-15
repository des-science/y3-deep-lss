#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_min_vmim
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v16/rot_in_place/cls/lensing/lmax_1024_min/logs/slurm-%j.out

# Dimensionality control for the full-VMIM ext test: retrain the lmax_1024 lensing Cls compression
# with the mutual-information target REDUCED to (Om, s8, w0) only, implicitly marginalizing the IA
# parameters like the DES Y3 SBI reference papers (whose flows live in this 3D space). 3-dim
# summaries + a 6-dim flow joint (vs 12 baseline / 22 ext). Readout: if the 2d/3d FoM(Om,S8) and
# marginals match the 6-param lmax_1024 baseline, compression and density estimation are healthy at
# 6 params and the ext-run degradation is a pure target/flow-dimensionality cost; a GAIN here would
# mean even the 6-param target already pays a dimensionality penalty vs the references' setup.
# Everything lands in cls/lensing/lmax_1024_min/, no existing run is touched; the lmax_1024 Cls
# cache is scale-dependent but probe/params-independent and already exists.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

PROBE="lensing_min"
SCALES="lmax_1024"
MLP="default"
LOSS="vmim_gmm"  # standardized GMM head; the recorded lmax_1024_min run used the pre-2026-07-12 vmim.yaml (GMM, unstandardized)
DATA="default"
MODEL_NAME="lmax_1024_min"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/lensing"
FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$LOG")"

srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
    --environment=tensorflow \
    --output="${LOG}_training.log" \
    python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
        --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
        --probes_config="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml" \
        --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
        --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
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
