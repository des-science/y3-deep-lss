#!/bin/bash
# Moved to deprecated/ 2026-08-04: superseded by submissions/clariden/cls/experiments/cls_experiment.sh, the
# generalized form of this launcher (MODEL_NAME/PROBE/LOSS/NET/CLS_CONFIG env vars).
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_ext_vmim
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v16/rot_in_place/cls/lensing/lmax_1024_ext/logs/slurm-%j.out

# Full VMIM constraining-power test (follow-up to the frozen-summary null result): retrain the
# lmax_1024 lensing Cls compression with the mutual-information target extended by the implicitly
# marginalized grid parameters (ns, Ob, H0, bary_Mc, bary_nu; see configs/probes/lensing_ext.yaml).
# The 11-dim summaries + flow then feed the usual inference stage, which auto-runs the
# reference-prior (Gower-Street) DES variants since ns/Ob/H0 are in the params list.
# Readout: FoM(Om,S8) of chain_DESy3_*_refpriors vs the plain chains, compared to the frozen-summary
# ext flow and the lmax_1024 baseline. Everything lands in cls/lensing/lmax_1024_ext/, no existing
# run is touched. The lmax_1024 Cls cache is scale-dependent but probe/params-independent and
# already exists, so no precache step is needed.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

PROBE="lensing_ext"
SCALES="lmax_1024"
MLP="default"
LOSS="vmim_gmm"  # standardized GMM head (vmim_ext.yaml was folded into vmim_gmm.yaml + the standardize_theta default)
DATA="default"
MODEL_NAME="lmax_1024_ext"

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
