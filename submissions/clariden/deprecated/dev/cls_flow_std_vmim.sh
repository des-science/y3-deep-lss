#!/bin/bash
# Moved to deprecated/ 2026-08-04: superseded by submissions/clariden/cls_experiment.sh, the
# generalized form of this launcher (MODEL_NAME/PROBE/LOSS/NET/CLS_CONFIG env vars).
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_flow_std_vmim
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v16/rot_in_place/cls/lensing/lmax_1024_flow_std/logs/slurm-%j.out

# Flow-head counterpart of cls_std_vmim.sh: 6-param lmax_1024 baseline with the RealNVP
# variational head (now the configs/loss/vmim.yaml default; recorded run: fac=1, 6 coupling layers, 2x128, theta
# standardization -- newly implemented for the flow head in nets/estimators/normalizing_flow.py).
# Tests whether the historical flow-head instability was the unstandardized theta (raw theta feeds
# the coupling MLPs directly). The head vali NLL is directly comparable to the GMM lmax_1024_std
# run (both physical-unit NLLs over the same target) = a bound-tightness measurement.
# Everything lands in cls/lensing/lmax_1024_flow_std/, no existing run is touched.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

PROBE="lensing"
SCALES="lmax_1024"
MLP="default"
LOSS="vmim"  # vmim_flow_std.yaml became the vmim.yaml default (flow head + standardization);
             # NOTE vmim.yaml now also sets permute: true -- the recorded lmax_1024_flow_std run
             # (job 2737883) trained WITHOUT permutation (equivalent downstream, see ablation)
DATA="default"
MODEL_NAME="lmax_1024_flow_std"

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
