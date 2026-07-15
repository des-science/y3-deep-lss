#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_min_fac2_vmim
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/v16/rot_in_place/cls/lensing/lmax_1024_min_fac2/logs/slurm-%j.out

# Decoupling test for the VMIM target-dimensionality penalty: 3-param target (Om, s8, w0) like
# lmax_1024_min, but with dim_summary_fac=2 so the summary stays 6-dim like the baseline. This
# separates the two components that were scaled together in the 3p/6p/11p ladder (dim_summary_fac=1
# ties dim(s) = n_params): the compression target content vs the flow's random-variable dimension.
# Readout against DESy3 FoM2D(Om,S8) wCDM min 452 / baseline 352: FoM ~ min -> the 6-dim summary
# costs the flow ~nothing and the 3p->6p penalty is compression-side (MI target diluting info into
# the IA directions); FoM ~ baseline -> the penalty is the flow's 6-dim density estimation and
# shrinking the target buys nothing on its own. Caveat (from the since-removed configs/loss/cls/vmim_flow_fac1.yaml):
# fac=2 once produced overconfident posteriors with the flow head (v28_vmim) -- check coverage,
# not just FoM. Everything lands in cls/lensing/lmax_1024_min_fac2/, no existing run is touched;
# the lmax_1024 Cls cache is scale-dependent but probe/params-independent and already exists.

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
LOSS="vmim_fac2"
DATA="default"
MODEL_NAME="lmax_1024_min_fac2"

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
