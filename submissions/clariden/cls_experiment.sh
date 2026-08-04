#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_experiment
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# One-off Cls-summary train+infer experiment on a single GPU/node -- generic form of what used
# to be several near-identical scripts under dev/ (cls_min_vmim.sh, cls_std_vmim.sh, etc.), one
# per (probe, loss, net) combination. Everything lands in cls/$PROBE/$MODEL_NAME/, no existing
# run is touched.
#
# Usage:
#   MODEL_NAME=<name> [PROBE=lensing] [LOSS=vmim] [NET=mlp] [CLS_CONFIG=default] \
#       sbatch --job-name=<name> cls_experiment.sh
#   # or point LOSS_CONFIG / NET_CONFIG at an ad hoc .yaml not yet under configs/{loss,cls/<net>}/
#
# For a multi-config sweep (N of these on N GPUs of one node), loop the srun calls yourself
# with "(...) &" / "wait" around this pattern -- see dev/ for examples of that shape.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="${VERSION:-v17}"
SUBVERSION="${SUBVERSION:-baseline}"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

PROBE="${PROBE:-lensing}"
LOSS="${LOSS:-vmim}"
NET="${NET:-mlp}"
CLS_CONFIG="${CLS_CONFIG:-default}"
SCALES="${SCALES:-8wl,32gc}"
DATA="${DATA:-default}"

LOSS_CONFIG="${LOSS_CONFIG:-$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml}"
NET_CONFIG="${NET_CONFIG:-$REPOS/y3-deep-lss/configs/cls/${NET}/${CLS_CONFIG}.yaml}"

: "${MODEL_NAME:?set MODEL_NAME, e.g. MODEL_NAME=my_experiment sbatch cls_experiment.sh}"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
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
        --loss_config="$LOSS_CONFIG" \
        --net_config="$NET_CONFIG" \
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
