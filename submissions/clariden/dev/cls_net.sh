#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_net
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
PROBE="lensing"
# PROBE="clustering"
# PROBE="combined"

# Learning-rate schedule = subfolder of configs/mlp/. The two folders hold the SAME four
# preprocessing variants and differ only in the schedule; flip this to test the other folder.
# LR_SCHED="lr_plateau"
LR_SCHED="lr_constant"

# Four network/preprocessing configs run in parallel (one GPU each):
#   default     -> asinh_per_feature transform, no PCA
#   log         -> fixed log1p transform, no PCA
#   pca         -> asinh_per_feature + PCA
#   pca_whiten  -> asinh_per_feature + PCA + whitening
NET_CONFIGS=("default" "log" "pca" "pca_whiten")

LOSS="vmim"
SCALES="8wl,32gc"
DATA="default"
BASE_MODEL_NAME="v36_net_${LR_SCHED}"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"

# Pre-compute the shared hard_rebinned Cls cache with full-node resources before the per-GPU
# training workers start. The cache is mlp/loss-config-independent (covers all pairs; probe and
# preprocessing selection happen at load time), and all four net configs share the same
# scale_cut / cls_n_bins, so a single precache serves all of them.
SCALE_CUT=$(srun -N1 --ntasks-per-node=1 --environment=tensorflow python -c "
import yaml
with open('$REPOS/y3-deep-lss/configs/mlp/${LR_SCHED}/${NET_CONFIGS[0]}.yaml') as f:
    print(yaml.safe_load(f).get('scale_cut', 'soft_pruned'))
")

if [ "$SCALE_CUT" = "hard_rebinned" ]; then
    LOG_PRECACHE="$OUTPUT/precache/logs/${SLURM_JOB_ID}_precache"
    mkdir -p "$(dirname "$LOG_PRECACHE")"
    srun -N1 --ntasks-per-node=1 --exclusive --cpus-per-task=288 --mem=450G \
        --environment=tensorflow \
        --output="${LOG_PRECACHE}.log" \
        python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
            --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
            --probes_config="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml" \
            --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
            --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
            --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${LR_SCHED}/${NET_CONFIGS[0]}.yaml" \
            --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
            --data_dir="$INPUT" \
            --out_dir="$INPUT" \
            --model_name="precache" \
            --precache_only
fi

for NET in "${NET_CONFIGS[@]}"; do
    MODEL_NAME="${BASE_MODEL_NAME}_${NET}"
    LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$LOG")"

    (
        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --environment=tensorflow \
            --output="${LOG}_training.log" \
            python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
                --probes_config="$REPOS/y3-deep-lss/configs/probes/${PROBE}.yaml" \
                --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
                --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
                --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${LR_SCHED}/${NET}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --data_dir="$INPUT" \
                --out_dir="$OUTPUT" \
                --model_name="$MODEL_NAME" \
                --include_grid

        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="${LOG}_inference.log" \
            bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$MODEL_NAME\" \
                --flow_config=\"$FLOW_CONFIG\" \
                --sample_posterior \
                --n_flows=4 \
                --include_grid"
    ) &
done

wait
