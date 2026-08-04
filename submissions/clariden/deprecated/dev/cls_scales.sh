#!/bin/bash
# Moved to deprecated/ 2026-08-04: broken -- --mlp_config/configs/mlp/ path no longer exists
# (current: --net_config/configs/cls/mlp/), plus hardcoded stale VERSION=v16/rot_in_place. Do not
# resurrect without porting to the current --net_config convention and v17/baseline.
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_scales
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

# Fixed probe for this sweep (cf. cls_training.sh, which instead loops over probes at a fixed scale).
PROBE="lensing"

# Sweep over these scale-cut configs from configs/scales, one per GPU (8wl,40gc excluded).
SCALES_LIST=("8wl,32gc" "lmax_1024" "unsmoothed")

MLP="default"
LOSS="vmim"
DATA="default"

# All runs land under this grouping dir; each scale is a model subdirectory (model_name = scale name),
# so outputs are runs/$VERSION/$SUBVERSION/cls/$PROBE/debug/scales/v1/<scale>/.
RUNS_DIR="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE/debug/scales/v1"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# For hard_rebinned: pre-compute the per-scale Cls cache with full-node resources before the per-GPU
# training workers start. The cache is scale-dependent (binning uses each config's l_min/l_max) but
# probe-independent (covers all pairs), so build one per scale with the combined probes. The build is
# idempotent — it returns immediately if the cache already exists — so scales whose cache was built by
# an earlier run (e.g. 8wl,32gc, lmax_1024) are skipped cheaply.
SCALE_CUT=$(srun -N1 --ntasks-per-node=1 --environment=tensorflow python -c "
import yaml
with open('$REPOS/y3-deep-lss/configs/mlp/${MLP}.yaml') as f:
    print(yaml.safe_load(f).get('scale_cut', 'soft_pruned'))
")

if [ "$SCALE_CUT" = "hard_rebinned" ]; then
    for SCALES in "${SCALES_LIST[@]}"; do
        LOG_PRECACHE="$RUNS_DIR/$SCALES/logs/${SLURM_JOB_ID}_precache"
        mkdir -p "$(dirname "$LOG_PRECACHE")"
        srun -N1 --ntasks-per-node=1 --exclusive --cpus-per-task=288 --mem=450G \
            --environment=tensorflow \
            --output="${LOG_PRECACHE}.log" \
            python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
                --probes_config="$REPOS/y3-deep-lss/configs/probes/combined.yaml" \
                --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
                --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
                --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${MLP}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --data_dir="$INPUT" \
                --out_dir="$INPUT" \
                --model_name="precache" \
                --precache_only
    done
fi

# Train + infer each scale on its own GPU, in parallel (one background subshell per scale).
for SCALES in "${SCALES_LIST[@]}"; do
    LOG="$RUNS_DIR/$SCALES/logs/${SLURM_JOB_ID}"
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
                --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${MLP}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --data_dir="$INPUT" \
                --out_dir="$RUNS_DIR" \
                --model_name="$SCALES" \
                --include_grid \
                --include_des \
                --include_mocks

        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="${LOG}_inference.log" \
            bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$RUNS_DIR\" \
                --model_name=\"$SCALES\" \
                --flow_config=\"$FLOW_CONFIG\" \
                --n_flows=4 \
                --sample_posterior \
                --include_grid \
                --include_des \
                --include_mocks"
    ) &
done

wait
