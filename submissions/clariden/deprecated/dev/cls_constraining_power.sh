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
#SBATCH --job-name=cls_constraining_power
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Constraining-power sweep for the weak-lensing Cls pipeline.
# Scale cuts are held FIXED at 8wl,32gc (the v33 baseline). Each experiment makes the fixed
# 6-dim summary carry more information (richer VMIM head, finer binning, residual/GELU MLP) or
# denoises the input -- all keep dim_summary_fac=1, since fac=2 (v28_vmim) was overconfident.
# Success = higher Omega_m-S8 FoM AT fixed-or-better SBC/TARP/HPD calibration.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
PROBE="lensing"
SCALES="8wl,32gc"
DATA="default"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# Experiment matrix: parallel arrays (name, mlp_config, loss_config). Four configs == one per
# GPU on a single node, lensing only. ALL use 32-bin input (cls_n_bins=32, set in the MLP
# config), so this is a clean 2x2 over the two remaining levers --
# architecture {default, resgelu=gelu+residual} x head {GMM, flow q(theta|s) fac:1}:
#   v34_base         -- default MLP,  GMM head
#   v34_flowhead     -- default MLP,  flow head        (head effect)
#   v34_resgelu      -- resgelu MLP,  GMM head         (architecture effect)
#   v34_resgelu_flow -- resgelu MLP,  flow head        (both)
EXP_NAMES=( "v34_base"   "v34_flowhead"       "v34_resgelu"        "v34_resgelu_flow"   )
EXP_MLP=(   "finebins32" "finebins32"         "resgelu_finebins32" "resgelu_finebins32" )
EXP_LOSS=(  "vmim_gmm"   "vmim"               "vmim_gmm"           "vmim"               )

# (the historical flow-head instability was unstandardized theta; heads are standardized by
# default since 2026-07-12, and the cls/ loss-config subdir was consolidated into configs/loss/)

# Optional clean-baseline rerun for apples-to-apples FoM comparison:
# EXP_NAMES+=( "v34_baseline" ); EXP_MLP+=( "default" ); EXP_LOSS+=( "cls/vmim" )

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"

# ---------------------------------------------------------------------------
# Precache the shared hard_rebinned Cls cache(s) with full-node resources before the
# per-GPU workers start. The cache is keyed by (cls_n_bins, scales) and is independent of
# the loss/model, so we precache once per distinct MLP config (different cls_n_bins ->
# different cache file). build_rebinned_cls_cache returns immediately if the file exists,
# so listing the same cls_n_bins more than once is cheap.
PRECACHE_MLPS=("finebins32")   # all four configs use 32-bin input -> single nb32 cache
# ---------------------------------------------------------------------------

for PC_MLP in "${PRECACHE_MLPS[@]}"; do
    SCALE_CUT=$(srun -N1 --ntasks-per-node=1 --environment=tensorflow python -c "
import yaml
with open('$REPOS/y3-deep-lss/configs/mlp/${PC_MLP}.yaml') as f:
    print(yaml.safe_load(f).get('scale_cut', 'soft_pruned'))
")
    if [ "$SCALE_CUT" = "hard_rebinned" ]; then
        LOG_PRECACHE="$OUTPUT/precache/logs/${SLURM_JOB_ID}_precache_${PC_MLP}"
        mkdir -p "$(dirname "$LOG_PRECACHE")"
        srun -N1 --ntasks-per-node=1 --exclusive --cpus-per-task=288 --mem=450G \
            --environment=tensorflow \
            --output="${LOG_PRECACHE}.log" \
            python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
                --probes_config="$REPOS/y3-deep-lss/configs/probes/combined.yaml" \
                --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
                --loss_config="$REPOS/y3-deep-lss/configs/loss/vmim_gmm.yaml" \
                --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${PC_MLP}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --data_dir="$INPUT" \
                --out_dir="$INPUT" \
                --model_name="precache" \
                --precache_only
    fi
done

# ---------------------------------------------------------------------------
# Train + infer each experiment. srun --exclusive --gpus-per-task=1 self-throttles to the
# node's available GPUs, so more experiments than GPUs simply queue.
for i in "${!EXP_NAMES[@]}"; do
    MODEL_NAME="${EXP_NAMES[$i]}"
    MLP="${EXP_MLP[$i]}"
    LOSS="${EXP_LOSS[$i]}"
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
    ) &
done

wait
