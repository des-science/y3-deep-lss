#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_bench_m1
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# bench_m1 optimizer/architecture sweep for the weak-lensing Cls pipeline, at the lmax_1024 scale cut.
# Runs compression (run_cls_training+evaluation.py) -> inference (run_inference.py) for every MLP
# config in configs/mlp/bench_m1, stored under cls/lensing/m1_<config>. All configs share
# cls_n_bins=16 + scale_cut=hard_rebinned, so a single rebinned-Cls precache (nb16 x lmax_1024)
# covers them all.
#
# The inference step auto-produces the DES posterior variants including the NEW combined
# w0 > -1 & NLA (bta = 0) chain (chain_DESy3_w0gt-1_nla.npy) -- emitted for any probe with bta in
# its params (lensing does), so no extra flag is needed here. See msi/utils/observations.py.
#
# 7 configs on 4 GPUs -> two waves (srun --exclusive --gpus-per-task=1 self-throttles to the node's
# GPUs, so the extra experiments simply queue). --time is sized for two waves of train+infer.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
PROBE="lensing"
SCALES="lmax_1024"
LOSS="cls/vmim"
DATA="default"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# Experiment matrix: one bench_m1 MLP config per entry, stored with the m1_ prefix. The loss and
# scale cut are held fixed; each config varies the optimizer/schedule/architecture (see the header
# comment in each yaml). MLP names are subdir-qualified (bench_m1/<name>).
EXP_NAMES=( "m1_default" "m1_cosine" "m1_deep" "m1_ema" "m1_pca" "m1_plateau" "m1_reg" )
EXP_MLP=(   "bench_m1/default" "bench_m1/cosine" "bench_m1/deep" "bench_m1/ema" \
            "bench_m1/pca" "bench_m1/plateau" "bench_m1/reg" )

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"

# ---------------------------------------------------------------------------
# Precache the shared hard_rebinned Cls cache once. The cache is keyed by (cls_n_bins, scales) and is
# independent of loss/model/probe; all bench_m1 configs use cls_n_bins=16, so one precache (using
# bench_m1/default to read cls_n_bins + scale_cut) suffices. build_rebinned_cls_cache returns
# immediately if the file already exists.
# ---------------------------------------------------------------------------
PRECACHE_MLPS=("bench_m1/default")

for PC_MLP in "${PRECACHE_MLPS[@]}"; do
    SCALE_CUT=$(srun -N1 --ntasks-per-node=1 --environment=tensorflow python -c "
import yaml
with open('$REPOS/y3-deep-lss/configs/mlp/${PC_MLP}.yaml') as f:
    print(yaml.safe_load(f).get('scale_cut', 'soft_pruned'))
")
    if [ "$SCALE_CUT" = "hard_rebinned" ]; then
        LOG_PRECACHE="$OUTPUT/precache/logs/${SLURM_JOB_ID}_precache_$(basename "$PC_MLP")"
        mkdir -p "$(dirname "$LOG_PRECACHE")"
        srun -N1 --ntasks-per-node=1 --exclusive --cpus-per-task=288 --mem=450G \
            --environment=tensorflow \
            --output="${LOG_PRECACHE}.log" \
            python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
                --probes_config="$REPOS/y3-deep-lss/configs/probes/combined.yaml" \
                --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
                --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
                --mlp_config="$REPOS/y3-deep-lss/configs/mlp/${PC_MLP}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --data_dir="$INPUT" \
                --out_dir="$INPUT" \
                --model_name="precache" \
                --precache_only
    fi
done

# ---------------------------------------------------------------------------
# Train + infer each experiment. srun --exclusive --gpus-per-task=1 self-throttles to the node's
# available GPUs, so more experiments than GPUs simply queue.
# ---------------------------------------------------------------------------
for i in "${!EXP_NAMES[@]}"; do
    MODEL_NAME="${EXP_NAMES[$i]}"
    MLP="${EXP_MLP[$i]}"
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
