#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_inference_probes
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
PROBES=("lensing" "clustering" "2x2pt" "combined")

MODEL_NAME="v26"

CONFIG_DIR="$REPOS/multiprobe-simulation-inference/configs/flow"

# Sweep entries are "config_basename:flow_label:n_flows". The flow_label keeps each entry's
# checkpoints/chains/plots/logs in a separate subdirectory so single- and ensemble-flow runs of the
# same config don't collide. n_flows=1 is a single LikelihoodFlow; n_flows>1 trains a
# LikelihoodFlowEnsemble (log-mean-exp over independently-seeded members -> broader, more conservative
# posteriors). Configs are run one group at a time (4 probes in parallel, one GPU each); keeping the
# node otherwise idle per group avoids contention that would distort the [timing] log lines.
#
# This sweep compares single vs 3-member ensemble at two capacities (both with weight decay enabled in
# the spline configs). Edit freely.
FLOW_CONFIGS=(
    "spline:spline:1"
    "spline:spline_ens3:3"
    "spline_large:spline_large:1"
    "spline_large:spline_large_ens3:3"
)

echo "Starting spline architecture sweep over ${#FLOW_CONFIGS[@]} configs x ${#PROBES[@]} probes..."

for ENTRY in "${FLOW_CONFIGS[@]}"; do
    IFS=':' read -r CONFIG_NAME FLOW_LABEL N_FLOWS <<< "$ENTRY"
    FLOW_CONFIG="$CONFIG_DIR/${CONFIG_NAME}.yaml"

    echo "=============================================================="
    echo "[$(date '+%F %T')] config=$CONFIG_NAME label=$FLOW_LABEL n_flows=$N_FLOWS"
    GROUP_START=$SECONDS

    for PROBE in "${PROBES[@]}"; do
        OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
        LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}_${FLOW_LABEL}"

        # --sample_posterior writes mcmc_samples.h5 over the held-out mocks (TARP/coverage input);
        # works for both a single flow and the ensemble via the default --mcmc_backend=torch_batched
        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="${LOG}_inference_probes.log" \
            bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$MODEL_NAME\" \
                --flow_config=\"$FLOW_CONFIG\" \
                --flow_label=\"$FLOW_LABEL\" \
                --n_flows=\"$N_FLOWS\" \
                --sample_posterior \
                --include_grid \
                --include_mocks" &
    done
    wait
    echo "[$(date '+%F %T')] config=$CONFIG_NAME done in $((SECONDS - GROUP_START))s (probe wall-clock; see *_inference_probes.log for per-stage [timing])"
done

echo "Sweep complete."
