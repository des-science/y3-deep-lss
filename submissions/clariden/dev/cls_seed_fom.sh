#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_seed_fom
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/slurm/slurm-%j.out

# Does the median-over-grid FoM move across TRAINING SEEDS?
#
# The median over the grid removes per-observation noise (finite posterior samples, and the real
# spread of FoM across the Om-S8 plane) -- that part is settled. It cannot remove training-seed
# noise, which is common-mode: a different seed learns a different summary, so every observation's
# FoM shifts together and the median rides along.
#
# How big that shift is for FoM specifically is an open question, and NOT answerable from the
# vali_nmse_cosmo scatter (sd ~2.3% across 3 MLP seeds): nmse is the accuracy of the posterior MEAN,
# FoM is the posterior WIDTH. Widths may well be the more stable of the two.
#
# So: run inference on the 3 seeds x 2 configs that already have trained checkpoints from jobs
# 2770501 / 2770593, and compare the seed spread of median FoM WITHIN a config against the gap
# BETWEEN configs. That is the number that says whether a single-seed v2_mlp vs v2_transformer FoM
# comparison means anything.
#
# Training-only runs, so grid preds exist but DES/mock preds do not -> --include_grid only.

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v17"
SUBVERSION="baseline"
PROBE="lensing_nla"
OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# 3 seeds x 2 configs. Seed 42 came from the ladder (job 2770501), 43/44 from the replication
# (job 2770593), so the model_name prefixes differ.
MODELS=("v1_bench_d128_L2" "v1_seeds_d128_L2_s43" "v1_seeds_d128_L2_s44"
        "v1_bench_mlp_ref"  "v1_seeds_mlp_ref_s43"  "v1_seeds_mlp_ref_s44")

mkdir -p "$MYSCRATCH/deep_lss/slurm"

i=0
for M in "${MODELS[@]}"; do
    LOG="$OUTPUT/$M/logs/${SLURM_JOB_ID}_inference"
    mkdir -p "$(dirname "$LOG")"

    (
        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="${LOG}.log" \
            bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$M\" \
                --flow_config=\"$FLOW_CONFIG\" \
                --n_flows=4 \
                --sample_posterior \
                --include_grid"
    ) &

    i=$((i + 1))
    if [ $((i % 4)) -eq 0 ]; then
        wait
    fi
done

wait
