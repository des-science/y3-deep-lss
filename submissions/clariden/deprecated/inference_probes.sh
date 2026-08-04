#!/bin/bash
# Moved to deprecated/ 2026-08-04: hardcoded to dead v16 model names (v33), never
# parameterized. For a one-off multi-probe inference sweep today, loop inference.sh by hand
# (see its header) or add a new script under dev/.
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=inference
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"

# SUMMARY="maps"
# MODEL="v8_cls"

SUMMARY="cls"
MODEL="v33"

# Each probe in PROBES runs on its own GPU in parallel.
# Use a single-element list to run just one probe on one GPU.
# PROBES=("lensing" "clustering" "combined")
PROBES=("lensing")

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

run_inference() {
    local probe="$1"
    local output="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/$SUMMARY/$probe"
    local log="$output/$MODEL/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$log")"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${log}_inference.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
            --out_dir=\"$output\" \
            --model_name=\"$MODEL\" \
            --flow_config=\"$FLOW_CONFIG\" \
            --n_flows=4 \
            --sample_posterior \
            --include_grid \
            --include_des \
            --include_mocks"
}

for probe in "${PROBES[@]}"; do
    run_inference "$probe" &
done
wait
