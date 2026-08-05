#!/bin/bash
# Moved to deprecated/ 2026-08-04: hardcoded to dead v16 model names (v8_cls) and a fixed
# step list, never parameterized. For a one-off multi-step inference sweep today, loop
# maps/rerun/inference.sh by hand with N_STEPS_2/--n_steps overrides (see its header) or add
# a new script under maps/experiments/.
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

# MODEL="v6"
MODEL="v8_cls"

PROBE="combined"

N_STEPS=(200000 250000)

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/maps/$PROBE"

run_inference() {
    local n_steps="$1"
    local log="$OUTPUT/$MODEL/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$log")"

    srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
        --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="${log}_${n_steps}_inference.log" \
        bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
            --out_dir=\"$OUTPUT\" \
            --model_name=\"$MODEL\" \
            --n_steps=$n_steps \
            --flow_config=\"$FLOW_CONFIG\" \
            --n_flows=4 \
            --sample_posterior \
            --include_grid \
            --include_des \
            --include_mocks"
}

for n_steps in "${N_STEPS[@]}"; do
    run_inference "$n_steps" &
done
wait
