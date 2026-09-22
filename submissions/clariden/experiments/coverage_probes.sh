#!/usr/bin/env bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=coverage_probes
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/coverage_probes-%j.out

# Four concurrent steps: matched baseline and extended likelihood for each probe.
set -euo pipefail
ulimit -c 0
REPOS=${REPOS:-/users/athomsen/dlss/repos}
ROOT=${ROOT:-/users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn}
TORCH_PYTHON=${TORCH_PYTHON:-/users/athomsen/dlss/torch_env/bin/python}
export PYTHONPATH="$REPOS/multiprobe-simulation-inference${PYTHONPATH:+:$PYTHONPATH}"
export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
PROBES=(lensing lensing clustering clustering)
ARMS=(all_long extended_long all_long extended_long)
for i in "${!ARMS[@]}"; do
    round="$ROOT/${PROBES[i]}/v1/flow_round5_probes_clean"
    arm=${ARMS[i]}
    [[ -f "$round/manifest.json" && -f "$round/arms/$arm.yaml" ]] || { echo "Missing prepared round: $round" >&2; exit 1; }
    [[ ! -e "$round/$arm.started" ]] || { echo "Already started: $round/$arm" >&2; exit 1; }
    printf 'GPU slot %s: %s / %s\n' "$i" "${PROBES[i]}" "$arm"
done
[[ ${DRYRUN:-0} != 1 ]] || exit 0
: "${SLURM_JOB_ID:?Submit with sbatch or set DRYRUN=1}"
PIDS=()
for i in "${!ARMS[@]}"; do
    round="$ROOT/${PROBES[i]}/v1/flow_round5_probes_clean"
    mkdir -p "$round/logs"
    srun -N1 --ntasks=1 --ntasks-per-node=1 --exact --exclusive \
        --gpus-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
        --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
        --output="$round/logs/${SLURM_JOB_ID}_${ARMS[i]}.log" \
        "$TORCH_PYTHON" -m msi.apps.coverage_round run --round "$round" --arm "${ARMS[i]}" &
    PIDS+=("$!")
done
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[i]}"; then
        echo "FAILED: ${PROBES[i]} / ${ARMS[i]}" >&2
        FAILED=1
    fi
done
(( FAILED == 0 )) || exit 1
# Score each completed pair in the allocation, keeping login-node work small.
for probe in lensing clustering; do
    round="$ROOT/$probe/v1/flow_round5_probes_clean"
    srun -N1 --ntasks=1 --ntasks-per-node=1 --exact --exclusive \
        --gpus-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
        --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
        "$TORCH_PYTHON" -m msi.apps.score_coverage_round --round "$round" --output "$round/scores.json"
done
