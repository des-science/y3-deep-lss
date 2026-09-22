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
#SBATCH --job-name=coverage_round5
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/coverage_round5-%j.out

# Prepared coverage experiments, following multiprobe-simulation-inference's
# submissions/clariden/flow_sweep.sh: four concurrent srun steps, one GPU each.
# Reserve four task slots and override the inherited node GPU count in each step.
# Without --gpus-per-node=1, srun can reserve all four GPUs and serialize the arms.
#
#   sbatch submissions/clariden/experiments/coverage_round.sh
#   DRYRUN=1 bash submissions/clariden/experiments/coverage_round.sh
#   ARMS="joint_long conditional_long" sbatch submissions/clariden/experiments/coverage_round.sh
#
# During the first real launch, check `squeue -s -u "$USER"` for four concurrent
# steps and the job log for "step creation still disabled" (there should be none).
# Submit with --uenv-passthrough=ignore if already inside a uenv session.
set -euo pipefail
ulimit -c 0

# Absolute roots work even when Slurm runs its spool copy of this script.
REPOS=${REPOS:-/users/athomsen/dlss/repos}
MSI_REPO="$REPOS/multiprobe-simulation-inference"
TORCH_PYTHON=${TORCH_PYTHON:-/users/athomsen/dlss/torch_env/bin/python}
ROUND_DIR=${ROUND_DIR:-/users/athomsen/dlss/scratch/runs/v18/default/maps_gcnn/combined/v1/flow_round5}
ARMS=${ARMS-"all_short all_long projected_long joint_long conditional_long wide_short wide_long extended_long"}
read -r -a ARM_LIST <<< "$ARMS"
GPUS_PER_NODE=4

export PYTHONPATH="$MSI_REPO${PYTHONPATH:+:$PYTHONPATH}"
export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg

[[ -f "$ROUND_DIR/manifest.json" ]] || { echo "Missing prepared manifest: $ROUND_DIR/manifest.json" >&2; exit 1; }
[[ -f "$MSI_REPO/msi/apps/coverage_round.py" ]] || { echo "Missing coverage round runner" >&2; exit 1; }
# The venv interpreter may link into /user-environment, mounted only inside uenv.
[[ -x "$TORCH_PYTHON" || -L "$TORCH_PYTHON" ]] || { echo "Missing Python executable: $TORCH_PYTHON" >&2; exit 1; }
(( ${#ARM_LIST[@]} > 0 )) || { echo "ARMS must not be empty" >&2; exit 1; }
declare -A SEEN=()
for arm in "${ARM_LIST[@]}"; do
    case "$arm" in
        all_short|all_long|projected_long|joint_long|conditional_long|wide_short|wide_long|extended_long) ;;
        *) echo "Unknown arm: $arm" >&2; exit 1 ;;
    esac
    [[ -z ${SEEN[$arm]+present} ]] || { echo "Duplicate arm: $arm" >&2; exit 1; }
    SEEN[$arm]=1
    [[ -f "$ROUND_DIR/arms/$arm.yaml" ]] || { echo "Missing prepared config for $arm" >&2; exit 1; }
    if [[ -e "$ROUND_DIR/${arm}.started" || -e "$ROUND_DIR/${arm}_ensemble_flow_229900" ]]; then
        echo "Arm already started or has results: $arm; choose fresh arms or a new round directory" >&2
        exit 1
    fi
done

N_ARMS=${#ARM_LIST[@]}
N_WAVES=$(( (N_ARMS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
printf 'coverage_round5: %s arms, %s GPUs, %s waves; one GPU per arm\n' "$N_ARMS" "$GPUS_PER_NODE" "$N_WAVES"
printf 'Prepared round: %s\n' "$ROUND_DIR"
for ((i=0; i<N_ARMS; i++)); do
    printf '  wave %s, slot %s: %s\n' "$((i / GPUS_PER_NODE + 1))" "$((i % GPUS_PER_NODE + 1))" "${ARM_LIST[i]}"
done
if [[ ${DRYRUN:-0} == 1 ]]; then
    echo "DRYRUN=1: no jobs or steps launched."
    exit 0
fi
: "${SLURM_JOB_ID:?Submit with sbatch, or set DRYRUN=1 for a local preview}"
mkdir -p "$ROUND_DIR/logs"

FAILED=()
for ((start=0; start<N_ARMS; start+=GPUS_PER_NODE)); do
    printf 'Starting wave %s/%s\n' "$((start / GPUS_PER_NODE + 1))" "$N_WAVES"
    PIDS=(); TAGS=()
    for ((i=start; i<start+GPUS_PER_NODE && i<N_ARMS; i++)); do
        arm=${ARM_LIST[i]}
        srun -N1 --ntasks=1 --ntasks-per-node=1 --exact --exclusive \
        --gpus-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
            --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="$ROUND_DIR/logs/${SLURM_JOB_ID}_${arm}.log" \
            "$TORCH_PYTHON" -m msi.apps.coverage_round run --round "$ROUND_DIR" --arm "$arm" &
        PIDS+=("$!"); TAGS+=("$arm")
    done
    # Launch all four BEFORE waiting, and preserve each step's failure status.
    for j in "${!PIDS[@]}"; do
        if ! wait "${PIDS[j]}"; then
            echo "ARM FAILED: ${TAGS[j]}" >&2
            FAILED+=("${TAGS[j]}")
        fi
    done
done
if (( ${#FAILED[@]} )); then
    echo "Failed arms: ${FAILED[*]}" >&2
    exit 1
fi
echo "All $N_ARMS arms completed."
