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
#SBATCH --job-name=coverage_round
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/coverage_round-%j.out

# Run prepared arms of the flow-coverage experiment, four concurrent srun steps, one GPU each.
# Reserve four task slots and override the inherited node GPU count in each step; without
# --gpus-per-node=1, srun can reserve all four GPUs and serialize the arms.
#
# THE ROUND DIRECTORY IS THE CONFIG. `python -m msi.apps.coverage_round prepare --preds <file>
# --output <dir>` freezes the inputs, the mock identities, the update budgets and one YAML per arm
# into <dir>, and hashes the code it depends on. This script only executes what that wrote: it reads
# the arm names and the checkpoint out of <dir>/manifest.json rather than carrying its own copy, so
# a new arm in coverage_round.ARMS needs no edit here. There is deliberately NO default round --
# a round is an experiment, and pointing at someone else's by accident is the failure this avoids.
#
#   ROUNDS=/path/to/round sbatch coverage_round.sh                  # every incomplete arm
#   ROUNDS=/path/to/round:joint_long,conditional_long sbatch ...    # named arms
#   ROUNDS="$A:all_long,extended_long $B:all_long,extended_long" SCORE=1 sbatch ...
#   ROUNDS=/path/to/round DRYRUN=1 bash coverage_round.sh           # plan only, login node
#
# SCORE=1 adds a scoring pass per round once its arms finish (msi.apps.score_coverage_round ->
# <round>/scores.json), which is what the per-probe confirmation rounds want; it stays in the
# allocation so no login-node work is needed.
#
# During the first real launch, check `squeue -s -u "$USER"` for four concurrent steps and the job
# log for "step creation still disabled" (there should be none).
# Submit with --uenv-passthrough=ignore if already inside a uenv session.
set -euo pipefail
ulimit -c 0

# Absolute roots work even when Slurm runs its spool copy of this script.
REPOS=${REPOS:-/users/athomsen/dlss/repos}
MSI_REPO="$REPOS/multiprobe-simulation-inference"
TORCH_PYTHON=${TORCH_PYTHON:-/users/athomsen/dlss/torch_env/bin/python}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SCORE=${SCORE:-0}

export PYTHONPATH="$MSI_REPO${PYTHONPATH:+:$PYTHONPATH}"
export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg

: "${ROUNDS:?Set ROUNDS to one or more <round_dir>[:arm,arm,...] entries}"
read -r -a ROUND_SPECS <<< "$ROUNDS"
[[ -f "$MSI_REPO/msi/apps/coverage_round.py" ]] || { echo "Missing coverage round runner" >&2; exit 1; }
# The venv interpreter may link into /user-environment, mounted only inside uenv.
[[ -x "$TORCH_PYTHON" || -L "$TORCH_PYTHON" ]] || { echo "Missing Python executable: $TORCH_PYTHON" >&2; exit 1; }

# manifest.json is read with the SYSTEM python3 (3.6 on Clariden, stdlib json only): the script body
# runs outside uenv, where $TORCH_PYTHON cannot resolve its /user-environment links.
manifest_field() {  # <round_dir> <field>  -- arms (space-separated) or checkpoint
    python3 -c '
import json, os, sys
round_dir = sys.argv[1]
m = json.load(open(round_dir + "/manifest.json"))
if sys.argv[2] == "arms":
    print(" ".join(m["arms"]))
    raise SystemExit
# Mirrors msi.apps.coverage_round.manifest_checkpoint: rounds prepared before `checkpoint` became a
# manifest key carry it only in the frozen prediction_file name.
if "checkpoint" in m:
    print(m["checkpoint"])
    raise SystemExit
stem = os.path.basename(m.get("prediction_file", "")).split(".")[0]
if stem.startswith("preds_") and stem[6:].isdigit():
    print(stem[6:])
    raise SystemExit
sys.exit("%s/manifest.json records no checkpoint; prepare a fresh round directory." % round_dir)
' "$1" "$2"
}

# --- Resolve every (round, arm) pair before launching anything -------------------------------

ROUND_OF=(); ARM_OF=()
declare -A SEEN=()
for spec in "${ROUND_SPECS[@]}"; do
    round=${spec%%:*}
    round=${round%/}
    [[ -f "$round/manifest.json" ]] || { echo "Missing prepared manifest: $round/manifest.json" >&2; exit 1; }
    known=$(manifest_field "$round" arms)
    checkpoint=$(manifest_field "$round" checkpoint)
    if [[ "$spec" == *:* ]]; then
        IFS=, read -r -a wanted <<< "${spec#*:}"
    else
        read -r -a wanted <<< "$known"
    fi
    (( ${#wanted[@]} > 0 )) || { echo "No arms selected for $round" >&2; exit 1; }
    for arm in "${wanted[@]}"; do
        [[ " $known " == *" $arm "* ]] || { echo "Unknown arm '$arm' in $round (has: $known)" >&2; exit 1; }
        [[ -f "$round/arms/$arm.yaml" ]] || { echo "Missing prepared config for $arm in $round" >&2; exit 1; }
        [[ -z ${SEEN["$round/$arm"]+present} ]] || { echo "Duplicate: $round/$arm" >&2; exit 1; }
        SEEN["$round/$arm"]=1
        # Skip a finished arm instead of failing, so re-submitting a partly-failed round is one
        # command. An arm that started but never completed still blocks -- its output is unvalidated.
        if [[ -e "$round/$arm.complete" ]]; then
            printf 'skipping completed arm: %s / %s\n' "$round" "$arm"
            continue
        fi
        if [[ -e "$round/$arm.started" || -e "$round/${arm}_ensemble_flow_${checkpoint}" ]]; then
            echo "Arm started but not complete: $round/$arm -- clear it or choose another round" >&2
            exit 1
        fi
        ROUND_OF+=("$round"); ARM_OF+=("$arm")
    done
done

N_ARMS=${#ARM_OF[@]}
(( N_ARMS > 0 )) || { echo "Nothing to run: every selected arm is already complete."; exit 0; }
N_WAVES=$(( (N_ARMS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
printf 'coverage_round: %s arms, %s GPUs, %s waves; one GPU per arm\n' "$N_ARMS" "$GPUS_PER_NODE" "$N_WAVES"
for ((i=0; i<N_ARMS; i++)); do
    printf '  wave %s, slot %s: %s / %s\n' \
        "$((i / GPUS_PER_NODE + 1))" "$((i % GPUS_PER_NODE + 1))" "${ROUND_OF[i]}" "${ARM_OF[i]}"
done
if [[ ${DRYRUN:-0} == 1 ]]; then
    echo "DRYRUN=1: no jobs or steps launched."
    exit 0
fi
: "${SLURM_JOB_ID:?Submit with sbatch, or set DRYRUN=1 for a local preview}"

# --- Launch, one wave at a time --------------------------------------------------------------

FAILED=()
for ((start=0; start<N_ARMS; start+=GPUS_PER_NODE)); do
    printf 'Starting wave %s/%s\n' "$((start / GPUS_PER_NODE + 1))" "$N_WAVES"
    PIDS=(); TAGS=()
    for ((i=start; i<start+GPUS_PER_NODE && i<N_ARMS; i++)); do
        round=${ROUND_OF[i]}; arm=${ARM_OF[i]}
        mkdir -p "$round/logs"
        srun -N1 --ntasks=1 --ntasks-per-node=1 --exact --exclusive \
            --gpus-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
            --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="$round/logs/${SLURM_JOB_ID}_${arm}.log" \
            "$TORCH_PYTHON" -m msi.apps.coverage_round run --round "$round" --arm "$arm" &
        PIDS+=("$!"); TAGS+=("$round / $arm")
    done
    # Launch the whole wave BEFORE waiting, and preserve each step's failure status.
    for j in "${!PIDS[@]}"; do
        if ! wait "${PIDS[j]}"; then
            echo "ARM FAILED: ${TAGS[j]}" >&2
            FAILED+=("${TAGS[j]}")
        fi
    done
done
if (( ${#FAILED[@]} )); then
    printf 'Failed arms:\n'; printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
echo "All $N_ARMS arms completed."

if [[ "$SCORE" == "1" ]]; then
    for round in $(printf '%s\n' "${ROUND_OF[@]}" | sort -u); do
        printf 'Scoring %s\n' "$round"
        srun -N1 --ntasks=1 --ntasks-per-node=1 --exact --exclusive \
            --gpus-per-node=1 --gpus-per-task=1 --cpus-per-task=72 --mem=110G \
            --cpu-bind=none --uenv=pytorch/v2.9.1:v2 --view=default \
            "$TORCH_PYTHON" -m msi.apps.score_coverage_round --round "$round" --output "$round/scores.json"
    done
fi
