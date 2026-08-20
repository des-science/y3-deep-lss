#!/bin/bash
# Submit one training.sh job (or, with CHAIN=1, an N-job training_chainer.sh chain) per config
# file in a directory -- generic form of the "for f in configs/.../*.yaml; do sbatch ...; done"
# loop that's been hand-retyped into bench_v4/v5/v6's docs each round. Maps-specific (hardcodes
# ../training.sh / ../training_chainer.sh) -- lives under maps/experiments/ for that reason.
#
# Usage:
#   ARCH=deepsphere PROBE=combined CONFIGS_GLOB="configs/deepsphere/dev/combined/bench_v7/*.yaml" \
#       MODEL_PREFIX=bench_v7 ./sweep_configs.sh
#   CHAIN=1 MAX_RUNS=2 ... ./sweep_configs.sh        # each config becomes a chained N-job run
#
# All training.sh env vars (VERSION, SUBVERSION, PROBE, LOSS, ARCH, SCALES, ...) are forwarded.

# --- Repository roots --------------------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
DEEP_LSS="$REPOS/y3-deep-lss"
MAPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."  # resolved from here, not the caller's CWD

# --- Overridable defaults ----------------------------------------------------------------------

: "${CONFIGS_GLOB:?set CONFIGS_GLOB, e.g. configs/deepsphere/dev/combined/bench_v7/*.yaml}"
: "${MODEL_PREFIX:?set MODEL_PREFIX, e.g. bench_v7}"
CHAIN="${CHAIN:-0}"  # 1 submits a training_chainer.sh chain per config instead of a single job

# --- Derived config list -----------------------------------------------------------------------

GLOB_ABS="$CONFIGS_GLOB"
case "$CONFIGS_GLOB" in /*) ;; *) GLOB_ABS="$DEEP_LSS/$CONFIGS_GLOB" ;; esac

shopt -s nullglob
CONFIGS=($GLOB_ABS)
shopt -u nullglob
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs matched $CONFIGS_GLOB" >&2
    exit 1
fi

# --- Submit one run per config -----------------------------------------------------------------

for f in "${CONFIGS[@]}"; do
    name="$(basename "${f%.yaml}")"
    model_dir="${MODEL_PREFIX}_${name}"
    echo "=== $model_dir  ($f) ==="
    if [ "$CHAIN" = "1" ]; then
        NET_CONFIG="$f" MODEL_DIR="$model_dir" "$MAPS_DIR/training_chainer.sh"
    else
        NET_CONFIG="$f" MODEL_DIR="$model_dir" \
            sbatch --job-name="$model_dir" --export=ALL "$MAPS_DIR/training.sh"
    fi
done
