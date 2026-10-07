#!/bin/bash
# Submits one prod/maps.sh job (CHAIN=1: one maps_chain.sh chain) per net config matching a glob,
# into maps/<probe>/<RUN_PREFIX>_<config name>. Run it on the login node; maps.sh variables forward.
#   PROBE=combined CONFIGS_GLOB="configs/maps/dev/deepsphere/combined/bench_v12/*.yaml" \
#       RUN_PREFIX=bench_v12 CHAIN=1 ./sweep_configs.sh

source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

: "${CONFIGS_GLOB:?set CONFIGS_GLOB, e.g. configs/maps/dev/deepsphere/combined/bench_v12/*.yaml}"
: "${RUN_PREFIX:?set RUN_PREFIX, e.g. bench_v12}"
CHAIN="${CHAIN:-0}"
PROD="$DEEP_LSS/submissions/clariden/prod"

resolve_glob "$CONFIGS_GLOB"
for f in "${CONFIGS[@]}"; do
    run="${RUN_PREFIX}_$(basename "${f%.yaml}")"
    echo "=== $run ($f)"
    if [ "$CHAIN" = "1" ]; then
        NET_CONFIG="$f" RUN="$run" JOB_NAME="$run" "$PROD/maps_chain.sh"
    elif [ "$DRYRUN" = "1" ]; then
        echo "NET_CONFIG=$f RUN=$run sbatch --job-name=$run --export=ALL $PROD/maps.sh"
    else
        NET_CONFIG="$f" RUN="$run" sbatch --job-name="$run" --export=ALL "$PROD/maps.sh"
    fi
done
