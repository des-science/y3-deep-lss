#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --job-name=bench_dataloader
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/claude/bench/dataloader/slurm-%j.out

# GridPipeline input-pipeline throughput (no network, no GPU), one-factor-at-a-time around the
# configs/maps/shared/dataloader.yaml knobs, one process per point. Results: claude/bench/dataloader/.
#   PROBES="lensing combined" sbatch benchmark_dataloader.sh

set -euo pipefail  # the `X=""; [ test ] && X=flag` idiom of the other scripts would abort here
source /users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/common.sh

PROBES="${PROBES:-lensing clustering combined}"
SCRIPT="$DEEP_LSS/deep_lss/apps/benchmark/benchmark_dataloader.py"
OUTDIR="$MYSCRATCH/deep_lss/claude/bench/dataloader/$SLURM_JOB_ID"
RESULTS="$OUTDIR/results.jsonl"
[ "$DRYRUN" = "1" ] || mkdir -p "$OUTDIR"

# Baseline = configs/maps/shared/dataloader.yaml (dset.training).
B_BATCH=16 B_READERS=16 B_PREFETCH=4 B_WORKERS=128 B_FSHUF=64 B_ESHUF=1024

run_probe() {
    local PROBE=$1
    local NET_CONFIG="$DEEP_LSS/configs/maps/prod/transformer/$PROBE/maps.yaml"

    bench() {
        local label=$1
        echo ">>> [$PROBE] $label"
        $SRUN --environment=tensorflow --gpu-bind=none \
            python "$SCRIPT" \
                --train_tfr_pattern="$TRAIN_TFR" \
                --net_config="$NET_CONFIG" \
                --probes_config="$DEEP_LSS/configs/probes/$PROBE.yaml" \
                --scales_config="$SCALES_CONFIG" \
                --data_config="$DATA_CONFIG" \
                --msfm_config="$MSFM_CONFIG" \
                --local_batch_size="$2" --n_readers="$3" --n_prefetch="$4" \
                --n_workers="$5" --file_name_shuffle_buffer="$6" --examples_shuffle_buffer="$7" \
                --measure_batches=40 \
                --label="$PROBE/$label" --results_file="$RESULTS" || echo "!!! failed: $PROBE $label"
    }

    bench baseline $B_BATCH $B_READERS $B_PREFETCH $B_WORKERS $B_FSHUF $B_ESHUF
    for v in 8 32 64; do bench "batch=$v" $v $B_READERS $B_PREFETCH $B_WORKERS $B_FSHUF $B_ESHUF; done
    for v in 8 32 64; do bench "readers=$v" $B_BATCH $v $B_PREFETCH $B_WORKERS $B_FSHUF $B_ESHUF; done
    for v in 2 8 -1; do bench "prefetch=$v" $B_BATCH $B_READERS $v $B_WORKERS $B_FSHUF $B_ESHUF; done
    for v in 256 512 2048; do bench "eshuf=$v" $B_BATCH $B_READERS $B_PREFETCH $B_WORKERS $B_FSHUF $v; done
    for v in 64 288; do bench "workers=$v" $B_BATCH $B_READERS $B_PREFETCH $v $B_FSHUF $B_ESHUF; done
}

for probe in $PROBES; do
    run_probe "$probe"
done
echo "Summarize: python $DEEP_LSS/deep_lss/apps/benchmark/benchmark_dataloader_summary.py $RESULTS"
