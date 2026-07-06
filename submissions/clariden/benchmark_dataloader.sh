#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --job-name=bench_dataloader
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/bench_dataloader/slurm-%j.out

# Benchmark the GridPipeline input pipeline (no network, no GPU) across an OFAT sweep of the
# performance knobs, for lensing / clustering / combined. One python process per configuration so
# each peak-RSS measurement is clean. Results are appended as JSON lines to $RESULTS.
#
# CPU-only benchmark, but we still take a whole node so the 288-core thread pools match training.
# Outputs go to /iopsstor scratch (home VAST quota silently drops big writes).

set -euo pipefail

# match training.sh thread-pool tuning for the 288 cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

VERSION="v16"
SUBVERSION="rot_in_place"
SCALES="8wl,32gc"
DATA="default"

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"
MSFM_CONFIG="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml"

OUTDIR="$MYSCRATCH/deep_lss/runs/bench_dataloader/${SLURM_JOB_ID:-manual}"
mkdir -p "$OUTDIR"
RESULTS="$OUTDIR/results.jsonl"
echo "Writing results to $RESULTS"

SCRIPT="$REPOS/y3-deep-lss/deep_lss/apps/benchmark_dataloader.py"

# probe -> (probes_config, net_config). Clustering has no dedicated net config yet; it reuses the
# lensing transformer architecture (arch is irrelevant to the input pipeline — same 4 z-bins, same
# native n_side, so its raw dataloading matches lensing). Combined exercises 8 z-bins.
run_probe () {
    local PROBE="$1"
    local PROBES_CONFIG NET_CONFIG
    case "$PROBE" in
        lensing)    PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/lensing.yaml"
                    NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/lensing/maps.yaml" ;;
        clustering) PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/clustering.yaml"
                    NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/lensing/maps.yaml" ;;
        combined)   PROBES_CONFIG="$REPOS/y3-deep-lss/configs/probes/combined.yaml"
                    NET_CONFIG="$REPOS/y3-deep-lss/configs/transformer/combined/maps.yaml" ;;
        *) echo "unknown probe $PROBE"; return 1 ;;
    esac

    # baseline knobs (the current maps.yaml defaults)
    local B_BATCH=16 B_READERS=64 B_PREFETCH=8 B_WORKERS=-1 B_FSHUF=64 B_ESHUF=256

    # one run with explicit knobs
    bench () {
        local label="$1" batch="$2" readers="$3" prefetch="$4" workers="$5" fshuf="$6" eshuf="$7"
        echo ">>> [$PROBE] $label  batch=$batch readers=$readers prefetch=$prefetch workers=$workers eshuf=$eshuf"
        srun --environment=tensorflow --gpu-bind=none \
            python "$SCRIPT" \
                --train_tfr_pattern="$TRAIN_TFR" \
                --net_config="$NET_CONFIG" \
                --probes_config="$PROBES_CONFIG" \
                --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --msfm_config="$MSFM_CONFIG" \
                --local_batch_size="$batch" --n_readers="$readers" --n_prefetch="$prefetch" \
                --n_workers="$workers" --file_name_shuffle_buffer="$fshuf" --examples_shuffle_buffer="$eshuf" \
                --measure_batches=40 \
                --label="${PROBE}/${label}" --results_file="$RESULTS" || echo "!!! run failed: $PROBE $label"
    }

    # baseline (run once)
    bench baseline $B_BATCH $B_READERS $B_PREFETCH $B_WORKERS $B_FSHUF $B_ESHUF
    # local_batch_size sweep (the sizes the user cares about)
    for v in 8 32 64;   do bench "batch=$v"    $v $B_READERS $B_PREFETCH $B_WORKERS $B_FSHUF $B_ESHUF; done
    # n_readers sweep (can we shrink it to save RAM?)
    for v in 8 16 32 128; do bench "readers=$v" $B_BATCH $v $B_PREFETCH $B_WORKERS $B_FSHUF $B_ESHUF; done
    # n_prefetch sweep (-1 == AUTOTUNE)
    for v in 2 4 -1;    do bench "prefetch=$v" $B_BATCH $B_READERS $v $B_WORKERS $B_FSHUF $B_ESHUF; done
    # examples_shuffle_buffer sweep (spare RAM -> larger?  is it per-node?)
    for v in 128 512 1024 2048; do bench "eshuf=$v" $B_BATCH $B_READERS $B_PREFETCH $B_WORKERS $B_FSHUF $v; done
    # n_workers sweep (bounded vs AUTOTUNE)
    for v in 64 128 288; do bench "workers=$v" $B_BATCH $B_READERS $B_PREFETCH $v $B_FSHUF $B_ESHUF; done
}

# PROBES may be overridden from the environment, e.g. PROBES="lensing" sbatch benchmark_dataloader.sh
PROBES="${PROBES:-lensing clustering combined}"
for probe in $PROBES; do
    run_probe "$probe"
done

echo "Done. Summarize with:"
echo "  python $REPOS/y3-deep-lss/deep_lss/apps/benchmark_dataloader_summary.py $RESULTS"
