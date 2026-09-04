#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --job-name=rate_probe
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

# Measure the SUSTAINED 4-GPU it/s of a maps net config, so n_steps can be sized from a real number
# instead of an extrapolation.
#
# WHY THIS EXISTS. Sizing n_steps wrong is asymmetric and expensive: oversizing forfeits the cosine
# anneal tail AND the eval tail (run_evaluation.py / run_inference.py never run, so the job produces
# nothing scorable at all), while undersizing only wastes wall. The two instruments that were
# available both mislead:
#   * shared/benchmark_sweep.sh is SINGLE-GPU and synthetic -- it under-predicted the ConvNeXt block's
#     real 4-GPU cost by ~40% in bench_v6, erring in the dangerous direction.
#   * extrapolating a measured rate across a width or batch change -- the bench_v6 pre-launch model
#     over-predicted the ConvNeXt advantage by ~25% and nearly killed three runs.
# This script just runs the real thing on 4 GPUs for a few thousand steps and reads the rate off it.
#
# It is the same code path as maps/training.sh (run_training.py, same strategy, same env), with three
# differences: n_steps is overridden to PROBE_STEPS, --wandb is NOT passed (throwaway runs should not
# land in the dashboard), and eval/inference never run. Output goes under deep_lss/claude/bench/.
#
# Usage -- one job per config, submit them in parallel:
#   PROBE=lensing ARCH=transformer \
#   NET_CONFIG=$PWD/configs/maps/dev/transformer/lensing/bench_v7/t1_cls.yaml \
#   sbatch --job-name=rate_lensing submissions/clariden/maps/benchmarks/rate_probe.sh
#
# Then size from the reported rate:
#   n_steps = it/s x 37800   (single probe, one 12 h job:  ~10.5 h training + eval/inference tail)
#   n_steps = it/s x 79200   (combined, 2 x 12 h chain:    ~11 h training in each job)
# and round DOWN to a multiple of 10k.

ulimit -c 0  # a crashing task would otherwise fill the /users quota with a core dump

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

# --- Repository and scratch roots ------------------------------------------------------------

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

DEEP_LSS="$REPOS/y3-deep-lss"
MSFM="$REPOS/multiprobe-simulation-forward-model"

# --- Overridable defaults ---------------------------------------------------------------------

# Same dataset/probe contract as training.sh: v18/default and v16 are extended-NLA and take the plain
# configs/probes/*.yaml; v17 is standard-NLA and needs the bta-free *_nla ones. Getting it wrong does
# not raise.
VERSION="${VERSION:-v18}"
SUBVERSION="${SUBVERSION:-default}"

SCALES="${SCALES:-8wl,32gc}"
LOSS="${LOSS:-vmim}"
PROBE="${PROBE:-lensing}"

# Set NET_CONFIG together with PROBE -- the per-probe configs differ in n_steps, smooth_nside and
# local_batch_size, and the default here is the lensing one. ARCH only names the output dir.
ARCH="${ARCH:-deepsphere}"
NET_CONFIG="${NET_CONFIG:-$DEEP_LSS/configs/maps/prod/deepsphere/lensing/maps+cls.yaml}"

# How many steps to time. Must clear XLA compilation and the dataloader ramp, span at least one
# validation pass (vali_every is typically 1000), AND cross `checkpoint_every` -- a probe that never
# checkpoints does not pay the checkpoint I/O the real run does, which is why the bench_v7 combined
# probe over-predicted its production rate by 9%. `checkpoint_every` is 5000 on the chained configs,
# so the default clears it. Raise it further if the rate is still drifting at the end of the log.
PROBE_STEPS="${PROBE_STEPS:-6000}"

STRATEGY="mirrored"
DATA="default"

# --- Derived paths ------------------------------------------------------------------------------

INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"
NET_NAME="$(basename "${NET_CONFIG%.yaml}")"
TAG="${TAG:-${ARCH}_${PROBE}_${NET_NAME}}"

OUT_DIR="${OUT_DIR:-$MYSCRATCH/deep_lss/claude/bench/rate_probe}"
MODEL_DIR="${MODEL_DIR:-$TAG}"
RUN_DIR="$OUT_DIR/$MODEL_DIR"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/${SLURM_JOB_ID}_training.log"

TRAIN_TFR="$INPUT/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

MSFM_CONFIG="$MSFM/configs/$VERSION/$SUBVERSION.yaml"
PROBES_CONFIG="$DEEP_LSS/configs/probes/${PROBE}.yaml"
SCALES_CONFIG="$DEEP_LSS/configs/scales/${SCALES}.yaml"
LOSS_CONFIG="$DEEP_LSS/configs/loss/${LOSS}.yaml"
DATA_CONFIG="$DEEP_LSS/configs/data/${DATA}.yaml"

# --- The timed config ---------------------------------------------------------------------------

# n_steps comes from run_training.py's --n_steps flag; the config itself is used verbatim, so the
# rate is measured on exactly the geometry that will be trained -- batch size, architecture,
# smoothing and all.
PROBE_CONFIG="$NET_CONFIG"

# The net config is composed: prod configs pull their dset: and cls: blocks in from shared/ via
# extends:, so grepping the file itself finds nothing (and the old batch-size grep matched on
# indentation depth, which composition changes anyway). Resolve once, read the key=value dump.
# Under srun because a bare python3 on a Clariden node is 3.6 with no yaml.
RESOLVED_NET="${LOG%.log}_net_config.txt"
srun --environment=tensorflow --gpu-bind=none --output="$RESOLVED_NET" \
    python -m deep_lss.utils.config_check resolve "$NET_CONFIG" --flat
if [ $? -ne 0 ]; then
    echo "Could not resolve $NET_CONFIG — see $RESOLVED_NET." >&2
    exit 1
fi

echo "=== rate_probe: $TAG"
echo "    net config : $NET_CONFIG"
echo "    timed steps: $PROBE_STEPS"
echo "    batch      : $(grep -E '^dset\.training\.grid\.local_batch_size=' "$RESOLVED_NET" | cut -d= -f2)"
echo "    run dir    : $RUN_DIR"

# --- Cls precache (maps+cls configs only) --------------------------------------------------------

# run_training.py only READS the rebinned-Cls cache and aborts if it is missing. Normally it is
# already built; this is here so a rate probe on a fresh dataset is not a confusing failure.
CLS_N_BINS=$(grep -E '^network\.cls\.n_bins=' "$RESOLVED_NET" | cut -d= -f2)
CLS_CACHE="$INPUT/cls/rebinned_nb${CLS_N_BINS}_${SCALES}.h5"
if grep -qE '^network\.cls\.' "$RESOLVED_NET" && [ ! -f "$CLS_CACHE" ]; then
    echo "Cls cache $CLS_CACHE is missing; build it with maps/training.sh before probing." >&2
    exit 1
fi

# --- Timed training run --------------------------------------------------------------------------

# No --wandb: these models are throwaway and should not appear in the dashboard.
srun --environment=tensorflow --gpu-bind=none --output="$LOG" \
    python "$DEEP_LSS/deep_lss/apps/run_training.py" \
        --dir_base="$OUT_DIR" \
        --dir_model="$MODEL_DIR" \
        --train_tfr_pattern="$TRAIN_TFR" \
        --data_dir="$INPUT" \
        --msfm_config="$MSFM_CONFIG" \
        --probes_config="$PROBES_CONFIG" \
        --scales_config="$SCALES_CONFIG" \
        --loss_config="$LOSS_CONFIG" \
        --data_config="$DATA_CONFIG" \
        --net_config="$PROBE_CONFIG" \
        --n_steps="$PROBE_STEPS" \
        --dist_strategy="$STRATEGY"
status=$?

# --- Report --------------------------------------------------------------------------------------

echo ""
echo "=== rate_probe result: $TAG (exit $status)"

# run_training.py now writes its own measured rate to throughput.json (ThroughputTracker), which is
# authoritative: it is binned, so it excludes compilation and exposes drift. Prefer it, and keep the
# log-scraping below only as a fallback for a run that died before writing the file.
if [ -f "$RUN_DIR/throughput.json" ]; then
    RATE=$(python3 -c "import json,sys; print(f\"{json.load(open(sys.argv[1]))['sustained_it_per_s']:.2f}\")" \
        "$RUN_DIR/throughput.json" 2>/dev/null)
    DRIFT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('drift_percent'))" \
        "$RUN_DIR/throughput.json" 2>/dev/null)
    echo "    drift across probe : ${DRIFT}%  (large drift means a fixed n_steps cannot be sized reliably --"
    echo "                         use training.wall_budget_seconds instead)"
fi

# !! DO NOT READ THE it/s FIGURE TQDM PRINTS. !!
# On its final line tqdm reports the CUMULATIVE average over the whole run, which includes XLA
# compilation and the dataloader ramp -- one-time costs that dominate a 3000-step probe and are
# negligible over a real 100k+ step run. Measured 2026-08-06 on the bench_v7 transformer arms, that
# drag is 29-44% (lensing 3.70 cumulative vs 4.78 sustained; clustering 4.87 vs 7.00), because
# jit_compile_body: true makes compilation expensive. Sizing off the cumulative figure would have
# undersized every arm by a third. The GCNN arms show a smaller but still real ~9%.
#
# So: parse (step, elapsed) pairs out of the progress bar and take a WINDOW over the second half,
# which is past compilation and still spans the validation stalls the real run also pays.
RATE=${RATE:-$(tr '\r' '\n' < "$LOG" | awk -v total="$PROBE_STEPS" '
    match($0, /[0-9]+\/[0-9]+ \[[0-9:]+</) {
        seg = substr($0, RSTART, RLENGTH)
        split(seg, a, /[\/ \[<]/)          # a[1]=step  a[2]=total  a[4]=elapsed
        step = a[1] + 0
        n = split(a[4], t, ":")            # MM:SS or H:MM:SS
        secs = (n == 3) ? t[1]*3600 + t[2]*60 + t[3] : t[1]*60 + t[2]
        if (step >= total/2 && !mid_set) { mid_s = step; mid_t = secs; mid_set = 1 }
        last_s = step; last_t = secs
    }
    END {
        if (mid_set && last_t > mid_t && last_s > mid_s)
            printf "%.2f", (last_s - mid_s) / (last_t - mid_t)
    }')}

if [ -z "$RATE" ]; then
    echo "Could not parse a rate from $LOG -- inspect it by hand." >&2
    tail -30 "$LOG" >&2
    exit "${status:-1}"
fi

# The budget constants below are MEASURED, not assumed: across the six bench_v7 single-probe runs the
# eval+inference tail was 1799-2173 s regardless of probe, architecture or step count, so a 12 h job
# has ~41.0 ks of training in it -- not the 37.8 ks originally guessed, which cost every one of those
# runs 1.0-1.7 h of idle GPU.
SINGLE=$(awk -v r="$RATE" 'BEGIN{printf "%d", int(r*41000/10000)*10000}')
CHAIN=$(awk -v r="$RATE" 'BEGIN{printf "%d", int(r*83900/10000)*10000}')

echo "    sustained rate : ${RATE} it/s (4 GPUs, ${PROBE_STEPS} steps)"
echo "    n_steps  1x12h : ${SINGLE}   (rate x 41000, rounded DOWN to 10k)"
echo "    n_steps  2x12h : ${CHAIN}   (rate x 83900, rounded DOWN to 10k)"
echo "    log            : $LOG"
echo ""
echo "    PREFER A WALL-CLOCK BUDGET over these step counts: set training.wall_budget_seconds to"
echo "    41000 (1 x 12 h) or 83900 (2 x 12 h) with n_steps: auto, and the run fills its allocation"
echo "    and anneals correctly whatever rate it actually achieves. A fixed n_steps only works if the"
echo "    rate is stable, and it is not always -- see deep_lss/utils/throughput.py."

printf '{"tag": "%s", "probe": "%s", "arch": "%s", "config": "%s", "timed_steps": %s, "it_per_s": %s, "n_steps_1x12h": %s, "n_steps_2x12h": %s}\n' \
    "$TAG" "$PROBE" "$ARCH" "$NET_CONFIG" "$PROBE_STEPS" "$RATE" "$SINGLE" "$CHAIN" \
    >> "$OUT_DIR/rates.jsonl"

exit "$status"
