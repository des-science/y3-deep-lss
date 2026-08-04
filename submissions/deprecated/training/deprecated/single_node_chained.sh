#!/bin/bash
#SBATCH --account=m5030_g
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=training
#SBATCH --output="./logs/v15/training_%j.log"

# -------- USER CONFIG --------
MAX_RUNS=5                  # total number of chained runs
RUN_NUM=${RUN_NUM:-1}       # defaults to 1 if not set
STRATEGY="mirrored"
VERSION="v15"
# PROBE="lensing"
# PROBE="clustering"
PROBE="combined"
BIAS="extended"
LOSS="mutual_info"
# -----------------------------

SUBVERSION=$BIAS
OUTPUT="./logs/$VERSION/$PROBE/$LOSS/${STRATEGY}_${SLURM_JOB_ID}"

if [[ $LOSS == "delta" ]]; then
    TRAINSET="fiducial"
else
    TRAINSET="grid"
fi

TRAIN_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/$TRAINSET/DESy3_${TRAINSET}_dmb_????.tfrecord"
FIDU_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/fiducial/DESy3_fiducial_dmb_????.tfrecord"
GRID_EVAL_TFR="/pscratch/sd/a/athomsen/v11desy3/$VERSION/$SUBVERSION/tfrecords/grid/DESy3_grid_dmb_????.tfrecord"

echo "===== Starting run $RUN_NUM of $MAX_RUNS ====="

# ---- Restore checkpoint for later runs ----
RESTORE_FLAG=""
if [ "$RUN_NUM" -gt 1 ]; then
    RESTORE_FLAG="--restore_checkpoint"
fi

# ---- Training command ----
srun --cpu-bind=threads --gpu-bind=none --output="${OUTPUT}_training.log" \
    python ../../deep_lss/apps/run_training.py \
    --dist_strategy="$STRATEGY" \
    --dir_base="/pscratch/sd/a/athomsen/run_files/$VERSION/$SUBVERSION/$PROBE/$LOSS" \
    --dir_model="default/v1" \
    --loss_function="$LOSS" \
    --train_tfr_pattern=$TRAIN_TFR \
    --grid_vali_tfr_pattern=$TRAIN_TFR \
    --dlss_config="configs/$VERSION/default/$PROBE/dlss.yaml" \
    --net_config="configs/$VERSION/deepsphere_default.yaml" \
    --msfm_config="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
    --slurm_output="${OUTPUT}_training" \
    $RESTORE_FLAG \
    --wandb \
    --wandb_tags "$VERSION" "$PROBE" "$LOSS" "$STRATEGY" "$BIAS" "$SUBVERSION" "resnet" \
    --wandb_notes="single $PROBE node run $RUN_NUM"

# NOTE this has the disadvantage that the job only gets queued once the previous one has finished. So time-wise, the 
# only advantage over submitting manually is that this submission takes place right away. But there's no speed up
# with respect to the queue

# ---- Schedule next run if needed ----
if [ "$RUN_NUM" -lt "$MAX_RUNS" ]; then
    NEXT_RUN=$((RUN_NUM + 1))
    sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,RUN_NUM=$NEXT_RUN $0
fi