#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --job-name=cls_training
#SBATCH --output=/users/athomsen/dlss/repos/y3-deep-lss/submissions/clariden/slurm/slurm-%j.out

export SLURM_CPUS_PER_TASK=72
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

# VERSION="v16"
VERSION="v17"
# SUBVERSION="rot_in_place"
SUBVERSION="baseline"
INPUT="$MYSCRATCH/deep_lss/data/$VERSION/$SUBVERSION"

# PROBES / NET / MODEL_NAME may be overridden from the environment (space-separated for PROBES),
# e.g.  NET=transformer MODEL_NAME=v39_transformer PROBES="lensing clustering 2x2pt combined" sbatch cls_training.sh
# The four default probes fit the node's 4 GPUs in a single wave.
#
# Each PROBES entry is "probe" or "probe:probes_config". The probe names the physical probe and is
# what the run directory is keyed on; the optional probes_config selects which configs/probes/*.yaml
# supplies it (defaults to the probe name). This keeps the run layout stable when a dataset needs a
# variant config -- e.g. on the bta-free v17+ data (msfm extended_nla: False),
#   PROBES="lensing:lensing_nla 2x2pt:2x2pt_nla combined:combined_nla clustering"
# trains from the *_nla configs but still writes to cls/lensing, cls/2x2pt, cls/combined.
read -r -a PROBES <<< "${PROBES:-lensing clustering 2x2pt combined}"

# Cls summary architecture: configs/cls/${NET}/${CLS_CONFIG}.yaml selects network.name (mlp | cls_cnn |
# cls_transformer). See run_cls_training+evaluation.py for the switch. Options: mlp | cnn | transformer.
NET="${NET:-mlp}"
CLS_CONFIG="${CLS_CONFIG:-default}"

LOSS="${LOSS:-vmim}"
# LOSS="vmim_vicreg_inv"
# LOSS="vmim_vicreg_inv_10"
# LOSS="vmim_vicreg"

SCALES="8wl,32gc"
# SCALES="8wl,40gc"
# SCALES="unsmoothed"
# SCALES="lmax_1024"

DATA="default"
MODEL_NAME="${MODEL_NAME:-v1}"

FLOW_CONFIG="$REPOS/multiprobe-simulation-inference/configs/flow/maf.yaml"

# For hard_rebinned: pre-compute the shared Cls cache with full-node resources
# before the per-GPU training workers start.  Runs once for all probes since the
# cache is probe-independent (covers all pairs; probe selection happens at load time).
SCALE_CUT=$(srun -N1 --ntasks-per-node=1 --environment=tensorflow python -c "
import yaml
with open('$REPOS/y3-deep-lss/configs/cls/${NET}/${CLS_CONFIG}.yaml') as f:
    print(yaml.safe_load(f).get('scale_cut', 'soft_pruned'))
")

if [ "$SCALE_CUT" = "hard_rebinned" ]; then
    LOG_PRECACHE="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/lensing/$MODEL_NAME/logs/${SLURM_JOB_ID}_precache"
    mkdir -p "$(dirname "$LOG_PRECACHE")"
    srun -N1 --ntasks-per-node=1 --exclusive --cpus-per-task=288 --mem=450G \
        --environment=tensorflow \
        --output="${LOG_PRECACHE}.log" \
        python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
            --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
            --probes_config="$REPOS/y3-deep-lss/configs/probes/combined.yaml" \
            --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
            --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
            --net_config="$REPOS/y3-deep-lss/configs/cls/${NET}/${CLS_CONFIG}.yaml" \
            --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
            --data_dir="$INPUT" \
            --out_dir="$INPUT" \
            --model_name="precache" \
            --precache_only
fi

for ENTRY in "${PROBES[@]}"; do
    # "probe" or "probe:probes_config" -- the probe keys the run dir, the config supplies the params
    PROBE="${ENTRY%%:*}"
    PROBE_CONFIG="${ENTRY##*:}"

    OUTPUT="$MYSCRATCH/deep_lss/runs/$VERSION/$SUBVERSION/cls/$PROBE"
    LOG="$OUTPUT/$MODEL_NAME/logs/${SLURM_JOB_ID}"
    mkdir -p "$(dirname "$LOG")"

    (
        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --environment=tensorflow \
            --output="${LOG}_training.log" \
            python "$REPOS/y3-deep-lss/deep_lss/apps/run_cls_training+evaluation.py" \
                --msfm_config="$REPOS/multiprobe-simulation-forward-model/configs/$VERSION/$SUBVERSION.yaml" \
                --probes_config="$REPOS/y3-deep-lss/configs/probes/${PROBE_CONFIG}.yaml" \
                --scales_config="$REPOS/y3-deep-lss/configs/scales/${SCALES}.yaml" \
                --loss_config="$REPOS/y3-deep-lss/configs/loss/${LOSS}.yaml" \
                --net_config="$REPOS/y3-deep-lss/configs/cls/${NET}/${CLS_CONFIG}.yaml" \
                --data_config="$REPOS/y3-deep-lss/configs/data/${DATA}.yaml" \
                --data_dir="$INPUT" \
                --out_dir="$OUTPUT" \
                --model_name="$MODEL_NAME" \
                --include_grid \
                --include_des \
                --include_mocks

        srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=72 --mem=110G \
            --uenv=pytorch/v2.9.1:v2 --view=default \
            --output="${LOG}_inference.log" \
            bash -c "source ~/dlss/torch_env/bin/activate && python $REPOS/multiprobe-simulation-inference/msi/apps/run_inference.py \
                --out_dir=\"$OUTPUT\" \
                --model_name=\"$MODEL_NAME\" \
                --flow_config=\"$FLOW_CONFIG\" \
                --n_flows=4 \
                --sample_posterior \
                --include_grid \
                --include_des \
                --include_mocks"
    ) &
done

wait
