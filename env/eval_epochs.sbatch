#!/bin/bash
#SBATCH --job-name=livip3d-eval
#SBATCH --partition=gpu_short
#SBATCH --account=perun26012552
#SBATCH --qos=perun26012552
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=3-8%6
#SBATCH --output=env/logs/eval_%A_%a.out
#
# Evaluate one checkpoint per array task: detection/tracking (tools/test.py
# --eval bbox) then trajectory prediction (tools/prediction_eval.py).
# Submit from the repo root:   sbatch env/eval_epochs.sbatch
#
# The array index IS the epoch number, so 10-20 covers epoch_10..epoch_20 and
# %6 caps how many run at once. Override the range on the command line:
#   sbatch --array=20 env/eval_epochs.sbatch            # just the last one
#   WORK_DIR=... sbatch --array=1-20%8 env/eval_epochs.sbatch
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"

CONFIG=${CONFIG:-work_dirs/perun/non-augmented/joint_img_guided_b16/livip3d_resnet50_lidar_img_guided.py}
WORK_DIR=${WORK_DIR:-work_dirs/perun/non-augmented/joint_img_guided_b16}
# The config dumped into the work dir, so eval matches exactly what was trained
# (including any --cfg-options the training job passed).
PRED_INFOS=${PRED_INFOS:-nuscenes_prediction_infos_val.json}
# Extra --cfg-options for inference-only switches, space separated, e.g.
#   EXTRA="model.score_fusion=geometric" TAG=_geom sbatch --array=6 ...
# TAG keeps the results in their own directory so runs can be compared.
EXTRA=${EXTRA:-}
TAG=${TAG:-}

EPOCH=${SLURM_ARRAY_TASK_ID:?run me as an array job, e.g. sbatch --array=20 ...}
CKPT=$WORK_DIR/epoch_${EPOCH}.pth
OUT_DIR=$WORK_DIR/eval/epoch_${EPOCH}${TAG}

if [ ! -f "$CKPT" ]; then
    echo "no checkpoint at $CKPT -- nothing to do"
    exit 0
fi

source env/requirements/activate.sh
mkdir -p "$OUT_DIR"

echo "host     : $(hostname)"
echo "config   : $CONFIG"
echo "epoch    : $EPOCH"
echo "ckpt     : $CKPT"
echo "out      : $OUT_DIR"
echo

# --output_dir is the jsonfile_prefix, so results_nusc.json lands in OUT_DIR.
# --out is left off on purpose: it only adds a large results_nusc.pkl that
# nothing downstream reads.
echo "===== [epoch $EPOCH] detection / tracking ====="
python tools/test.py "$CONFIG" "$CKPT" --eval bbox --output_dir "$OUT_DIR" \
    ${EXTRA:+--cfg-options $EXTRA}

echo
echo "===== [epoch $EPOCH] trajectory prediction ====="
python tools/prediction_eval.py \
    --result_path "$OUT_DIR/results_nusc.json" \
    --prediction_infos_path "$PRED_INFOS"

echo
echo "===== [epoch $EPOCH] done -> $OUT_DIR ====="
