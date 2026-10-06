#!/bin/bash
#SBATCH --job-name=livip3d-eval
#SBATCH --partition=gpu_short
#SBATCH --account=perun26012552
#SBATCH --qos=perun26012552
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=env/logs/eval_%j.out
#
# Evaluate several checkpoints on ONE node, one per GPU in parallel:
# detection/tracking (tools/test.py --eval bbox) then trajectory prediction
# (tools/prediction_eval.py). Submit from the repo root.
#
# One node allocation covers the whole sweep -- the account limit is 4 nodes,
# so an array job with one task per epoch would eat all of it.
#
#   WORK_DIR=work_dirs/.../s2_img_guided_b16 sbatch env/eval_epochs.sbatch
#   WORK_DIR=... EPOCHS="3 4 5 6" sbatch env/eval_epochs.sbatch
#   WORK_DIR=... EXTRA="model.score_fusion=geometric" TAG=_geom sbatch env/eval_epochs.sbatch
#
# More epochs than GPUs is fine: they wrap around and share one, which an eval
# at batch 1 has plenty of memory for.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"

WORK_DIR=${WORK_DIR:-work_dirs/perun/non-augmented/s2_img_guided_b16}
# Default to the config mmdet dumped into the work dir, so the model matches
# the weights even if the source config is edited later.
CONFIG=${CONFIG:-$(ls "$WORK_DIR"/*.py 2>/dev/null | head -1)}
EPOCHS=${EPOCHS:-1 2 3 4 5 6}
GPUS=${GPUS:-8}
PRED_INFOS=${PRED_INFOS:-nuscenes_prediction_infos_val.json}
# Extra --cfg-options for inference-only switches, space separated, e.g.
#   EXTRA="model.score_fusion=geometric" TAG=_geom
# TAG keeps the results in their own directory so variants sit side by side.
EXTRA=${EXTRA:-}
TAG=${TAG:-}

if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
    echo "no config found in $WORK_DIR -- pass CONFIG=..." >&2
    exit 1
fi

source env/requirements/activate.sh

echo "host     : $(hostname)"
echo "work dir : $WORK_DIR"
echo "config   : $CONFIG"
echo "epochs   : $EPOCHS"
[ -n "$EXTRA" ] && echo "cfg-opts : $EXTRA"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo
date

# --output_dir is the jsonfile_prefix, so results_nusc.json lands in OUT_DIR.
# --out is left off on purpose: it only adds a large results_nusc.pkl that
# nothing downstream reads.
pids=()
labels=()
gpu=0
for EP in $EPOCHS; do
    CKPT=$WORK_DIR/epoch_${EP}.pth
    if [ ! -f "$CKPT" ]; then
        echo "  epoch $EP: no checkpoint, skipping"
        continue
    fi
    OUT_DIR=$WORK_DIR/eval/epoch_${EP}${TAG}
    mkdir -p "$OUT_DIR"
    echo "  epoch $EP -> GPU $gpu -> $OUT_DIR/eval.log"
    (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES=$gpu
        python tools/test.py "$CONFIG" "$CKPT" --eval bbox --output_dir "$OUT_DIR" \
            ${EXTRA:+--cfg-options $EXTRA}
        python tools/prediction_eval.py \
            --result_path "$OUT_DIR/results_nusc.json" \
            --prediction_infos_path "$PRED_INFOS"
    ) > "$OUT_DIR/eval.log" 2>&1 &
    pids+=($!)
    labels+=("$EP")
    gpu=$(( (gpu + 1) % GPUS ))
done

if [ ${#pids[@]} -eq 0 ]; then
    echo "no checkpoints matched -- nothing to do"
    exit 0
fi

echo
fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  epoch ${labels[$i]}: ok"
    else
        echo "  epoch ${labels[$i]}: FAILED (see $WORK_DIR/eval/epoch_${labels[$i]}${TAG}/eval.log)"
        fail=1
    fi
done
date

echo
echo "===== prediction metrics ====="
for EP in $EPOCHS; do
    f=$WORK_DIR/eval/epoch_${EP}${TAG}/prediction_metrics.json
    [ -f "$f" ] && printf '  epoch %-3s %s\n' "$EP" "$(tr -d '\n ' < "$f")"
done

exit $fail
