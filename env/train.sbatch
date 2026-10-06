#!/bin/bash
#SBATCH --job-name=livip3d-s1
#SBATCH --partition=gpu_short
#SBATCH --account=perun26012552
#SBATCH --qos=perun26012552
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=env/logs/train_s1_%j.out
#
# LiViP3D stage 1: LiDAR-only, 20 epochs, 8 GPUs x 8 clips = effective batch 64.
# Submit from the repo root:   sbatch env/train_s1.sbatch
#
# Only the image backbone/neck are pretrained (DETR3D, frozen). The LiDAR
# backbone, neck and heatmap head train from scratch at the base lr, following
# TransFusion's first stage.
#
# Everything about the run -- batch size, lr, warmup, epochs, checkpointing --
# lives in the config; this script only selects it and launches. GPUS must
# match --gres above. To continue a killed run:
#   RESUME=work_dirs/.../latest.pth sbatch env/train_s1.sbatch
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"

CONFIG=${CONFIG:-plugin/vip3d/configs/perun/non-augmented/livip3d_resnet50_joint.py}
WORK_DIR=${WORK_DIR:-work_dirs/perun/non-augmented/joint_img_guided_necktrain_b16}
GPUS=${GPUS:-8}
BATCH=${BATCH:-2}              # clips per GPU; effective batch = GPUS * BATCH
KEEP_CKPTS=${KEEP_CKPTS:--1}   # -1 = keep every epoch (20 x 231 MB on a 500 GB home)
RESUME=${RESUME:-}             # checkpoint to continue from; empty = start fresh
# Extra --cfg-options, space separated, for sweeps that vary one knob:
#   BATCH=1 EXTRA="optimizer.lr=3.5e-4 lr_config.warmup_iters=700" sbatch ...
EXTRA=${EXTRA:-}

source env/requirements/activate.sh

# dist_train.sh defaults to 29401; randomise so two jobs on one node don't clash
export PORT=${PORT:-$((29400 + RANDOM % 500))}

# Everything else -- lr, warmup, epochs, schedule -- comes from the config.
OPTS=(data.samples_per_gpu="$BATCH"
      checkpoint_config.max_keep_ckpts="$KEEP_CKPTS")
# shellcheck disable=SC2206  -- word splitting is how EXTRA becomes several options
[ -n "$EXTRA" ] && OPTS+=($EXTRA)

ARGS=(--work-dir "$WORK_DIR" --cfg-options "${OPTS[@]}")
if [ -n "$RESUME" ]; then
    ARGS+=(--resume-from "$RESUME")
fi

echo "host      : $(hostname)"
echo "config    : $CONFIG"
echo "work dir  : $WORK_DIR"
echo "gpus      : $GPUS x $BATCH clips = effective batch $((GPUS * BATCH))"
echo "port      : $PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo

# No srun: dist_train.sh already spawns one process per GPU via
# torch.distributed.launch, and nesting the two confuses the rank assignment.
bash tools/dist_train.sh "$CONFIG" "$GPUS" "${ARGS[@]}"
