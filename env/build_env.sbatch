#!/bin/bash
#SBATCH --job-name=livip3d-build-env
#SBATCH --partition=cpu_short
#SBATCH --account=perun26012552
#SBATCH --qos=perun26012552
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=env/logs/build_env_%j.out
#
# Builds the LiViP3D conda env for Perun's H200 GPUs (sm_90).
# Submit from the repo root:   sbatch env/build_env.sbatch
#
# Stack: python 3.8, torch 2.1.2+cu118, mmcv-full 1.7.2 (source), mmdet 2.28.2,
#        mmseg 0.30.0, mmdet3d 0.17.1 (source + env/mmdet3d-v0.17.1-torch2.patch).
# CUDA extensions are compiled here without a GPU (FORCE_CUDA=1) for ARCH_LIST.
set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
ENV_PREFIX=${ENV_PREFIX:-$HOME/miniconda3/envs/livip3d}
ARCH_LIST=${ARCH_LIST:-"8.9;9.0+PTX"}   # 9.0 = H200 (gpu nodes), 8.9 = L4 (viz nodes)
JOBS=${SLURM_CPUS_PER_TASK:-16}
BUILD_DIR=/tmp/livip3d_build_${SLURM_JOB_ID:-local}   # node-local; the Slurm epilog wipes /tmp

step() { echo; echo "===== [$(date +%T)] $*"; }

if [ -e "$ENV_PREFIX" ]; then
    echo "ERROR: $ENV_PREFIX already exists. Remove it first: conda env remove -p $ENV_PREFIX"
    exit 1
fi
command -v gcc >/dev/null && command -v g++ >/dev/null || { echo "ERROR: gcc/g++ not found on $(hostname)"; exit 1; }
mkdir -p "$BUILD_DIR"
export TMPDIR=$BUILD_DIR PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1

step "host $(hostname), $JOBS cpus, arch list $ARCH_LIST"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

step "conda: python 3.8 + CUDA 11.8 toolkit -> $ENV_PREFIX"
conda create -y -p "$ENV_PREFIX" --override-channels --strict-channel-priority \
    -c nvidia/label/cuda-11.8.0 -c conda-forge \
    python=3.8 pip cuda-toolkit
conda activate "$ENV_PREFIX"
export CUDA_HOME=$CONDA_PREFIX
which python nvcc gcc
nvcc --version | tail -2
gcc --version | head -1

step "pip: torch + binary deps"
pip install -r "$REPO/env/requirements-perun.txt"
pip freeze > "$BUILD_DIR/constraints.txt"    # lock these so later installs cannot move them
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list())"

step "mmcv-full 1.7.2 from source"
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS=$JOBS \
    pip install --no-build-isolation -c "$BUILD_DIR/constraints.txt" mmcv-full==1.7.2

step "mmdet 2.28.2 + mmsegmentation 0.30.0"
pip install -c "$BUILD_DIR/constraints.txt" mmdet==2.28.2 mmsegmentation==0.30.0

step "mmdet3d v0.17.1 (patched) from source"
git -c advice.detachedHead=false clone -q --depth 1 --branch v0.17.1 \
    https://github.com/open-mmlab/mmdetection3d.git "$BUILD_DIR/mmdetection3d"
git -C "$BUILD_DIR/mmdetection3d" apply "$REPO/env/mmdet3d-v0.17.1-torch2.patch"
( cd "$BUILD_DIR/mmdetection3d" && \
  FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS=$JOBS \
  pip install --no-build-isolation --no-deps . )

step "nuscenes-devkit + lyft_dataset_sdk (--no-deps: both hard-require GUI opencv-python)"
pip install --no-deps nuscenes-devkit==1.1.11 lyft_dataset_sdk==0.0.8

pip freeze > "$REPO/env/pip-freeze.txt"

step "sanity check (CPU node: imports + compiled archs only)"
cd "$REPO"
PYTHONPATH="$REPO" python - <<'EOF'
import os, re, glob, subprocess
import torch, torchvision, numpy, numba, mmcv, mmdet, mmseg, mmdet3d
for m in (torch, torchvision, numpy, numba, mmcv, mmdet, mmseg, mmdet3d):
    print(f'{m.__name__:12s} {m.__version__}')
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('mmcv ops compiled with CUDA', get_compiling_cuda_version(), '/', get_compiler_version())
from mmdet3d.ops import Voxelization, iou3d, spconv  # noqa: loads mmdet3d CUDA extensions
import nuscenes, lyft_dataset_sdk  # noqa
import plugin  # noqa: registers LiViP3D modules
exts = glob.glob(os.path.join(os.path.dirname(mmcv.__file__), '_ext*.so')) + \
       glob.glob(os.path.join(os.path.dirname(mmdet3d.__file__), 'ops', 'voxel', 'voxel_layer*.so'))
for so in exts:
    out = subprocess.run(['cuobjdump', '--list-elf', so], capture_output=True, text=True).stdout
    print(os.path.basename(so), sorted(set(re.findall(r'sm_\d+', out))))
print('SANITY OK')
EOF

conda clean -y --tarballs >/dev/null
step "done: $ENV_PREFIX ($(du -sh "$ENV_PREFIX" | cut -f1))"
