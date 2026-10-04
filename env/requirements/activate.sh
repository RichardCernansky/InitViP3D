# Activate the LiViP3D env inside a Slurm job:
#   source env/requirements/activate.sh
# (conda env `vip`, built by env/requirements/build_env.sbatch)
#
# Find the repo root by walking up to the directory that holds plugin/, rather
# than assuming how deep this file sits -- it has moved once already.
_LIVIP3D_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && \
                while [ "$PWD" != / ] && [ ! -d "$PWD/plugin" ]; do cd ..; done; pwd)
if [ ! -d "$_LIVIP3D_REPO/plugin" ]; then
    echo "activate.sh: cannot find the repo root (no plugin/ above $(dirname "${BASH_SOURCE[0]}"))" >&2
    return 1 2>/dev/null || exit 1
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vip
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH="$_LIVIP3D_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset _LIVIP3D_REPO
