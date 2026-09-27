# Activate the LiViP3D env inside a Slurm job:   source env/activate.sh
# (conda env `vip`, built by env/build_env.sbatch)
_LIVIP3D_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vip
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH="$_LIVIP3D_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset _LIVIP3D_REPO
