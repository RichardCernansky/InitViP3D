# Activate the LiViP3D env inside a Slurm job:   source env/activate.sh
# (built by env/build_env.sbatch)
_LIVIP3D_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${LIVIP3D_ENV:-$HOME/miniconda3/envs/livip3d}"
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH="$_LIVIP3D_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset _LIVIP3D_REPO
