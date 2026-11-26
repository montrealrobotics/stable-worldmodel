#!/usr/bin/env bash
#SBATCH --job-name=compare_pusht_cameras
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

# Re-render a collected dataset's states in stock PushT-v1 and OverheadPushT-v1
# (scripts/data/compare_pusht_cameras.py). Rendering needs a GPU (Vulkan).
#   cluv submit mila scripts/cluv/compare_pusht_cameras.sbatch -- \
#     --dataset /network/scratch/o/ozgur.aslan/le-wm/datasets/basecam_smoke.lance
# PNGs go to $SCRATCH/le-wm/camera_compare/<job id>/ unless --out-dir is given.
set -euo pipefail

module load cuda/12.6.0/cudnn/9.3
module load anaconda/3
VENV_PATH="${VENV_PATH:-$SCRATCH/le-wm/.swm}"
source "$VENV_PATH/bin/activate"
export VIRTUAL_ENV="$VENV_PATH" PATH="$VENV_PATH/bin:$PATH"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MANI_EXPS_DIR="${MANI_EXPS_DIR:-$HOME/git/mani-exps}"
export LD_LIBRARY_PATH="$MANI_EXPS_DIR/.local-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

python -u scripts/data/compare_pusht_cameras.py \
  --out-dir "$SCRATCH/le-wm/camera_compare/${SLURM_JOB_ID}" \
  "$@"
