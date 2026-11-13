#!/usr/bin/env bash
#SBATCH --job-name=replay_determinism
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:00:00
#SBATCH --output=replay_determinism_%j.out

# Replay a recorded batch of the noisy-replay dataset in the GPU sim
# (scripts/data/check_replay_determinism.py). Submit from the repo root;
# arguments go to the script, e.g. --batch 3 --restore-step 20:
#   VENV_TAR=$SCRATCH/le-wm/swm_venv.tar sbatch scripts/cluv/check_replay_determinism.sbatch
set -euo pipefail

DATASET_SOURCE="${DATASET_SOURCE:-$SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance.tar.gz}"
ARCHIVE_NAME="$(basename "$DATASET_SOURCE")"
COLLECTION_JSON="${COLLECTION_JSON:-$(dirname "$DATASET_SOURCE")/${ARCHIVE_NAME%.lance.tar.gz}.collection.json}"

module load cuda/12.6.0/cudnn/9.3
module load anaconda/3
VENV_PATH="${VENV_PATH:-$SCRATCH/le-wm/.swm}"
# VENV_TAR: a tar of the venv holding a top-level .swm/, extracted to local disk.
if [[ -n "${VENV_TAR:-}" ]]; then
  tar -xf "$VENV_TAR" -C "$SLURM_TMPDIR"
  VENV_PATH="$SLURM_TMPDIR/.swm"
fi
source "$VENV_PATH/bin/activate"
# activate hardcodes the original venv path; point PATH at the copy in use.
export VIRTUAL_ENV="$VENV_PATH" PATH="$VENV_PATH/bin:$PATH"

mkdir -p "$SLURM_TMPDIR/datasets"
tar -xf "$DATASET_SOURCE" -C "$SLURM_TMPDIR/datasets"

export STABLEWM_HOME="$SCRATCH/le-wm"
# ManiSkill's Vulkan runtime, as in eval_vision_lewm.sbatch.
MANI_EXPS_DIR="${MANI_EXPS_DIR:-$HOME/git/mani-exps}"
export LD_LIBRARY_PATH="$MANI_EXPS_DIR/.local-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

run() {
  python scripts/data/check_replay_determinism.py \
    --dataset "${ARCHIVE_NAME%.tar.gz}" \
    --cache-dir "$SLURM_TMPDIR" \
    --collection-json "$COLLECTION_JSON" \
    "$@"
}
# FRESH_PAIR=1: run twice in separate processes (each a fresh sim); the second
# compares its A1 replay with the first's.
if [[ -n "${FRESH_PAIR:-}" ]]; then
  run --save "$SLURM_TMPDIR/a1.pt" "$@"
  echo "=== second process ==="
  run --compare "$SLURM_TMPDIR/a1.pt" "$@"
else
  run "$@"
fi
