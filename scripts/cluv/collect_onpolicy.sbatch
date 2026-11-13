#!/usr/bin/env bash
#SBATCH --job-name=collect_onpolicy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --output=collect_onpolicy_%j.out

# One round of on-policy rollouts (scripts/train/collect_onpolicy.py). Submit
# from the repo root; arguments go to the collector, e.g. a restore check:
#   CHECKPOINT=$SCRATCH/le-wm/checkpoints/noisy_lewm_res_nobn_w0.009_11311542/weights_epoch_20.pt \
#   OUTPUT=$SCRATCH/le-wm/onpolicy/verify.lance \
#   sbatch scripts/cluv/collect_onpolicy.sbatch --verify --num-clips 64 --clips-per-batch 8
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the weights .pt (config.yaml beside it)}"
OUTPUT="${OUTPUT:?Set OUTPUT to the rollout .lance path}"
DATASET_SOURCE="${DATASET_SOURCE:-$SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance.tar.gz}"

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

mkdir -p "$SLURM_TMPDIR/datasets" "$(dirname "$OUTPUT")"
tar -xf "$DATASET_SOURCE" -C "$SLURM_TMPDIR/datasets"
ARCHIVE_NAME="$(basename "$DATASET_SOURCE")"

export HF_HOME="$SCRATCH/huggingface"
export TORCH_HOME="$SCRATCH/torch_cache"
export STABLEWM_HOME="$SCRATCH/le-wm"
# ManiSkill's Vulkan runtime, as in eval_vision_lewm.sbatch.
MANI_EXPS_DIR="${MANI_EXPS_DIR:-$HOME/git/mani-exps}"
export LD_LIBRARY_PATH="$MANI_EXPS_DIR/.local-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

python scripts/train/collect_onpolicy.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "${ARCHIVE_NAME%.tar.gz}" \
  --cache-dir "$SLURM_TMPDIR" \
  --output "$OUTPUT" \
  "$@"
