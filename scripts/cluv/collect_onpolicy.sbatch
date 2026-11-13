#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=6:00:00

# One on-policy collection round (scripts/train/collect_onpolicy.py) for cluv:
#   cluv submit mila scripts/cluv/collect_onpolicy.sbatch --gres=gpu:l40s:1 -- \
#     --dataset=overhead_rgb_noisy_replay_100k.lance.tar.gz \
#     --checkpoint=noisy_lewm_res_nobn_w0.009_11311542/weights_epoch_20.pt \
#     --output-name=nobn_onpolicy_r1 --num-clips 20000 --seed 1
# --dataset names an archive in $SCRATCH/le-wm/cluv_data.
# --checkpoint is absolute or relative to $SCRATCH/le-wm/checkpoints.
# Rollouts land in $SCRATCH/le-wm/onpolicy/<output-name>.lance.

set -euo pipefail

DATASET_ARCHIVE=""
CHECKPOINT=""
OUTPUT_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASET_ARCHIVE="${1#*=}"; shift ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --output-name=*) OUTPUT_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done
DATASET_ARCHIVE="${DATASET_ARCHIVE:?Pass --dataset=<archive in \$SCRATCH/le-wm/cluv_data>}"
CHECKPOINT="${CHECKPOINT:?Pass --checkpoint=<weights .pt>}"
OUTPUT_NAME="${OUTPUT_NAME:?Pass --output-name=<name>}"
[[ "$CHECKPOINT" == /* ]] || CHECKPOINT="$SCRATCH/le-wm/checkpoints/$CHECKPOINT"
[[ -e "$CHECKPOINT" ]] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 1; }

mkdir -p "$SLURM_TMPDIR/datasets" "$SCRATCH/le-wm/onpolicy"
tar -xf "$SCRATCH/le-wm/cluv_data/$DATASET_ARCHIVE" -C "$SLURM_TMPDIR/datasets"

export HF_HOME="$SCRATCH/huggingface"
export TORCH_HOME="$SCRATCH/torch_cache"
export STABLEWM_HOME="$SCRATCH/le-wm"
# ManiSkill renders with Vulkan; reuse the mani-exps runtime when present.
VULKAN_LIBS="${MANI_EXPS_DIR:-$HOME/git/mani-exps}/.local-vulkan/usr/lib/x86_64-linux-gnu"
[[ -d "$VULKAN_LIBS" ]] && export LD_LIBRARY_PATH="$VULKAN_LIBS:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

uv run --extra train --extra data --extra maniskill python scripts/train/collect_onpolicy.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "${DATASET_ARCHIVE%.tar.gz}" \
  --cache-dir "$SLURM_TMPDIR" \
  --output "$SCRATCH/le-wm/onpolicy/$OUTPUT_NAME.lance" \
  "$@"
