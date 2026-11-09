#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# Cluster-agnostic copy of slurm_scripts/train_res_lewm_search.sbatch for cluv:
#   cluv submit rorqual scripts/cluv/train_res_lewm_search.sbatch --gpus=h100:1 -- \
#     --dataset=overhead_rgb_noisy_replay_100k.lance.tar.gz \
#     --init-from=noisy_lewm_res_nobn_w0.009_11311542/weights_epoch_20.pt \
#     --run-name=noisy_lewm_res_nobn_search_h5 <hydra overrides...>
# --dataset names an archive in $SCRATCH/le-wm/cluv_data.
# --init-from is absolute or relative to $SCRATCH/le-wm/checkpoints.
# The run lands in $SCRATCH/le-wm/checkpoints/<run-name>_<job id>.

set -euo pipefail

DATASET_ARCHIVE=""
INIT_FROM=""
RUN_NAME="res_lewm_search"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASET_ARCHIVE="${1#*=}"; shift ;;
    --init-from=*) INIT_FROM="${1#*=}"; shift ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done
DATASET_ARCHIVE="${DATASET_ARCHIVE:?Pass --dataset=<archive in \$SCRATCH/le-wm/cluv_data>}"
INIT_FROM="${INIT_FROM:?Pass --init-from=<weights .pt, absolute or under \$SCRATCH/le-wm/checkpoints>}"
[[ "$INIT_FROM" == /* ]] || INIT_FROM="$SCRATCH/le-wm/checkpoints/$INIT_FROM"
[[ -e "$INIT_FROM" ]] || { echo "init checkpoint not found: $INIT_FROM" >&2; exit 1; }
DATASET_SOURCE="$SCRATCH/le-wm/cluv_data/$DATASET_ARCHIVE"

mkdir -p "$SLURM_TMPDIR/datasets"
tar -xf "$DATASET_SOURCE" -C "$SLURM_TMPDIR/datasets"
DATASET_NAME="${DATASET_ARCHIVE%.tar.gz}"

export LOCAL_DATASET_DIR="$SLURM_TMPDIR"
export WANDB_DIR="$SCRATCH"
export HF_HOME="$SCRATCH/huggingface"
export TORCH_HOME="$SCRATCH/torch_cache"
export SPT_CACHE_DIR="$SCRATCH/le-wm"
export STABLEWM_HOME="$SCRATCH/le-wm"

RUN_NAME="${RUN_NAME}_${SLURM_JOB_ID}"
# img_size and history_size must match the pretrained run. Compute nodes may
# be offline (UV_OFFLINE=1): the extras must already be in the uv cache.
uv run --extra train --extra data python scripts/train/lewm.py \
  --config-name=lewm_search \
  init_from="$INIT_FROM" \
  data=maniskill_vision \
  data.dataset.name="$DATASET_NAME" \
  img_size=128 \
  wm.history_size=1 \
  output_model_name="$RUN_NAME" \
  subdir="$RUN_NAME" \
  loader.num_workers=8 \
  loader.batch_size=256 \
  wandb.enabled=true \
  "$@"
