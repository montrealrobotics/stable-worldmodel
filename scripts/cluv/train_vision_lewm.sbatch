#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# Cluster-agnostic copy of slurm_scripts/train_vision_lewm.sbatch for cluv:
#   cluv submit rorqual scripts/cluv/train_vision_lewm.sbatch --gpus=h100:1 -- \
#     --dataset=overhead_rgb_noisy_replay_hold5_20k.lance.tar.gz \
#     --run-name=noisyhold5_20k_lewm_res <hydra overrides...>
# --dataset names an archive in $SCRATCH/le-wm/cluv_data (filled by `cluv sync`).
# The run (weights, config, hydra, offline wandb) lands in
# $SCRATCH/le-wm/checkpoints/<run-name>_<job id>.

set -euo pipefail

DATASET_ARCHIVE=""
RUN_NAME="lewm_vision"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASET_ARCHIVE="${1#*=}"; shift ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done
DATASET_ARCHIVE="${DATASET_ARCHIVE:?Pass --dataset=<archive in \$SCRATCH/le-wm/cluv_data>}"
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
# Compute nodes may be offline (UV_OFFLINE=1): the extras must already be in
# the uv cache, e.g. from `cluv run <cluster> uv sync --extra train --extra data`.
uv run --extra train --extra data python scripts/train/lewm.py \
  --config-name="${CONFIG_NAME:-lewm}" \
  data=maniskill_vision \
  data.dataset.name="$DATASET_NAME" \
  output_model_name="$RUN_NAME" \
  subdir="$RUN_NAME" \
  ckpt_path="${CKPT_PATH:-null}" \
  img_size=128 \
  patch_size=16 \
  loader.num_workers=10 \
  loader.prefetch_factor=2 \
  loader.batch_size=1024 \
  trainer.max_epochs=100 \
  trainer.max_steps=-1 \
  wandb.enabled=true \
  wm.history_size=1 \
  optimizer.lr=1e-4 \
  "$@"
