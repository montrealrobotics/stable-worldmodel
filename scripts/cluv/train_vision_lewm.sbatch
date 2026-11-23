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
# Optional eval as checkpoints land (flags before the hydra overrides):
#   --eval-epochs=5,10,15   epochs whose weights_epoch_N.pt to hand to the hook
#   --eval-hook=<script>    run as `bash <script> <run dir> <epoch>` (with SLURM_* unset except SLURM_CONF, so it can
#                           sbatch) once that file is complete (size unchanged for 30 s); a marker
#                           .eval_submitted_<epoch> in the run dir stops repeats, also across resumes
# --dataset names an archive in $SCRATCH/le-wm/cluv_data (filled by `cluv sync`).
# The run (weights, config, hydra, offline wandb) lands in
# $SCRATCH/le-wm/checkpoints/<run-name>_<job id>.

set -euo pipefail

DATASET_ARCHIVE=""
RUN_NAME="lewm_vision"
EVAL_EPOCHS=""
EVAL_HOOK=""
CONFIG_NAME="${CONFIG_NAME:-lewm}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config=*) CONFIG_NAME="${1#*=}"; shift ;;
    --dataset=*) DATASET_ARCHIVE="${1#*=}"; shift ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    --eval-epochs=*) EVAL_EPOCHS="${1#*=}"; EVAL_EPOCHS="${EVAL_EPOCHS//,/ }"; shift ;;
    --eval-hook=*) EVAL_HOOK="${1#*=}"; shift ;;
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
RUN_DIR="$STABLEWM_HOME/checkpoints/$RUN_NAME"

# Hand each listed epoch's weights to $EVAL_HOOK once, as soon as the file is complete
# (torch.save is not atomic: wait until its size is unchanged for 30 s).
submit_ready_evals() {
  local e f size
  for e in $EVAL_EPOCHS; do
    f="$RUN_DIR/weights_epoch_$e.pt"
    [[ -e "$RUN_DIR/.eval_submitted_$e" || ! -s "$f" ]] && continue
    size=$(stat -c %s "$f") || continue; sleep "${EVAL_SETTLE_SECONDS:-30}"
    [[ "$(stat -c %s "$f" 2>/dev/null)" == "$size" ]] || continue
    if env $(compgen -e | grep '^SLURM_' | grep -v '^SLURM_CONF$' | sed 's/^/-u /') bash "$EVAL_HOOK" "$RUN_DIR" "$e"; then
      touch "$RUN_DIR/.eval_submitted_$e"
      echo "[eval-on-save] submitted epoch $e"
    else
      echo "[eval-on-save] hook failed for epoch $e; retrying later" >&2
    fi
  done
}
WATCHER=""
if [[ -n "$EVAL_EPOCHS" ]]; then
  [[ -f "${EVAL_HOOK:?--eval-epochs needs --eval-hook}" ]] || { echo "no eval hook $EVAL_HOOK" >&2; exit 1; }
  ( while true; do submit_ready_evals; sleep "${EVAL_POLL_SECONDS:-120}"; done ) &
  WATCHER=$!
fi
# Compute nodes may be offline (UV_OFFLINE=1): the extras must already be in
# the uv cache, e.g. from `cluv run <cluster> uv sync --extra train --extra data`.
STATUS=0
uv run --extra train --extra data python scripts/train/lewm.py \
  --config-name="$CONFIG_NAME" \
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
  "$@" || STATUS=$?

if [[ -n "$WATCHER" ]]; then  # last pass for the epochs saved at the end
  kill "$WATCHER" 2>/dev/null || true
  submit_ready_evals
fi
exit "$STATUS"
