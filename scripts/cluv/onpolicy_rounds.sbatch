#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# On-policy post-training in one job: K rounds of
#   collect (scripts/train/collect_onpolicy.py, the latest model's search plans
#   executed in the sim) -> train TOTAL/K steps with MSE + SIGReg on the
#   dataset plus every round's rollouts so far (config lewm_onpolicy).
#
#   cluv submit mila scripts/cluv/onpolicy_rounds.sbatch --gpus=l40s:1 -- \
#     --dataset=overhead_rgb_noisy_replay_100k.lance.tar.gz \
#     --init-from=noisy_lewm_res_nobn_w0.009_11311542/weights_epoch_20.pt \
#     --name=noisy_lewm_res_nobn_onpolicy_k4 --rounds=4 \
#     loss.sigreg.weight=0.009 <more lewm.py overrides...>
#
# Outputs (named without the job id, so a resubmission resumes):
#   rollouts    $SCRATCH/le-wm/onpolicy/<name>_r<r>.lance
#   checkpoints $SCRATCH/le-wm/checkpoints/<name>_r<r>/weights_step_<TOTAL/K>.pt
# A round whose output already exists is skipped; partial outputs of an
# interrupted round are moved aside and the round is redone.

set -euo pipefail

DATASET_ARCHIVE=""
INIT_FROM=""
NAME=""
ROUNDS=4
TOTAL_STEPS=10000
NUM_CLIPS=20000 # per round; x 14 plans (4 noisy + 10 elites) rollouts
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASET_ARCHIVE="${1#*=}"; shift ;;
    --init-from=*) INIT_FROM="${1#*=}"; shift ;;
    --name=*) NAME="${1#*=}"; shift ;;
    --rounds=*) ROUNDS="${1#*=}"; shift ;;
    --total-steps=*) TOTAL_STEPS="${1#*=}"; shift ;;
    --num-clips=*) NUM_CLIPS="${1#*=}"; shift ;;
    *) break ;;
  esac
done
DATASET_ARCHIVE="${DATASET_ARCHIVE:?Pass --dataset=<archive in \$SCRATCH/le-wm/cluv_data>}"
INIT_FROM="${INIT_FROM:?Pass --init-from=<weights .pt>}"
NAME="${NAME:?Pass --name=<run name>}"
[[ "$INIT_FROM" == /* ]] || INIT_FROM="$SCRATCH/le-wm/checkpoints/$INIT_FROM"
[[ -e "$INIT_FROM" ]] || { echo "init checkpoint not found: $INIT_FROM" >&2; exit 1; }
STEPS=$((TOTAL_STEPS / ROUNDS))
CKPT_ROOT="$SCRATCH/le-wm/checkpoints"
DATA_ROOT="$SCRATCH/le-wm/onpolicy"

mkdir -p "$SLURM_TMPDIR/datasets" "$DATA_ROOT"
tar -xf "$SCRATCH/le-wm/cluv_data/$DATASET_ARCHIVE" -C "$SLURM_TMPDIR/datasets"
DATASET_NAME="${DATASET_ARCHIVE%.tar.gz}"

export LOCAL_DATASET_DIR="$SLURM_TMPDIR"
export WANDB_DIR="$SCRATCH"
export HF_HOME="$SCRATCH/huggingface"
export TORCH_HOME="$SCRATCH/torch_cache"
export SPT_CACHE_DIR="$SCRATCH/le-wm"
export STABLEWM_HOME="$SCRATCH/le-wm"
# ManiSkill renders with Vulkan; reuse the mani-exps runtime when present.
VULKAN_LIBS="${MANI_EXPS_DIR:-$HOME/git/mani-exps}/.local-vulkan/usr/lib/x86_64-linux-gnu"
[[ -d "$VULKAN_LIBS" ]] && export LD_LIBRARY_PATH="$VULKAN_LIBS:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

UV=(uv run --extra train --extra data --extra maniskill)
stamp() { date +%Y%m%d-%H%M%S; }

model="$INIT_FROM"
collected=""
for r in $(seq 1 "$ROUNDS"); do
  data="$DATA_ROOT/${NAME}_r$r.lance"
  run="${NAME}_r$r"
  weights="$CKPT_ROOT/$run/weights_step_$STEPS.pt"

  if [[ -e "$data" ]]; then
    echo "round $r: rollouts exist, skipping collection"
  else
    tmp="$DATA_ROOT/${NAME}_r${r}_tmp.lance"
    [[ -e "$tmp" ]] && mv "$tmp" "$DATA_ROOT/${NAME}_r${r}_partial_$(stamp).lance"
    echo "round $r: collecting with $model"
    "${UV[@]}" python scripts/train/collect_onpolicy.py \
      --checkpoint "$model" --dataset "$DATASET_NAME" --cache-dir "$SLURM_TMPDIR" \
      --output "$tmp" --num-clips "$NUM_CLIPS" --seed "$r"
    mv "${tmp%.lance}.collection.json" "${data%.lance}.collection.json"
    mv "$tmp" "$data"
  fi
  collected="${collected:+$collected,}$data"

  if [[ -e "$weights" ]]; then
    echo "round $r: weights exist, skipping training"
  else
    [[ -e "$CKPT_ROOT/$run" ]] && mv "$CKPT_ROOT/$run" "$CKPT_ROOT/${run}_partial_$(stamp)"
    echo "round $r: training $STEPS steps on dataset + [$collected]"
    "${UV[@]}" python scripts/train/lewm.py \
      --config-name=lewm_onpolicy \
      init_from="$model" \
      data=maniskill_vision \
      data.dataset.name="$DATASET_NAME" \
      img_size=128 \
      wm.history_size=1 \
      output_model_name="$run" \
      subdir="$run" \
      loader.num_workers=8 \
      loader.batch_size=256 \
      wandb.enabled=true \
      "onpolicy.datasets=[$collected]" \
      trainer.max_steps="$STEPS" \
      save_step_interval="$STEPS" \
      "$@"
    [[ -e "$weights" ]] || { echo "round $r: $weights missing after training" >&2; exit 1; }
  fi
  model="$weights"
done
echo "done: $model"
