#!/usr/bin/env bash
#SBATCH --job-name=onpolicy_rounds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=onpolicy_rounds_%j.out

# On-policy post-training in one job: ROUNDS rounds of
#   collect (scripts/train/collect_onpolicy.py: the latest model's search plans
#   run in the sim) -> train TOTAL_STEPS/ROUNDS steps with MSE + SIGReg on the
#   dataset plus every round's rollouts so far (config lewm_onpolicy).
# Submit from the repo root; arguments are lewm.py overrides, e.g.
#   INIT_FROM=$SCRATCH/le-wm/checkpoints/noisy_lewm_res_nobn_w0.009_11311542/weights_epoch_20.pt \
#   NAME=noisy_lewm_res_nobn_onpolicy_k4 VENV_TAR=$SCRATCH/le-wm/swm_venv.tar \
#   sbatch scripts/cluv/onpolicy_rounds.sbatch loss.sigreg.weight=0.009
# Rounds are named without the job id ($SCRATCH/le-wm/onpolicy/<NAME>_r<r>.lance,
# checkpoints/<NAME>_r<r>/), and finished rounds are skipped, so resubmitting
# the same command resumes.
set -euo pipefail

INIT_FROM="${INIT_FROM:?Set INIT_FROM to the starting weights .pt}"
NAME="${NAME:?Set NAME to the run name}"
ROUNDS="${ROUNDS:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-10000}"
NUM_CLIPS="${NUM_CLIPS:-20000}" # per round; x 14 plans (4 noisy + 10 elites) rollouts
DATASET_SOURCE="${DATASET_SOURCE:-$SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance.tar.gz}"
STEPS=$((TOTAL_STEPS / ROUNDS))

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

mkdir -p "$SLURM_TMPDIR/datasets" "$SCRATCH/le-wm/onpolicy"
tar -xf "$DATASET_SOURCE" -C "$SLURM_TMPDIR/datasets"
ARCHIVE_NAME="$(basename "$DATASET_SOURCE")"
DATASET_NAME="${ARCHIVE_NAME%.tar.gz}"

export LOCAL_DATASET_DIR="$SLURM_TMPDIR"
export WANDB_DIR="$SCRATCH"
export HF_HOME="$SCRATCH/huggingface"
export TORCH_HOME="$SCRATCH/torch_cache"
export SPT_CACHE_DIR="$SCRATCH/le-wm"
export STABLEWM_HOME="$SCRATCH/le-wm"
# ManiSkill's Vulkan runtime, as in eval_vision_lewm.sbatch.
MANI_EXPS_DIR="${MANI_EXPS_DIR:-$HOME/git/mani-exps}"
export LD_LIBRARY_PATH="$MANI_EXPS_DIR/.local-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

model="$INIT_FROM"
collected=""
for r in $(seq 1 "$ROUNDS"); do
  run="${NAME}_r$r"
  data="$SCRATCH/le-wm/onpolicy/$run.lance"
  weights="$STABLEWM_HOME/checkpoints/$run/weights_step_$STEPS.pt"

  if [[ ! -e "$data" ]]; then
    # Collect under a temporary name so an interrupted round is never reused.
    tmp="$SCRATCH/le-wm/onpolicy/${run}_tmp.lance"
    rm -rf "$tmp" "${tmp%.lance}.collection.json"
    python scripts/train/collect_onpolicy.py \
      --checkpoint "$model" --dataset "$DATASET_NAME" --cache-dir "$SLURM_TMPDIR" \
      --output "$tmp" --num-clips "$NUM_CLIPS" --seed "$r"
    mv "${tmp%.lance}.collection.json" "${data%.lance}.collection.json"
    mv "$tmp" "$data"
  fi
  collected="${collected:+$collected,}$data"

  if [[ ! -e "$weights" ]]; then
    python scripts/train/lewm.py \
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
  fi
  model="$weights"
done
echo "done: $model"
