#!/usr/bin/env bash
#SBATCH --job-name=collect_maniskill_pusht
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --output=collect_maniskill_pusht_%j.out

# PPO expert rollouts on OverheadPushT-v1 (scripts/data/collect_maniskill_pusht.py),
# with the flags of the overhead_rgb_* datasets collected from mani-exps. The PPO
# run has no training_config.json, so policy/env flags are explicit here.
# The policy always acts on overhead_camera; CAMERA picks the saved pixels
# (default base_camera = ManiSkill's stock PushT view). Extra arguments go to
# the collector and override the ones below, e.g. the noisy-replay 100k recipe:
#   cluv submit mila scripts/cluv/collect_maniskill_pusht.sbatch -- \
#     --output-path $SCRATCH/le-wm/datasets/basecam_rgb_noisy_replay_100k.lance \
#     --num-episodes 21800 --noisy-replays 4 --noise-std 0.1 --noise-hold-steps 1 --seed 40000
# The dataset (and its _episodes table) is archived to <output>.tar.gz at the end.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/network/scratch/o/ozgur.aslan/maniskill_pusht_ppo/runs/pusht_overhead_rgb_spatial_softmax_learned_fs1_ar1_10754956/final_ckpt.pt}"
CAMERA="${CAMERA:-base_camera}"
ARCHIVE="${ARCHIVE:-1}"

module load cuda/12.6.0/cudnn/9.3
module load anaconda/3
# .swm holds ManiSkill 3.0.1 and an editable install of this checkout.
VENV_PATH="${VENV_PATH:-$SCRATCH/le-wm/.swm}"
if [[ -n "${VENV_TAR:-}" ]]; then
  tar -xf "$VENV_TAR" -C "$SLURM_TMPDIR"
  VENV_PATH="$SLURM_TMPDIR/.swm"
fi
source "$VENV_PATH/bin/activate"
export VIRTUAL_ENV="$VENV_PATH" PATH="$VENV_PATH/bin:$PATH"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
# ManiSkill's Vulkan runtime, as in the mani-exps collection script.
MANI_EXPS_DIR="${MANI_EXPS_DIR:-$HOME/git/mani-exps}"
export LD_LIBRARY_PATH="$MANI_EXPS_DIR/.local-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

python -u scripts/data/collect_maniskill_pusht.py \
  --checkpoint "$CHECKPOINT" \
  --policy-type rgb --encoder spatial_softmax \
  --std-mode learned --initial-std 0.3 --min-std 0.1 --max-std 0.5 \
  --env-id OverheadPushT-v1 --control-mode pd_ee_delta_xy \
  --reward-mode normalized_dense \
  --camera-name "$CAMERA" --image-size 128 \
  --max-steps 100 --num-envs 64 --write-batch-size 64 \
  --reconfiguration-freq 0 --policy-mode deterministic \
  "$@"

if [[ "$ARCHIVE" == 1 ]]; then
  OUTPUT=""
  prev=""
  for arg in "$@"; do
    [[ "$prev" == --output-path ]] && OUTPUT="$arg"
    [[ "$arg" == --output-path=* ]] && OUTPUT="${arg#*=}"
    prev="$arg"
  done
  if [[ -n "$OUTPUT" ]]; then
    dir="$(dirname "$OUTPUT")"
    name="$(basename "$OUTPUT" .lance)"
    tar -czf "$OUTPUT.tar.gz.partial" -C "$dir" "$name.lance" "${name}_episodes.lance"
    mv "$OUTPUT.tar.gz.partial" "$OUTPUT.tar.gz"
    echo "Archived to $OUTPUT.tar.gz"
  fi
fi
