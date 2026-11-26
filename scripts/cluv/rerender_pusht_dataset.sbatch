#!/usr/bin/env bash
#SBATCH --job-name=rerender_pusht_dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00

# Re-render an OverheadPushT dataset from its saved states with another camera
# (scripts/data/rerender_pusht_dataset.py); trajectories stay identical.
#   cluv submit mila scripts/cluv/rerender_pusht_dataset.sbatch -- \
#     --source /network/scratch/o/ozgur.aslan/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance \
#     --output /network/scratch/o/ozgur.aslan/le-wm/datasets/basecam_fov1_noisy_replay_100k.lance \
#     --camera-fov 1.0
# The output (and its _episodes table) is archived to <output>.tar.gz at the end.
set -euo pipefail

# Not ARCHIVE: Mila login shells export it as the archive storage path.
COLLECT_ARCHIVE="${COLLECT_ARCHIVE:-1}"

module load cuda/12.6.0/cudnn/9.3
module load anaconda/3
VENV_PATH="${VENV_PATH:-$SCRATCH/le-wm/.swm}"
source "$VENV_PATH/bin/activate"
export VIRTUAL_ENV="$VENV_PATH" PATH="$VENV_PATH/bin:$PATH"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MANI_EXPS_DIR="${MANI_EXPS_DIR:-$HOME/git/mani-exps}"
export LD_LIBRARY_PATH="$MANI_EXPS_DIR/.local-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

python -u scripts/data/rerender_pusht_dataset.py "$@"

if [[ "$COLLECT_ARCHIVE" == 1 ]]; then
  OUTPUT=""
  LIMITED=0
  prev=""
  for arg in "$@"; do
    [[ "$prev" == --output ]] && OUTPUT="$arg"
    [[ "$arg" == --output=* ]] && OUTPUT="${arg#*=}"
    [[ "$arg" == --limit* ]] && LIMITED=1
    prev="$arg"
  done
  # Test runs (--limit) are not archived.
  if [[ -n "$OUTPUT" && "$LIMITED" == 0 ]]; then
    dir="$(dirname "$OUTPUT")"
    name="$(basename "$OUTPUT" .lance)"
    tar -czf "$OUTPUT.tar.gz.partial" -C "$dir" "$name.lance" "${name}_episodes.lance"
    mv "$OUTPUT.tar.gz.partial" "$OUTPUT.tar.gz"
    echo "Archived to $OUTPUT.tar.gz"
  fi
fi
