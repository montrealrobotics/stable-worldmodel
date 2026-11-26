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
# Writes and archives on $SLURM_TMPDIR, then copies <output>.lance, its _episodes
# table, the collection JSON and <output>.tar.gz to the --output directory.
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

# Write and archive on node-local disk, then copy to the requested location
# once (the network filesystem is slow for ~20 GB of small writes and reads).
ARGS=()
OUTPUT=""
LIMITED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --output=*) OUTPUT="${1#*=}"; shift ;;
    --limit*) LIMITED=1; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
OUTPUT="${OUTPUT:?Pass --output <dataset>.lance}"
[[ -e "$OUTPUT" ]] && { echo "output exists: $OUTPUT" >&2; exit 1; }
DEST_DIR="$(dirname "$OUTPUT")"
NAME="$(basename "$OUTPUT" .lance)"
LOCAL_DIR="${SLURM_TMPDIR:?}/rerender"
mkdir -p "$LOCAL_DIR" "$DEST_DIR"

python -u scripts/data/rerender_pusht_dataset.py \
  --output "$LOCAL_DIR/$NAME.lance" "${ARGS[@]}"

FILES=("$NAME.lance" "${NAME}_episodes.lance")
# Test runs (--limit) are not archived.
if [[ "$COLLECT_ARCHIVE" == 1 && "$LIMITED" == 0 ]]; then
  if command -v pigz >/dev/null; then GZ=(pigz -p "${SLURM_CPUS_PER_TASK:-8}"); else GZ=(gzip); fi
  tar -C "$LOCAL_DIR" -cf - "${FILES[@]}" | "${GZ[@]}" > "$LOCAL_DIR/$NAME.lance.tar.gz"
  cp "$LOCAL_DIR/$NAME.lance.tar.gz" "$OUTPUT.tar.gz.partial"
  mv "$OUTPUT.tar.gz.partial" "$OUTPUT.tar.gz"
  echo "Archived to $OUTPUT.tar.gz"
fi
cp "$LOCAL_DIR/$NAME.collection.json" "$DEST_DIR/"
for f in "${FILES[@]}"; do
  cp -r "$LOCAL_DIR/$f" "$DEST_DIR/$f.partial"
  mv "$DEST_DIR/$f.partial" "$DEST_DIR/$f"
done
echo "Copied dataset to $DEST_DIR/$NAME.lance"
