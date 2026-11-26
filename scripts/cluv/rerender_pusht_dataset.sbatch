#!/usr/bin/env bash
#SBATCH --job-name=rerender_pusht_dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
set -euo pipefail

# Re-render a dataset's frames from its saved states (scripts/data/rerender_pusht_dataset.py).
# Args: <source .lance> <output name> [rerender options]. Writes and archives on
# $SLURM_TMPDIR, then copies the dataset, episodes table, JSON and .tar.gz to DEST_DIR.
#   cluv submit mila scripts/cluv/rerender_pusht_dataset.sbatch -- \
#     /network/scratch/o/ozgur.aslan/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance \
#     basecam_fov1_noisy_replay_100k --camera-fov 1.0
SOURCE="${1:?Pass the source .lance dataset as the first argument}"
NAME="${2:?Pass the output dataset name (no .lance) as the second argument}"
shift 2
DEST_DIR=/network/scratch/o/ozgur.aslan/le-wm/datasets
PROJECT_DIR="$PWD"
[[ ! -e "$DEST_DIR/$NAME.lance" ]] || { echo "$DEST_DIR/$NAME.lance exists" >&2; exit 1; }

module load cuda/12.6.0/cudnn/9.3
module load anaconda/3
VENV_PATH=/network/scratch/o/ozgur.aslan/le-wm/.swm
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$HOME/git/mani-exps/.local-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "$SLURM_TMPDIR"
"$VENV_PATH/bin/python" -u "$PROJECT_DIR/scripts/data/rerender_pusht_dataset.py" \
  --source "$SOURCE" --output "$NAME.lance" "$@"

tar -czf "$NAME.lance.tar.gz" "$NAME.lance" "${NAME}_episodes.lance"
cp -r "$NAME.lance" "${NAME}_episodes.lance" "$NAME.collection.json" "$NAME.lance.tar.gz" "$DEST_DIR/"
echo "Copied $NAME to $DEST_DIR"
