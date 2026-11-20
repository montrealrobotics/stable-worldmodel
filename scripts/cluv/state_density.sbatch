#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00

# State-density / coverage analysis (scripts/data/state_density.py), CPU only:
# rorqual's rrg-gberseth allocation is GPU-only, so CPU jobs need --account=def-gberseth:
#   cluv submit rorqual scripts/cluv/state_density.sbatch --job-name=state_density --account=def-gberseth -- \
#     --dataset=noisy=overhead_rgb_noisy_replay_100k.lance.tar.gz \
#     --run-name=noisy_100k \
#     --split-by replay_idx
# --dataset=<label>=<source>, repeatable. <source> is an archive in
#   $SCRATCH/le-wm/cluv_data (extracted to local disk), or an absolute path to
#   a .lance table / a directory holding one (read in place, e.g. on-policy rollouts).
# Everything after the --dataset/--run-name flags goes to state_density.py.
# Results: $SCRATCH/logs/stable-worldmodel/state_density/<run-name>_<job id>
#   (the project's logs/ symlink).

set -euo pipefail

DATASETS=()
RUN_NAME="state_density"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASETS+=("${1#*=}"); shift ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done
[[ ${#DATASETS[@]} -gt 0 ]] || { echo "Pass --dataset=<label>=<archive or path> at least once" >&2; exit 1; }

SPECS=()
for spec in "${DATASETS[@]}"; do
  label="${spec%%=*}"
  source="${spec#*=}"
  [[ "$label" != "$spec" ]] || { echo "Expected <label>=<source>, got $spec" >&2; exit 1; }
  if [[ "$source" == /* ]]; then
    [[ -e "$source" ]] || { echo "not found: $source" >&2; exit 1; }
    SPECS+=("$label=$source")
  else
    archive="$SCRATCH/le-wm/cluv_data/$source"
    [[ -e "$archive" ]] || { echo "archive not found: $archive" >&2; exit 1; }
    dest="$SLURM_TMPDIR/datasets/$label"
    mkdir -p "$dest"
    echo "extracting $archive -> $dest"
    tar -xf "$archive" -C "$dest"
    SPECS+=("$label=$dest")
  fi
done

OUT="$SCRATCH/logs/stable-worldmodel/state_density/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl"

# Compute nodes may be offline (UV_OFFLINE=1): the same extras as training,
# which are already in the uv cache (matplotlib comes in with train).
uv run --extra train --extra data python scripts/data/state_density.py \
  "${SPECS[@]}" \
  --out "$OUT" \
  "$@"
