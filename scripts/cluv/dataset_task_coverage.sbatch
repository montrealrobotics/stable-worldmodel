#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Dataset coverage of the eval tasks (scripts/data/dataset_task_coverage.py), CPU only.
# Runs where the eval results, the expert goal cache and the dataset live (Mila):
#   cluv submit mila scripts/cluv/dataset_task_coverage.sbatch --job-name=task_coverage -- \
#     --run-name=noisy_100k
# Optional flags (before any script args):
#   --dataset=<path>   default $SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance
#   --run-name=<name>
# Everything else goes to the script (eval runs as name=run_id:epoch; default: the ep20 runs).
# Results: $SCRATCH/logs/stable-worldmodel/dataset_task_coverage/<run-name>_<job id>
# Plots (laptop, after copying the folder): python scripts/data/plot_task_coverage.py <folder>

set -euo pipefail

DATASET="$SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance"
RUN_NAME="dataset_task_coverage"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done

OUT="$SCRATCH/logs/stable-worldmodel/dataset_task_coverage/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl"

uv run --extra train --extra data python scripts/data/dataset_task_coverage.py \
  --dataset "$DATASET" --out "$OUT" "$@"
