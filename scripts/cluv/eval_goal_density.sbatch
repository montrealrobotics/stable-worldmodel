#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

# Eval goals vs training-data density (scripts/data/eval_goal_density.py), CPU only.
# Runs where the eval results, the expert goal cache and the dataset live (Mila):
#   cluv submit mila scripts/cluv/eval_goal_density.sbatch --job-name=eval_goal_density -- \
#     --run-name=sched_n100_all
# Optional flags (before any script args):
#   --results-glob=<glob>   default $SCRATCH/le-wm/evals/sched_n100/*/random_goal_pusht_results.txt
#   --dataset=<path>        default $SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance
#   --cache-dir=<dir>       default $SCRATCH/le-wm/expert_goal_cache
# Everything else goes to the script (e.g. --run-filter 'noisy_lewm_res_nobn').
# Results: $SCRATCH/logs/stable-worldmodel/eval_goal_density/<run-name>_<job id>

set -euo pipefail

LEWM="$SCRATCH/le-wm"
RESULTS_GLOB="$LEWM/evals/sched_n100/*/random_goal_pusht_results.txt"
DATASET="$LEWM/datasets/overhead_rgb_noisy_replay_100k.lance"
CACHE_DIR="$LEWM/expert_goal_cache"
RUN_NAME="eval_goal_density"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-glob=*) RESULTS_GLOB="${1#*=}"; shift ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --cache-dir=*) CACHE_DIR="${1#*=}"; shift ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done

OUT="$SCRATCH/logs/stable-worldmodel/eval_goal_density/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl"

# Private venv per job: `cluv sync` / `cluv submit` run a plain `uv sync` (no extras) on the shared
# project .venv, which deletes the train/data extras under any job still using it.
export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:?}/venv"
uv run --extra train --extra data python scripts/data/eval_goal_density.py \
  --results-glob "$RESULTS_GLOB" \
  --cache-dir "$CACHE_DIR" \
  --dataset "$DATASET" \
  --out "$OUT" \
  "$@"
