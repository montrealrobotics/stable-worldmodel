#!/usr/bin/env bash
# Submit the sched_n100 eval of one checkpoint on Mila (run on a login node; used by
# scripts/cluv/watch_evals_mila.sh):  bash scripts/cluv/eval_checkpoint_mila.sh <run dir> <epoch>
# Submits one sched_n100 eval array for that checkpoint: goal offsets 10/15/20/25 (one task each,
# horizons 1-5 in parallel), CEM, n=100, same settings as the earlier chained evals.
set -euo pipefail
RUN_DIR=$1
EPOCH=$2
L=/network/scratch/o/ozgur.aslan/le-wm
GPU=${EVAL_GPU:-l40s}
ARCHIVE=overhead_rgb_noisy_replay_100k.lance.tar.gz
cd "$(dirname "$0")/../.."
CHECKPOINTS="$RUN_DIR/weights_epoch_$EPOCH.pt" GOAL_OFFSETS='10 15 20 25' \
VENV_TAR=$L/swm_venv.tar DATASET_TAR=$L/datasets/$ARCHIVE EXPERT_CACHE_TAR=$L/expert_goal_cache.tar \
EVAL_SAVE_DIR=$L/evals/sched_n100 HORIZON='1 2 3 4 5' CEM_ONLY=1 NUM_EVAL=100 PARALLEL_HORIZONS=1 \
EVAL_CONFIG=maniskill_overhead DATASET_NAME=overhead_rgb_noisy_replay_100k.lance \
sbatch --parsable --array=0-3 --partition=long --gres=gpu:$GPU:1 --cpus-per-task=10 --mem=32G --time=3:00:00 \
  --job-name="eval-$(basename "$RUN_DIR")-e$EPOCH" \
  slurm_scripts/eval_vision_lewm.sbatch solver.batch_size=100 ++save_videos=false  # Mila copy (gitignored, not synced)
