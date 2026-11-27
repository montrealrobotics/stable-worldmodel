#!/usr/bin/env bash
# Submit the sched_n100 eval of one checkpoint on Mila (run on a login node; used by
# scripts/cluv/watch_evals_mila.sh):  bash scripts/cluv/eval_checkpoint_mila.sh <run dir> <epoch>
# Submits one sched_n100 eval array for that checkpoint: goal offsets 10/15/20/25 (one task each,
# horizons 1-5 in parallel), CEM, n=100, same settings as the earlier chained evals.
# Optional: EVAL_DIR=<name> (folder under $L/evals, default sched_n100) and extra Hydra overrides
# for eval_wm.py after the epoch, e.g. the oracle planning cost (scripts/plan/pose_probe.py):
#   EVAL_DIR=oracle_cost bash scripts/cluv/eval_checkpoint_mila.sh <run dir> 20 \
#     objective=pose_probe objective.probe_path=$L/pose_probes/<run>/<label>_both.pt
set -euo pipefail
RUN_DIR=$1
EPOCH=$2
shift 2
L=/network/scratch/o/ozgur.aslan/le-wm
GPU=${EVAL_GPU:-l40s}
# Camera variant: DATASET (training dataset name, no .lance) and EVAL_CONFIG. Runs trained on
# basecam_* datasets: DATASET=basecam_fov1_noisy_replay_100k EVAL_CONFIG=maniskill_basecam.
DATASET=${DATASET:-overhead_rgb_noisy_replay_100k}
EVAL_CONFIG=${EVAL_CONFIG:-maniskill_overhead}
# Expert goals: a frozen cache tar (top-level expert_goal_cache/) is copied to each job;
# without one, goals are built in a shared per-config dir (the cache key includes the camera,
# so caches never mix). Freeze a complete dir with:
#   tar -cf $L/expert_goal_cache_<config>.tar -C $L --transform 's,^expert_goal_cache_<config>,expert_goal_cache,' expert_goal_cache_<config>
CACHE_TAR=$L/expert_goal_cache_$EVAL_CONFIG.tar
[[ "$EVAL_CONFIG" == maniskill_overhead ]] && CACHE_TAR=$L/expert_goal_cache.tar
if [[ -f "$CACHE_TAR" ]]; then
  export EXPERT_CACHE_TAR=$CACHE_TAR
else
  export EXPERT_CACHE_DIR=$L/expert_goal_cache_$EVAL_CONFIG; mkdir -p "$EXPERT_CACHE_DIR"
fi
cd "$(dirname "$0")/../.."
CHECKPOINTS="$RUN_DIR/weights_epoch_$EPOCH.pt" GOAL_OFFSETS='10 15 20 25' \
VENV_TAR=$L/swm_venv.tar DATASET_TAR=$L/datasets/$DATASET.lance.tar.gz \
EVAL_SAVE_DIR=$L/evals/${EVAL_DIR:-sched_n100} HORIZON='1 2 3 4 5' CEM_ONLY=1 NUM_EVAL=100 PARALLEL_HORIZONS=1 \
EVAL_CONFIG=$EVAL_CONFIG DATASET_NAME=$DATASET.lance \
sbatch --parsable --array=0-3 --partition=long --gres=gpu:$GPU:1 --cpus-per-task=10 --mem=32G --time=3:00:00 \
  --job-name="eval-${EVAL_DIR:+$EVAL_DIR-}$(basename "$RUN_DIR")-e$EPOCH" \
  slurm_scripts/eval_vision_lewm.sbatch solver.batch_size=100 ++save_videos=false "$@"  # Mila copy (gitignored, not synced)
