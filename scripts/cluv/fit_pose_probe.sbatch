#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00

# Pose probes for the oracle planning cost (scripts/plan/fit_pose_probe.py), one GPU.
# Reads the extracted dataset under $SCRATCH/le-wm/datasets (Mila):
#   cluv submit mila scripts/cluv/fit_pose_probe.sbatch --partition=long --gpus=l40s:1 \
#     --job-name=fit_pose_probe -- --run-name=ep20 \
#     base_s1=noisy_lewm_res_nobn_w0.009_s1_27001817/weights_epoch_20.pt \
#     inbatch=noisy_lewm_res_nobn_scratch_search_w0.01_inbatch40_s3072_11811959/weights_epoch_20.pt
# Optional flag (before the checkpoints): --run-name=<name>. Everything else goes to the script
# (checkpoints as label=path relative to $SCRATCH/le-wm/checkpoints, --num-clips, --horizon, ...).
# Results: $SCRATCH/le-wm/pose_probes/<run-name>_<job id>/<label>_{real,both}.pt + summary.md
# Plan with them: scripts/cluv/eval_checkpoint_mila.sh (EVAL_DIR=oracle_cost, objective=pose_probe).

set -euo pipefail

RUN_NAME="pose_probe"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done

OUT="$SCRATCH/le-wm/pose_probes/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export STABLEWM_HOME="$SCRATCH/le-wm"
export HF_HOME="$SCRATCH/huggingface"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl"

# Private venv per job: `cluv sync` / `cluv submit` run a plain `uv sync` (no extras) on the shared
# project .venv, which deletes the train/data extras under any job still using it.
export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:?}/venv"
uv run --extra train --extra data python scripts/plan/fit_pose_probe.py --out "$OUT" "$@"
