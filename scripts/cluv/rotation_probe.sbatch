#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00

# Rotation probe of LeWM checkpoints (scripts/plan/rotation_probe.py), one GPU.
# Reads the extracted dataset under $SCRATCH/le-wm/datasets (Mila):
#   cluv submit mila scripts/cluv/rotation_probe.sbatch --partition=long --gpus=l40s:1 \
#     --job-name=rotation_probe -- --run-name=ep20 \
#     inbatch=noisy_lewm_res_nobn_scratch_search_w0.01_inbatch40_s3072_11811959/weights_epoch_20.pt \
#     cem_s1=noisy_lewm_res_nobn_scratch_search_w0.01_s1_27001812/weights_epoch_20.pt \
#     base_s1=noisy_lewm_res_nobn_w0.009_s1_27001817/weights_epoch_20.pt
# Optional flag (before the checkpoints): --run-name=<name>. Everything else goes to the script
# (checkpoints as label=path relative to $SCRATCH/le-wm/checkpoints, --num-clips, --horizon, ...).
# Results: $SCRATCH/logs/stable-worldmodel/rotation_probe/<run-name>_<job id>
# Plots (laptop, after copying the folder): python scripts/plan/plot_rotation_probe.py <folder>

set -euo pipefail

RUN_NAME="rotation_probe"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    *) break ;;
  esac
done

OUT="$SCRATCH/logs/stable-worldmodel/rotation_probe/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export STABLEWM_HOME="$SCRATCH/le-wm"
export HF_HOME="$SCRATCH/huggingface"
export MPLCONFIGDIR="$SLURM_TMPDIR/mpl"

# Private venv per job: `cluv sync` / `cluv submit` run a plain `uv sync` (no extras) on the shared
# project .venv, which deletes the train/data extras under any job still using it.
export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:?}/venv"
uv run --extra train --extra data python scripts/plan/rotation_probe.py --out "$OUT" "$@"
