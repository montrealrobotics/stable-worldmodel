#!/usr/bin/env bash
# Submit evals of training runs' epoch checkpoints as soon as they are saved.
# Run on a Mila login node, in tmux, from the repository root:
#   bash scripts/cluv/watch_evals_mila.sh <train job id> [<train job id> ...]
#   EPOCHS="5 10 15 16 17 18 19 20" (default)  POLL_SECONDS=120
# For each job, the run dir is $L/checkpoints/*_<job id>. Each listed weights_epoch_N.pt is
# handed to scripts/cluv/eval_checkpoint_mila.sh once complete (size unchanged for 30 s;
# torch.save is not atomic). A marker .eval_submitted_N in the run dir prevents repeats, also
# when the watcher is restarted. A job is done when all its epochs are submitted, or when it
# has left the queue (after a last pass). The watcher exits when every job is done.
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${LEWM_ROOT:-/network/scratch/o/ozgur.aslan/le-wm}
EPOCHS=${EPOCHS:-5 10 15 16 17 18 19 20}
POLL_SECONDS=${POLL_SECONDS:-120}
(( $# )) || { echo "usage: $0 <train job id> [...]" >&2; exit 1; }

log() { echo "$(date '+%F %T') $*"; }

# Prints the run dir of a job (empty until the job has created it).
run_dir() { compgen -G "$L/checkpoints/*_$1" | head -1 || true; }

# Submits ready epochs of one run; prints how many listed epochs are still unsubmitted.
submit_ready() {
  local dir=$1 e f size left=0
  for e in $EPOCHS; do
    [[ -e "$dir/.eval_submitted_$e" ]] && continue
    f="$dir/weights_epoch_$e.pt"
    if [[ -s "$f" ]] && size=$(stat -c %s "$f") && sleep "${SETTLE_SECONDS:-30}" && [[ "$(stat -c %s "$f")" == "$size" ]]; then
      if id=$(bash scripts/cluv/eval_checkpoint_mila.sh "$dir" "$e"); then
        touch "$dir/.eval_submitted_$e"
        log "$(basename "$dir") epoch $e -> eval $id" >&2
        continue
      fi
      log "$(basename "$dir") epoch $e: eval submission failed, retrying" >&2
    fi
    left=$((left + 1))
  done
  echo "$left"
}

job_queued() {  # a failed squeue (controller timeout) counts as still queued
  local out
  out=$(squeue -h -j "$1" -o %i 2>/dev/null) || return 0
  [[ -n "$out" ]]
}

declare -A DONE=()
while (( ${#DONE[@]} < $# )); do
  for job in "$@"; do
    [[ -n "${DONE[$job]:-}" ]] && continue
    queued=0; job_queued "$job" && queued=1  # before the pass, so a last pass follows the job's end
    dir=$(run_dir "$job")
    left=all  # no run dir yet
    [[ -n "$dir" ]] && left=$(submit_ready "$dir")
    if [[ -n "$dir" && "$left" == 0 ]]; then
      DONE[$job]=1; log "job $job: all epochs submitted"
    elif (( ! queued )); then
      DONE[$job]=1; log "job $job left the queue; $left listed epoch(s) never saved or submitted"
    fi
  done
  (( ${#DONE[@]} < $# )) && sleep "$POLL_SECONDS"
done
log "done"
