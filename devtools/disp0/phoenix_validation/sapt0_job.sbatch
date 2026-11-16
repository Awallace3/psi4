#!/usr/bin/env bash
# Resources and all paths are explicit in the immutable submission manifest.
set -euo pipefail
umask 077
: "${SAPT0_SOURCE:?}" "${SAPT0_PIN:?}" "${SAPT0_BUILD_ROOT:?}" "${SAPT0_RUN_ROOT:?}" \
  "${SAPT0_INPUTS:?}" "${SAPT0_SYSTEM:?}" "${SAPT0_PYTHON:?}" "${SLURM_JOB_ID:?}"
[[ "$(git -C "$SAPT0_SOURCE" rev-parse HEAD)" == "$SAPT0_PIN" ]]
[[ -z "$(git -C "$SAPT0_SOURCE" status --porcelain)" ]]
printf 'Job: %s\nQOS: %s\nPartition: %s\n' \
    "$SLURM_JOB_ID" "${SLURM_JOB_QOS:-unknown}" "${SLURM_JOB_PARTITION:-unknown}"
run="$SAPT0_RUN_ROOT/job-$SLURM_JOB_ID"
mkdir "$run"
export TMPDIR="/storage/scratch1/3/awallace43/sapt0-$SLURM_JOB_ID"
mkdir -m 0700 "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD OMP_PROC_BIND OMP_PLACES
export LD_LIBRARY_PATH="$(dirname "$(dirname "$SAPT0_PYTHON")")/lib"
term_received=0
child=
on_term() {
    term_received=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    code=$?
    (( term_received == 0 )) || code=143
    printf 'exit_code=%s\nterm_received=%s\n' "$code" "$term_received" > "$run/job-status.env"
    trap - EXIT
    exit "$code"
}
trap on_term TERM INT
trap on_exit EXIT
srun --kill-on-bad-exit=1 "$SAPT0_PYTHON" -B \
    "$SAPT0_SOURCE/devtools/disp0/phoenix_validation/sapt0_job.py" \
    --build-root "$SAPT0_BUILD_ROOT" --inputs "$SAPT0_INPUTS" \
    --system "$SAPT0_SYSTEM" --output "$run" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
