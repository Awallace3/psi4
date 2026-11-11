#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${FSAPT_SOURCE:?}" "${FSAPT_PYTHON:?}" "${FSAPT_RUN_ROOT:?}" \
  "${FSAPT_MANIFEST_SHA256:?}" "${SLURM_JOB_ID:?}"
run="$FSAPT_RUN_ROOT/job-$SLURM_JOB_ID"
mkdir "$run"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD OMP_PROC_BIND OMP_PLACES
export LD_LIBRARY_PATH="$(dirname "$(dirname "$FSAPT_PYTHON")")/lib"
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
srun --kill-on-bad-exit=1 "$FSAPT_PYTHON" -B \
    "$FSAPT_SOURCE/devtools/fsapt_integration/phoenix_timing.py" \
    --manifest "$FSAPT_RUN_ROOT/manifest.json" --output "$run" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
