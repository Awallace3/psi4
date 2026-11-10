#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${DISP0_SOURCE:?}" "${DISP0_PIN:?}" "${DISP0_RUN_ROOT:?}" "${SLURM_JOB_ID:?}"
export DISP0_JOB_RUN="$DISP0_RUN_ROOT/job-$SLURM_JOB_ID"
mkdir "$DISP0_JOB_RUN"
term_received=0
child=
on_term() { term_received=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
on_exit() {
    status=$?
    (( term_received == 0 )) || status=143
    printf 'exit_code=%s\nterm_received=%s\n' "$status" "$term_received" > "$DISP0_JOB_RUN/job-status.env"
    trap - EXIT
    exit "$status"
}
trap on_term TERM INT
trap on_exit EXIT
[[ "$(git -C "$DISP0_SOURCE" rev-parse HEAD)" == "$DISP0_PIN" ]]
[[ -z "$(git -C "$DISP0_SOURCE" status --porcelain)" ]]
# Same readonly environment used by the CPU prerequisite; no package installs.
env="/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/opt064-770ecf90108aa7783da766bf0402d8c267dcbbed-14254131-0/environment"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD OMP_PROC_BIND OMP_PLACES
export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$env/lib" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
export CONDA_PREFIX="$env" MKL_INTERFACE_LAYER=LP64
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1
# The Python step creates/binds its private scratch AFTER srun has applied
# scheduler step-environment rewrites. Never export a stale batch TMPDIR proof.
srun --kill-on-bad-exit=1 "$env/bin/python" -B \
    "$DISP0_SOURCE/devtools/disp0/phoenix_validation/sealed_h200.py" \
    --execute "$DISP0_RUN_ROOT/manifest.json" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
