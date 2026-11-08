#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${DISP0_OUTER:?}" "${DISP0_OUTER_PIN:?}" "${DISP0_RUN_ROOT:?}" "${SLURM_JOB_ID:?}"
: "${DISP0_GATE_ENV:?}" "${DISP0_STAGE_THREADS:?}" "${DISP0_MANIFEST_SHA256:?}"
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
trap on_term TERM INT HUP
trap on_exit EXIT
[[ "$(git -C "$DISP0_OUTER" rev-parse HEAD)" == "$DISP0_OUTER_PIN" ]]
[[ -z "$(git -C "$DISP0_OUTER" status --porcelain)" ]]
env="$DISP0_GATE_ENV"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD OMP_PROC_BIND OMP_PLACES
export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$env/lib" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$DISP0_STAGE_THREADS" MKL_NUM_THREADS="$DISP0_STAGE_THREADS"
export OPENBLAS_NUM_THREADS="$DISP0_STAGE_THREADS" OMP_DYNAMIC=FALSE
export CONDA_PREFIX="$env" MKL_INTERFACE_LAYER=LP64
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1
driver="${DISP0_EXECUTE_DRIVER:-stage_scaling.py}"
case "$driver" in stage_scaling.py|scf_repeat_probe.py) ;; *) exit 2 ;; esac
srun --kill-on-bad-exit=1 "$env/bin/python" -B \
    "$DISP0_OUTER/devtools/disp0/phoenix_validation/$driver" \
    --execute "$DISP0_RUN_ROOT/manifest.json" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
