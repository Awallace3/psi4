#!/usr/bin/env bash
# Explicit resources, QoS and absolute paths are supplied by submit_gate.py.
set -euo pipefail
umask 077
: "${DISP0_SOURCE:?}" "${DISP0_RUN_ROOT:?}" "${SLURM_JOB_ID:?}"
run="$DISP0_RUN_ROOT/job-$SLURM_JOB_ID"
mkdir "$run"
export DISP0_JOB_RUN="$run"
term_received=0
child=
on_term() {
    term_received=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    status=$?
    set +e
    (( term_received == 0 )) || status=143
    printf 'exit_code=%s\nterm_received=%s\n' "$status" "$term_received" > "$run/job-status.env"
    trap - EXIT
    exit "$status"
}
trap on_term TERM INT
trap on_exit EXIT
srun --kill-on-bad-exit=1 bash "$DISP0_SOURCE/devtools/disp0/phoenix_validation/gate_payload.sh" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
