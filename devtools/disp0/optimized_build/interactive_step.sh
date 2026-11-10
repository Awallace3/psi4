#!/usr/bin/env bash
# Inside this allocation's exclusive srun, never on the login node.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${DISP0_OPT_SOURCE:?}" "${DISP0_OPT_PIN:?}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_STEP_ID" =~ ^[0-9]+$ ]]
[[ "${SLURM_CPUS_PER_TASK:?}" == 8 && -z "${SLURM_JOB_GPUS:-}" ]]
P=/storage/project/r-cs207-0/awallace43
export DISP0_OPT_INTERACTIVE=1
export DISP0_OPT_RUN="$P/runs/saptdft-cuest-disp0/opt064-$DISP0_OPT_PIN-$SLURM_JOB_ID-$SLURM_STEP_ID"
info="$(scontrol show job "$SLURM_JOB_ID" -o)"
printf '%s\n' "$info"
[[ "$info" == *" Account=gts-cs207-chemx "* && "$info" == *" QOS=embers "* \
    && "$info" == *" JobState=RUNNING "* && "$info" == *" NumCPUs=8 "* \
    && "$info" == *" MinMemoryNode=48G "* && "$info" == *" Requeue=0 "* ]]
child=
signalled=0
term() { signalled=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
finish() {
    code=$?
    (( signalled == 0 )) || code=143
    status="$P/runs/saptdft-cuest-disp0/logs/$SLURM_JOB_ID-$SLURM_STEP_ID.opt064-status"
    printf 'exit_code=%s\nsignalled=%s\nrun=%s\n' "$code" "$signalled" "$DISP0_OPT_RUN" > "$status" || code=1
    trap - EXIT
    exit "$code"
}
trap term TERM INT
trap finish EXIT
bash "$DISP0_OPT_SOURCE/devtools/disp0/optimized_build/payload.sh" &
child=$!
set +e
wait "$child"; code=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; code=$?; done
exit "$code"
