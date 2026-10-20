#!/usr/bin/env bash
# Proposal only: fresh explicit CPU approval + exact-route/headroom checks required.
#SBATCH --job-name=disp0-e2-revised
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@90
#SBATCH --no-requeue
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/logs/%x-%j.err
set -euo pipefail
umask 077
P=/storage/project/r-cs207-0/awallace43
metadata="$P/runs/saptdft-cuest-disp0/logs/${SLURM_JOB_ID:-unassigned}.revised-status"
child=
term_received=0
on_term() { term_received=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
on_exit() {
    status=$?
    set +e
    (( term_received == 0 )) || status=143
    if ! printf 'exit_code=%s\nterm_received=%s\nrun=%s\n' \
        "$status" "$term_received" "${DISP0_RUN:-uninitialized}" > "$metadata"; then
        status=1
    elif (( status == 0 )); then
        printf 'CPU_PREPARATION_EXIT_ZERO\n' >> "$metadata" || status=1
    else
        printf 'FAILED_OR_SIGNALLED\n' >> "$metadata" || status=1
    fi
    trap - EXIT
    exit "$status"
}
trap on_term TERM INT
trap on_exit EXIT
: "${SLURM_JOB_ID:?}" "${DISP0_SOURCE:?}" "${DISP0_PIN:?}"
: "${DISP0_NEW_CPU_BUILD_APPROVAL_REFERENCE:?fresh CPU approval required}"
[[ "$DISP0_PIN" =~ ^[0-9a-f]{40}$ ]]
export DISP0_RUN="$P/runs/saptdft-cuest-disp0/revised-build-$DISP0_PIN-$SLURM_JOB_ID"
info="$(scontrol show job "$SLURM_JOB_ID" -o)"
printf '%s\n' "$info"
[[ "$info" == *" Account=gts-cs207-chemx "* && "$info" == *" QOS=embers "* \
   && "$info" == *" Requeue=0 "* && "$info" == *" NumCPUs=8 "* ]]
srun bash "$DISP0_SOURCE/devtools/disp0/revised_cpu_payload.sh" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
