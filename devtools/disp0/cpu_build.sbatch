#!/usr/bin/env bash
#SBATCH --job-name=disp0-e2-build
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
umask 027
P=/storage/project/r-cs207-0/awallace43
export DISP0_RUN="$P/runs/saptdft-cuest-disp0/build-${DISP0_PIN:?}-$SLURM_JOB_ID"
metadata="$P/runs/saptdft-cuest-disp0/logs/$SLURM_JOB_ID.status"
jobinfo="$(scontrol show job "$SLURM_JOB_ID" -o)"
printf '%s\n' "$jobinfo"
[[ "$jobinfo" == *" Account=gts-cs207-chemx "* && "$jobinfo" == *" QOS=embers "* ]]
term_received=0
child=
on_term() {
    term_received=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    status=$?
    (( term_received == 0 )) || status=143
    printf 'exit_code=%s\nterm_received=%s\nrun=%s\n' "$status" "$term_received" "$DISP0_RUN" > "$metadata"
    # A clean shell exit is preparation evidence only, never GPU validation.
    if (( status == 0 )); then
        printf 'CPU_PREPARATION_EXIT_ZERO\n' >> "$metadata"
    else
        printf 'FAILED_OR_SIGNALLED\n' >> "$metadata"
    fi
    trap - EXIT
    exit "$status"
}
trap on_term TERM INT
trap on_exit EXIT
srun bash "${DISP0_SOURCE:?}/devtools/disp0/cpu_build_payload.sh" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
