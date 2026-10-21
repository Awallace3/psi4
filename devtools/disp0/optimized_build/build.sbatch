#!/usr/bin/env bash
# One user-requested fresh preparation build, never a replay of an input job.
#SBATCH --job-name=disp0-opt064
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=TERM@90
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/logs/%x-%j.err
set -euo pipefail
umask 077
: "${DISP0_OPT_SOURCE:?}" "${DISP0_OPT_PIN:?}"
P=/storage/project/r-cs207-0/awallace43
export DISP0_OPT_RUN="$P/runs/saptdft-cuest-disp0/opt064-$DISP0_OPT_PIN-$SLURM_JOB_ID"
status_file="$P/runs/saptdft-cuest-disp0/logs/$SLURM_JOB_ID.opt064-status"
child=
signalled=0
term() { signalled=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
finish() {
    code=$?
    (( signalled == 0 )) || code=143
    printf 'exit_code=%s\nsignalled=%s\nrun=%s\n' "$code" "$signalled" "$DISP0_OPT_RUN" > "$status_file" || code=1
    trap - EXIT
    exit "$code"
}
trap term TERM INT
trap finish EXIT
info="$(scontrol show job "$SLURM_JOB_ID" -o)"
printf '%s\n' "$info"
[[ "$info" == *" Account=gts-cs207-chemx "* && "$info" == *" QOS=embers "* \
    && "$info" == *" Requeue=0 "* && "$info" == *" NumCPUs=8 "* ]]
[[ -z "${SLURM_JOB_GPUS:-}" ]]
srun bash "$DISP0_OPT_SOURCE/devtools/disp0/optimized_build/payload.sh" &
child=$!
set +e
wait "$child"; code=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; code=$?; done
exit "$code"
