#!/usr/bin/env bash
# INPUT-ONLY approval; never a dependency/scientific build or GPU request.
#SBATCH --job-name=disp0-sdk-input
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --signal=TERM@30
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/logs/%x-%j.err
set -euo pipefail
umask 077
P=/storage/project/r-cs207-0/awallace43
child=
term_received=0
statusfile="$P/runs/saptdft-cuest-disp0/logs/${SLURM_JOB_ID:?}.input-status"
on_term() { term_received=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
on_exit() {
    rc=$?
    set +e
    (( term_received == 0 )) || rc=143
    (set -o noclobber; printf 'exit_code=%s\nterm_received=%s\nrun=%s\n' \
        "$rc" "$term_received" "${DISP0_INPUT_RUN:-uninitialized}" > "$statusfile") || rc=1
    trap - EXIT
    exit "$rc"
}
trap on_term TERM INT
trap on_exit EXIT
: "${DISP0_INPUT_SOURCE:?}" "${DISP0_INPUT_PIN:?}" "${DISP0_INPUT_APPROVAL:?fresh approval required}"
[[ "$DISP0_INPUT_PIN" =~ ^[0-9a-f]{40}$ ]]
info="$(scontrol show job "$SLURM_JOB_ID" -o)"
printf '%s\n' "$info"
[[ "$info" == *" Account=gts-cs207-chemx "* && "$info" == *" QOS=embers "* \
   && "$info" == *" Requeue=0 "* && "$info" == *" NumCPUs=1 "* \
   && "$info" == *" NumNodes=1 "* && "$info" == *" TimeLimit=00:10:00 "* \
   && "$info" == *" Partition=cpu-small "* ]]
[[ "$info" == *" MinMemoryNode=2G "* || "$info" == *" MinMemoryNode=2048M "* ]]
[[ "$info" != *"gres/gpu"* && "$info" != *"gres:gpu"* ]]
export DISP0_INPUT_RUN="$P/runs/saptdft-cuest-disp0/input-prep-$DISP0_INPUT_PIN-$SLURM_JOB_ID"
srun --ntasks=1 --cpus-per-task=1 bash "$DISP0_INPUT_SOURCE/devtools/disp0/input_prep_payload.sh" &
child=$!
set +e
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
set -e
exit "$rc"
