#!/usr/bin/env bash
#SBATCH --job-name=ccasp-q-molecular-replay
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=TERM@120
# Same reviewed finite srun/TERM-finalization pattern as frozen dc916.
set -euo pipefail
umask 077
export GIT_OPTIONAL_LOCKS=0
args=("$@")
source_root=
for ((i=0; i<${#args[@]}; i++)); do
    [[ ${args[i]} != --launch-token && ${args[i]} != --exit-code ]] || exit 2
    if [[ ${args[i]} == --source ]]; then source_root=${args[i+1]}; fi
done
[[ $source_root == /* ]] || exit 2
helper=$source_root/devtools/isapol_q_molecular_replay/replay.py
token=$(/usr/bin/python3 -I -B -c 'import uuid; print(uuid.uuid4())')
child= interrupted=0 finalizing=0 failure_cleanup=0 status=0
failed_finalization() {
    failure_cleanup=1
    /usr/bin/python3 -I -B "$helper" finalize "${args[@]}" \
        --launch-token "$token" --exit-code "$status" || true
}
term() {
    interrupted=1
    status=143
    (( failure_cleanup == 0 )) || return 0
    if [[ -n $child ]] && kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null || true
    elif (( finalizing )); then
        failed_finalization
    fi
}
finish() {
    status=$?
    trap - EXIT
    finalizing=1
    (( interrupted == 0 )) || status=143
    (( status == 0 )) || failure_cleanup=1
    /usr/bin/python3 -I -B "$helper" finalize "${args[@]}" \
        --launch-token "$token" --exit-code "$status" &
    child=$!
    if (( interrupted && !failure_cleanup )); then term; fi
    wait "$child" || status=$?
    if (( interrupted )); then
        while kill -0 "$child" 2>/dev/null; do wait "$child" || true; done
        status=143
    fi
    child=
    (( status == 0 )) || failed_finalization
    exit "$status"
}
trap term TERM INT
trap finish EXIT
(( interrupted == 0 )) || exit 143
/opt/slurm/current/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=30 \
    /usr/bin/python3 -I -B "$helper" run "${args[@]}" --launch-token "$token" &
child=$!
if (( interrupted )); then term; fi
status=0
wait "$child" || status=$?
if (( interrupted )); then
    while kill -0 "$child" 2>/dev/null; do wait "$child" || true; done
    status=143
fi
child=
exit "$status"
