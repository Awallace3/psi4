#!/usr/bin/env bash
#SBATCH --job-name=ccasp-q-oracle
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
# Parent supplies absolute --chdir/--output/--error in a fresh existing log root.
set -euo pipefail
umask 077
args=("$@")
source_root=
for ((i=0; i<${#args[@]}; i++)); do
    [[ ${args[i]} != --launch-token && ${args[i]} != --exit-code ]] || exit 2
    if [[ ${args[i]} == --source ]]; then source_root=${args[i+1]}; fi
done
[[ $source_root == /* ]] || exit 2
helper=$source_root/devtools/isapol_q_omp/ops.py
token=$(/usr/bin/python3 -I -B -c 'import uuid; print(uuid.uuid4())')
child= interrupted=0
term() { interrupted=1; [[ -z $child ]] || kill -TERM "$child" 2>/dev/null || true; }
finish() {
    status=$?
    trap - EXIT
    trap '' TERM INT
    (( interrupted == 0 )) || status=143
    /usr/bin/python3 -I -B "$helper" oracle-finalize "${args[@]}" \
        --launch-token "$token" --exit-code "$status" || status=$?
    exit "$status"
}
trap term TERM INT
trap finish EXIT
(( interrupted == 0 )) || exit 143
/opt/slurm/current/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=30 \
    /usr/bin/python3 -I -B "$helper" oracle "${args[@]}" --launch-token "$token" &
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
