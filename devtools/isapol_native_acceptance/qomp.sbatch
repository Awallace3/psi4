#!/bin/bash
#SBATCH --job-name=ccasp-qomp-qualification
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --signal=TERM@120
# Parent supplies absolute sibling logs. All modes except build require 2h.
builtin set -euo pipefail
builtin umask 077
# Shell builtins ONLY until inherited loader/Git/PATH selectors are gone.
# This cannot undo the scheduler's initial shell loader or site startup.
while IFS= builtin read -r key; do
    case "$key" in
        USER|LOGNAME|HOME|SLURM_JOB_ID|SLURM_JOBID|SLURM_CPUS_PER_TASK|SLURM_CLUSTER_NAME|\
        SLURMD_NODENAME|SLURM_JOB_NODELIST|SLURM_NODELIST|SLURM_JOB_NUM_NODES|SLURM_NNODES|\
        SLURM_NTASKS|SLURM_NPROCS|SLURM_TASKS_PER_NODE|SLURM_JOB_CPUS_PER_NODE|SLURM_NODEID|\
        SLURM_PROCID|SLURM_LOCALID|SLURM_STEP_ID) ;;
        *) builtin unset "$key" ;;
    esac
done < <(builtin compgen -e)
builtin export PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 SLURM_EXPORT_ENV=ALL
builtin export GIT_OPTIONAL_LOCKS=0
[[ $# -eq 4 || $# -eq 7 ]] || exit 2
mode=$1 source_root=$2 commit=$3 tree=$4
[[ $mode == build || $mode == checks || $mode == collect-existing || $mode == checks-existing ]] || exit 2
[[ $source_root == /* && $commit =~ ^[0-9a-f]{40}$ && $tree =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ $mode != build || $# -eq 4 ]] || exit 2
[[ $mode != checks || $# -eq 7 ]] || exit 2
[[ $mode != collect-existing || $# -eq 4 ]] || exit 2
[[ $mode != checks-existing || $# -eq 7 ]] || exit 2
helper=$source_root/devtools/isapol_native_acceptance/phoenix_build.py
payload=$helper
[[ $mode != checks ]] || payload=$source_root/devtools/isapol_s66_campaign/phoenix_installed_checks.py
interface=qomp
if [[ $mode == *-existing ]]; then
    payload=$source_root/devtools/isapol_s66_campaign/phoenix_installed_checks.py
    helper=$payload
    interface=qomp-existing
fi
[[ -f $helper && -f $payload ]] || exit 2
token=$(/usr/bin/python3 -I -S -B -c 'import uuid; print(uuid.uuid4())')
args=(--ops-commit "$commit" --ops-tree "$tree" --launch-token "$token")
if [[ $mode == checks ]]; then
    args+=(--build-identity-sha256 "$5" --build-record-sha256 "$6" --collection-sha256 "$7")
fi
if [[ $mode == checks-existing ]]; then
    args+=(--verification-sha256 "$5" --collection-sha256 "$6" --collection-result-sha256 "$7")
fi
child= interrupted=0 finalizing=0 failure_cleanup=0 status=0
failed_finalization() {
    failure_cleanup=1
    /usr/bin/python3 -I -S -B "$helper" "$interface" finalize --mode "$mode" \
        --ops-commit "$commit" --ops-tree "$tree" --launch-token "$token" --exit-code "$status" || true
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
    /usr/bin/python3 -I -S -B "$helper" "$interface" finalize --mode "$mode" \
        --ops-commit "$commit" --ops-tree "$tree" --launch-token "$token" --exit-code "$status" &
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
    /usr/bin/python3 -I -S -B "$payload" "$interface" "$mode" "${args[@]}" &
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
