#!/bin/bash
# Scheduler resource/QoS/dependency arguments are explicit at submission.
# Args match recovery_job.sh; source/build/input paths are immutable.
set -euo pipefail
repo=${1:?}; commit=${2:?}; build=${3:?}; run=${4:?}; data=${5:?}
build_job=${6:?}; gpu=${7:-H200}; qos=${8:-embers}
mkdir "$(dirname "$run")/batch-metadata"
metadata="$(dirname "$run")/batch-metadata"
{
    printf 'job=%s\nhost=%s\ncommit=%s\nqos=%s\nstarted=%s\n' \
        "${SLURM_JOB_ID:?}" "$(hostname -f)" "$commit" "$qos" "$(date -Iseconds)"
    printf 'cpus=%s\nmemory_mb=%s\nvisible_gpus=%s\n' \
        "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" "${CUDA_VISIBLE_DEVICES:-unset}"
    lscpu
} > "$metadata/resources.txt"
child=; signalled=0
terminate() {
    signalled=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
trap terminate TERM INT
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres="gpu:${gpu,,}:1" \
    bash "$repo/devtools/benchmarks/cuest_direct/recovery_job.sh" \
    "$repo" "$commit" "$build" "$run" "$data" "$build_job" "$gpu" "$qos" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
(( signalled == 0 )) || status=143
printf 'exit=%s\nsignal=%s\nended=%s\n' "$status" "$signalled" "$(date -Iseconds)" \
    > "$metadata/terminal.txt"
exit "$status"
