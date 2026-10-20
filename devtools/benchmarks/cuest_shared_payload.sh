#!/usr/bin/env bash
# Invoked only inside the serialized srun step, never directly on a login node.
set -euo pipefail
manifest=$1 step=$2 seconds=$3
[[ "$4" == -- ]]
shift 4
uuid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpu_uuid"])' "$manifest")
[[ "$uuid" == GPU-* ]]
date -u +%FT%TZ > "$step/GPU_CHECK_STARTED"
check_gpu_empty() {
    local target=$1
    nvidia-smi -i "$uuid" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits > "$target" || return 1
    # NVML normally returns an empty table. Accept its explicit empty message.
    ! grep -qvE '^[[:space:]]*$|^No running processes found' "$target"
}
if ! check_gpu_empty "$step/gpu-processes-before.csv"; then
    echo "GPU not idle before step; no command launched" > "$step/CLEANUP_UNVERIFIED"
    exit 96
fi
rc=0
timeout --signal=TERM --kill-after=30s "$seconds" "$@" || rc=$?
# srun owns the complete step/cgroup; do not let a surviving child leak into
# the next agent's timings. Never kill an unknown process to clear this gate.
if ! check_gpu_empty "$step/gpu-processes-after.csv"; then
    echo "GPU cleanup not established; owner intervention required" > "$step/CLEANUP_UNVERIFIED"
    exit 96
fi
date -u +%FT%TZ > "$step/GPU_CLEANUP_VERIFIED"
exit "$rc"
