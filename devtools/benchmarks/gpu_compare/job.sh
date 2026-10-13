#!/usr/bin/env bash
# Submit with explicit resources, absolute logs, and --qos=embers (see README).
# Arguments: REPO COMMIT RUN_DIR {setup|campaign} [system ...]
set -euo pipefail
umask 027
repo=$1
pin=$2
run=$3
mode=$4
shift 4
[[ "$repo" = /* && "$run" = /* && "$pin" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$pin" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
mkdir -p "$run"/{metadata,logs}
child=
term_received=0
on_term() {
    term_received=1
    [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    status=$?
    (( term_received )) && status=143
    printf 'exit_code=%s\nterm_received=%s\n' "$status" "$term_received" \
        > "$run/metadata/shell-exit.txt"
    if (( status == 0 )); then
        touch "$run/metadata/COMPLETED"
    else
        touch "$run/metadata/FAILED"
    fi
}
trap on_term TERM
trap on_exit EXIT
export COMPARE_REPO="$repo" COMPARE_RUN="$run"
export COMPARE_PSI_ENV="${COMPARE_PSI_ENV:-/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt}"
export COMPARE_GPU_ENV="${COMPARE_GPU_ENV:-/storage/project/r-cs207-0/awallace43/miniconda/envs/gpu4pyscf_compare}"
export COMPARE_PSI_PACKAGE="${COMPARE_PSI_PACKAGE:-/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft_cuest/build_cuda_libxc/stage/lib/psi4}"
export COMPARE_PSI_CUDA_LIB="${COMPARE_PSI_CUDA_LIB:-/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda/lib64}"
srun /bin/bash "$repo/devtools/benchmarks/gpu_compare/payload.sh" "$mode" "$@" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
