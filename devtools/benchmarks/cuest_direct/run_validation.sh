#!/bin/bash
# Usage: /bin/bash run_validation.sh ABS_REPO COMMIT ABS_BUILD ABS_OUTPUT [QOS] [COMPILED_COMMIT] [H200|H100] [full|rks]
# Defaults to embers. Any inferno submission requires explicit per-job approval.
# Build artifacts gate execution; a newer validation-only harness may use a pinned older core.
set -euo pipefail
umask 027
repo=${1:?absolute pinned source}
commit=${2:?commit}
build=${3:?absolute build output}
out=${4:?unique GPU validation output}
expected_qos=${5:-embers}
compiled_commit=${6:-$commit}
expected_gpu=${7:-H200}
suite=${8:-full}
slurm_bin=${GPU_COMPARE_SLURM_BIN:-/opt/slurm/current/bin}
[[ "$slurm_bin" == /* ]]
[[ "$expected_qos" == embers || "$expected_qos" == inferno ]]
[[ "$expected_gpu" == H200 || "$expected_gpu" == H100 ]]
[[ "$suite" == full || "$suite" == rks ]]
[[ "$repo" == /* && "$build" == /* && "$out" == /* ]]
[[ -r "$build/build-manifest.json" && ! -e "$out" ]]
export SLURM_JOB_QOS
SLURM_JOB_QOS=$("$slurm_bin/squeue" -h -j "${SLURM_JOB_ID:?}" -o '%q')
[[ "$SLURM_JOB_QOS" == "$expected_qos" ]] || { echo "Unexpected allocation QoS" >&2; exit 2; }
env_root=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
cuda=/storage/project/r-cs207-0/awallace43/software/toolchains/cuda13
xc=/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda
sdk=$("$env_root/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cuest_library"])' \
    "$build/build-manifest.json")
unset LD_PRELOAD
export PATH="$env_root/bin:$cuda/bin:$PATH" PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "$sdk"):$xc/lib64:$cuda/lib64:$env_root/lib"
export PYTHONPATH="$build/build/stage/lib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS=1
export TMPDIR="/storage/scratch1/3/awallace43/cuest-direct/${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"

child=
terminated=0
on_term() {
    terminated=1
    [[ -z "$child" ]] || kill -TERM -- "-$child" 2>/dev/null || true
}
trap on_term TERM
setsid /usr/bin/time -v "$env_root/bin/python" "$repo/devtools/benchmarks/cuest_direct/validate.py" \
    --repo "$repo" --commit "$commit" --compiled-commit "$compiled_commit" \
    --expected-gpu "$expected_gpu" --suite "$suite" --build "$build" --out "$out" &
child=$!
set +e
wait "$child"
status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
(( terminated == 0 )) || exit 143
exit "$status"
