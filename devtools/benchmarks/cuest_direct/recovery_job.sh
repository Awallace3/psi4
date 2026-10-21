#!/bin/bash
# Run only inside the separately authorized, serialized GPU allocation.
# Args: REPO COMMIT BUILD NEW_RUN DATA BUILD_JOB GPU QOS
set -euo pipefail
umask 027
repo=${1:?}; commit=${2:?}; build=${3:?}; run=${4:?}; data=${5:?}
build_job=${6:?}; gpu=${7:-H200}; qos=${8:-embers}
: "${GPU_COMPARE_CUDA12_RECEIPT:?absolute independently reviewed CUDA12 receipt}"
: "${GPU_COMPARE_CUDA12_RECEIPT_SHA256:?independent reviewed CUDA12 receipt SHA256}"
[[ "$repo" == /* && "$build" == /* && "$run" == /* && "$data" == /* ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
[[ "${SLURM_CPUS_PER_TASK:?}" == 8 ]]
base=/storage/project/r-cs207-0/awallace43
gpu_env="$base/miniconda/envs/gpu4pyscf_compare"
psi_env="$base/miniconda/envs/p4cuest_sapt"
source /etc/profile.d/z00_lmod_pace.sh
module load cuda/12.6.1
cuda12="${CUDA_HOME:-${CUDA_PATH:?}}/lib64"
export PYTHONNOUSERSITE=1 PYTHONPATH=""
unset LD_PRELOAD
export LD_LIBRARY_PATH="$gpu_env/lib:$cuda12"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export SLURM_JOB_QOS="$qos"
# Small direct-SCF temporary files: job-owned project storage, not failed FDDS
# Lustre scratch or the disposable private /tmp bind. Retain all evidence.
export TMPDIR="${run}-scratch"
mkdir "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" PYSCF_TMPDIR="$TMPDIR"
export CUEST_COMPARE_TMPDIR="$TMPDIR"
cd "$(dirname "$run")"
exec "$gpu_env/bin/python" "$repo/devtools/benchmarks/cuest_direct/recovery.py" \
    --repo "$repo" --commit "$commit" --build "$build" --output "$run" --data "$data" \
    --build-job "$build_job" --gpu "$gpu" --qos "$qos" \
    --gpu-env "$gpu_env" --psi-env "$psi_env" --cuda12-lib "$cuda12" \
    --psi-cuda-lib "$base/software/toolchains/cuda13/lib64" \
    --psi-xc-lib "$base/software/libxc-7.1.2-cuda/lib64"
