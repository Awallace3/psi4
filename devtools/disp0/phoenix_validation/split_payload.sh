#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${DISP0_SOURCE:?}" "${DISP0_PIN:?}" "${DISP0_JOB_RUN:?}" "${DISP0_GATE_ENV:?}" \
  "${DISP0_SPLIT_THREADS:?}" "${DISP0_SPLIT_KIND:?}" "${SLURM_JOB_ID:?}"
[[ "$(git -C "$DISP0_SOURCE" rev-parse HEAD)" == "$DISP0_PIN" ]]
[[ -z "$(git -C "$DISP0_SOURCE" status --porcelain)" ]]
[[ "${SLURM_CPUS_PER_TASK:?}" == "$DISP0_SPLIT_THREADS" ]]
if [[ "$DISP0_SPLIT_KIND" == gpu ]]; then
    [[ "$DISP0_SPLIT_THREADS" == 8 && -n "${SLURM_JOB_GPUS:-}" ]]
    export TMPDIR="/storage/scratch1/3/awallace43/disp0-split-$SLURM_JOB_ID"
elif [[ "$DISP0_SPLIT_KIND" == cpu ]]; then
    [[ "$DISP0_SPLIT_THREADS" == 24 && -z "${SLURM_JOB_GPUS:-}" ]]
    export CUDA_VISIBLE_DEVICES=""
    # /scratch/$JOB_ID itself is site-owned, not a user-writable parent.
    # Use the scheduler's actual writable temporary directory; the Python
    # admission still requires a real node-local filesystem and free space.
    base="${TMPDIR:-${SLURM_TMPDIR:-}}"
    [[ -n "$base" && -d "$base" && -w "$base" ]]
    export TMPDIR="$base/disp0-split-$SLURM_JOB_ID"
else
    exit 1
fi
mkdir -m 0700 "$TMPDIR"
export TMPDIR="$(readlink -f -- "$TMPDIR")"
export DISP0_SPLIT_SCRATCH_CREATED="$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD OMP_PROC_BIND OMP_PLACES
export PATH="$DISP0_GATE_ENV/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$DISP0_GATE_ENV/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS="$DISP0_SPLIT_THREADS" MKL_NUM_THREADS="$DISP0_SPLIT_THREADS"
export OPENBLAS_NUM_THREADS="$DISP0_SPLIT_THREADS"
export CONDA_PREFIX="$DISP0_GATE_ENV" MKL_INTERFACE_LAYER=LP64
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1
cd "$DISP0_JOB_RUN"
driver="${DISP0_SPLIT_DRIVER:-split_jobs.py}"
[[ "$driver" == split_jobs.py || "$driver" == grouped_jobs.py || "$driver" == benchmark_suite.py ]]
exec "$DISP0_GATE_ENV/bin/python" -B \
    "$DISP0_SOURCE/devtools/disp0/phoenix_validation/$driver" --output "$DISP0_JOB_RUN"
