#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${DISP0_SOURCE:?}" "${DISP0_PIN:?}" "${DISP0_JOB_RUN:?}" "${DISP0_GATE_ENV:?}" \
  "${DISP0_FULL_CONFIG:?}" "${DISP0_FULL_CONFIG_SHA256:?}" "${SLURM_JOB_ID:?}"
[[ "$(git -C "$DISP0_SOURCE" rev-parse HEAD)" == "$DISP0_PIN" ]]
[[ -z "$(git -C "$DISP0_SOURCE" status --porcelain)" ]]
[[ "${SLURM_CPUS_PER_TASK:?}" == 8 && -n "${SLURM_JOB_GPUS:-}" ]]
[[ "${SLURM_MEM_PER_NODE:?}" == 163840 ]]
export TMPDIR="/storage/scratch1/3/awallace43/disp0-full-h200-$SLURM_JOB_ID"
mkdir -m 0700 "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD OMP_PROC_BIND OMP_PLACES
export PATH="$DISP0_GATE_ENV/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$DISP0_GATE_ENV/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_DYNAMIC=FALSE
export CONDA_PREFIX="$DISP0_GATE_ENV" MKL_INTERFACE_LAYER=LP64
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1
cd "$DISP0_JOB_RUN"
exec "$DISP0_GATE_ENV/bin/python" -B \
    "$DISP0_SOURCE/devtools/disp0/phoenix_validation/run_full.py" \
    --config "$DISP0_FULL_CONFIG" --config-sha256 "$DISP0_FULL_CONFIG_SHA256" \
    --inputs "${DISP0_FULL_INPUTS:?}" --inputs-sha256 "${DISP0_FULL_INPUTS_SHA256:?}" \
    --release "${DISP0_FULL_RELEASE:?}" --release-sha256 "${DISP0_FULL_RELEASE_SHA256:?}" \
    --output "$DISP0_JOB_RUN"
