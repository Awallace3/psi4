#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${DISP0_SOURCE:?}" "${DISP0_PIN:?}" "${DISP0_JOB_RUN:?}" "${DISP0_GATE_ENV:?}" \
  "${DISP0_GATE_CONFIG:?}" "${DISP0_GATE_CONFIG_SHA256:?}" "${SLURM_JOB_ID:?}" \
  "${DISP0_CPU_SYSTEM:?}" "${DISP0_CPU_MEMORY:?}" "${DISP0_INPUTS:?}" "${DISP0_INPUTS_SHA256:?}"
[[ "$(git -C "$DISP0_SOURCE" rev-parse HEAD)" == "$DISP0_PIN" ]]
[[ -z "$(git -C "$DISP0_SOURCE" status --porcelain)" ]]
[[ "${SLURM_CPUS_PER_TASK:?}" == 24 && -z "${SLURM_JOB_GPUS:-}" ]]
base="${TMPDIR:-/scratch/}"
[[ -d "$base" && -w "$base" ]]
export TMPDIR="$base/disp0-cpu-$SLURM_JOB_ID"
mkdir -m 0700 "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD OMP_PROC_BIND OMP_PLACES
export PATH="$DISP0_GATE_ENV/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$DISP0_GATE_ENV/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 LANG=C LC_ALL=C
export CONDA_PREFIX="$DISP0_GATE_ENV" MKL_INTERFACE_LAYER=LP64
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1
cd "$DISP0_JOB_RUN"
"$DISP0_GATE_ENV/bin/python" -B \
    "$DISP0_SOURCE/devtools/disp0/phoenix_validation/validate_cpu_allocation.py"
exec "$DISP0_GATE_ENV/bin/python" -B \
    "$DISP0_SOURCE/devtools/disp0/phoenix_validation/compare_cpu.py" \
    --config "$DISP0_GATE_CONFIG" --config-sha256 "$DISP0_GATE_CONFIG_SHA256" \
    --inputs "$DISP0_INPUTS" --inputs-sha256 "$DISP0_INPUTS_SHA256" \
    --system "$DISP0_CPU_SYSTEM" --memory "$DISP0_CPU_MEMORY" --output "$DISP0_JOB_RUN"
