#!/bin/bash
set -eo pipefail
RUN=/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-timing/20260909-utilization-retry
PSI4_BUILD=/storage/project/r-cs207-0/awallace43/runs/psi4-grac/20260908T200831Z/build
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate p4cuest_sapt
unset GCC_ROOT GCCROOT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then export TMPDIR=/scratch; fi
export TMPDIR="$(mktemp -d "$TMPDIR/psi4-profile-${SLURM_JOB_ID}-XXXXXX")"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
eval "$("$PSI4_BUILD/stage/bin/psi4" --psiapi)"
cd "$RUN"
printf 'job=%s step=%s node=%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$(hostname)" > allocation.txt
command -v nsys > nsys-path.txt || true
exec "$CONDA_PREFIX/bin/python" profile-gpu.py
