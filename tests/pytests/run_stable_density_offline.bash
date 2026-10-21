#!/bin/bash
# Portable fresh-output validation of the explicit diagnostic only.
set -euo pipefail
if [[ $# != 1 ]]; then
    printf 'usage: PYTHON=python3 %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
ulimit -S -v 8388608; ulimit -H -v 8388608
ulimit -S -c 0; ulimit -H -c 0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PSIPATH
TESTS="$(dirname "$(realpath "$0")")"
exec "${PYTHON:-python3}" -B "$TESTS/run_stable_density_tests.py" "$1"
