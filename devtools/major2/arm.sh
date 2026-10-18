#!/usr/bin/env bash
# Job-local runtime only. Arguments: new stage, then argv; no package overlays.
set -euo pipefail
stage=$1; shift
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
unset PYTHONPATH PYTHONHOME
P=/storage/project/r-cs207-0/awallace43
E=$P/miniconda/envs/p4cuest_sapt
export CUDA_HOME=$P/software/toolchains/cuda13 CUDA_PATH=$P/software/toolchains/cuda13
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$P/software/cuest-0.2.50-a06e8148/lib/13:$P/software/libxc-7.1.2-cuda-sm80-89-90-120/lib64:$E/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
# Never eval stdout of another arm's executable. --psiapi is checked by the probe.
test -x "$stage/bin/psi4"
eval "$("$stage/bin/psi4" --psiapi)"
export MAJOR2_EXPECTED_STAGE="$stage"
exec "$@"
