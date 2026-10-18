#!/usr/bin/env bash
# Job-local runtime only. Arguments: new stage, original runtime.env, then argv.
set -euo pipefail
stage=$1; runtime_file=$2; shift 2
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
unset PYTHONPATH PYTHONHOME
P=/storage/project/r-cs207-0/awallace43
E=$P/miniconda/envs/p4cuest_sapt
DEPS=$P/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage
export CUDA_HOME=$P/software/toolchains/cuda13 CUDA_PATH=$P/software/toolchains/cuda13
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
# Preserve the helper's generated file verbatim. It is NOT the effective runtime.
source "$runtime_file"
effective_path="$P/software/cuest-0.2.50-a06e8148/lib/13:$P/software/libxc-7.1.2-cuda-sm80-89-90-120/lib64:$DEPS/lib:$E/lib"
export LD_LIBRARY_PATH="$effective_path"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
# Never eval stdout of another arm's executable. --psiapi is checked by the probe.
test -x "$stage/bin/psi4"
export PYTHONPATH=""
eval "$("$stage/bin/psi4" --psiapi)"
export LD_LIBRARY_PATH="$effective_path"
export MAJOR2_HELPER_RUNTIME_ENV="$runtime_file"
export MAJOR2_EXPECTED_STAGE="$stage"
exec "$@"
