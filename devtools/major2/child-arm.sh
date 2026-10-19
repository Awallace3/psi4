#!/usr/bin/env bash
# ONLY six real JK children. Inherited arm.sh is byte-identical; this owned copy
# adds strict numerical-ENV checks before any native --psiapi and before exec.
set -euo pipefail
H="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
P=/storage/project/r-cs207-0/awallace43
E=$P/miniconda/envs/p4cuest_sapt
check_control() {
  "$E/bin/python" -B "$H/workspace_control.py" --child-shell-check
}
check_control
stage=$1; runtime_file=$2; shift 2
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
unset PYTHONPATH PYTHONHOME
DEPS=$P/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage
export CUDA_HOME=$P/software/toolchains/cuda13 CUDA_PATH=$P/software/toolchains/cuda13
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
source "$runtime_file"
effective_path="$P/software/cuest-0.2.50-a06e8148/lib/13:$P/software/libxc-7.1.2-cuda-sm80-89-90-120/lib64:$DEPS/lib:$E/lib"
export LD_LIBRARY_PATH="$effective_path"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
test -x "$stage/bin/psi4"
export PYTHONPATH=""
check_control
eval "$("$stage/bin/psi4" --psiapi)"
export LD_LIBRARY_PATH="$effective_path"
export MAJOR2_HELPER_RUNTIME_ENV="$runtime_file"
export MAJOR2_EXPECTED_STAGE="$stage"
check_control
exec "$@"
