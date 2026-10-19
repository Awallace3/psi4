#!/usr/bin/env bash
# LOGIN SAFE: MODULE-ONLY reproduction, not arm.sh, --psiapi, imports or compile.
set -euo pipefail
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
unset PYTHONPATH PYTHONHOME
P=/storage/project/r-cs207-0/awallace43
E=$P/miniconda/envs/p4cuest_sapt
export CUDA_HOME=$P/software/toolchains/cuda13 CUDA_PATH=$P/software/toolchains/cuda13
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
exec "$E/bin/python" -B "$here/module_env_check.py"
