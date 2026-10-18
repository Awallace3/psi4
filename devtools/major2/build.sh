#!/usr/bin/env bash
# Initialize modules once before the UNCHANGED per-arm helper; make CUDA13 tools
# (not module CUDA12 cuobjdump) first even after module initialization.
set -euo pipefail
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
P=/storage/project/r-cs207-0/awallace43
export PATH="$P/miniconda/envs/p4cuest_sapt/bin:$P/software/toolchains/cuda13/bin:$PATH"
test "$(command -v cuobjdump)" = "$P/software/toolchains/cuda13/bin/cuobjdump"
exec "$@"
