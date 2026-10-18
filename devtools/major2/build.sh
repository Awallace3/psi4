#!/usr/bin/env bash
# Initialize modules before the UNCHANGED per-arm helper. CUDA13's thin
# prefix has no cuobjdump; the pinned module CUDA12 tool is ELF/SASS inspection
# ONLY. Compiler/header/runtime selection stays CUDA13, with no PATH fallback.
set -euo pipefail
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
P=/storage/project/r-cs207-0/awallace43
export PATH="$P/miniconda/envs/p4cuest_sapt/bin:$P/software/toolchains/cuda13/bin:$PATH"
INSPECTOR=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/cuda-12.6.1-fgbhrk26npivybjj65zswagcf4zv6p2n/bin/cuobjdump
INSPECTOR_SHA=1a53d40230259b7904d068305c9c3b368746423e973df34fba55c5e2f1d35996
test -x "$INSPECTOR"
test "$(command -v cuobjdump)" = "$INSPECTOR"
actual_sha=$(sha256sum "$INSPECTOR")
test "${actual_sha%% *}" = "$INSPECTOR_SHA"
version=$("$INSPECTOR" --version)
[[ "$version" == *"V12.6.68"* ]]
printf 'ELF/SASS inspector ONLY: %s\nsha256=%s\n%s\n' "$INSPECTOR" "$INSPECTOR_SHA" "$version" >&2
exec "$@"
