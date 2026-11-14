#!/bin/bash
# CPU-only native compile/link gate. This does NOT qualify SAPT0 admission.
# Usage: bash build_cuest0301.sh <owned root> <frozen source> <full commit>
set -euo pipefail
root=$(realpath "${1:?owned root}")
source_root=$(realpath "${2:?frozen source}")
pin=${3:?full commit}
env_root=$HOME/gits/psi4-disp0-cachy/environment-v2
sdk=$root/sdk
build=$root/build-native-0301-${pin:0:10}
stage=$root/install-native-0301-${pin:0:10}
[[ $(git -C "$source_root" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$source_root" status --porcelain --untracked-files=no) ]]
[[ ! -e "$build" && ! -e "$stage" ]]
[[ $(sha256sum "$sdk/lib/libcuest.so.0.3.0" | cut -d' ' -f1) == \
    cb810a15dff2d0c9c0c83cb56336bf80d77a74ace53f6e4d30921726679fc221 ]]
export PATH="$env_root/bin:/usr/bin:/bin"
export CONDA_PREFIX="$env_root" MATH_ROOT="$env_root"
export CUDA_HOME="$env_root" CUDA_PATH="$env_root" PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$sdk/lib:$env_root/lib"
# The outer superbuild invokes another Ninja build; constrain that child too.
export CMAKE_BUILD_PARALLEL_LEVEL=2
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH GCC_ROOT
export TMPDIR="$root/runs/native-build-scratch"
mkdir -p "$TMPDIR"
cxx="$env_root/bin/x86_64-conda-linux-gnu-c++"
"$cxx" -std=c++17 -Wall -Wextra -Werror -fsyntax-only -I"$sdk/include" \
    "$source_root/devtools/disp0/cuest0301_abi_probe.cc"
rp="$sdk/lib;$env_root/lib"
cmake -S "$source_root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$stage" \
    -DCMAKE_INSTALL_LIBDIR=lib -DPYMOD_INSTALL_LIBDIR=/python3.13/site-packages \
    -DCMAKE_C_COMPILER="$env_root/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_PREFIX_PATH="$env_root" \
    -DPython_EXECUTABLE="$env_root/bin/python" -DCUDAToolkit_ROOT="$env_root" \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="$env_root/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
    -DBLAS_TYPE=MKL -DLAPACK_TYPE=MKL -DCMAKE_INSIST_FIND_PACKAGE_Libxc=ON \
    -DLibxc_DIR="$env_root/lib/cmake/Libxc" \
    -DENABLE_cuEST=ON -DcuEST_ROOT="$sdk" \
    -DcuEST_INCLUDE_DIR="$sdk/include" -DcuEST_LIBRARY="$sdk/lib/libcuest.so.0.3.0" \
    -DENABLE_XHOST=OFF -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \
    -DCMAKE_BUILD_RPATH="$rp" -DCMAKE_INSTALL_RPATH="$rp" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Dpsi4_CMAKE_UNITY_BUILD_BATCH_SIZE=1
# Two workers avoid the previously observed desktop compiler OOM.
cmake --build "$build" -j 2
cmake --install "$build"
printf '%s\n' "$pin" > "$stage/BUILT_FROM"
printf '%s\n' "PASS_NATIVE_BUILD_ONLY: no GPU execution; SAPT0 admission remains unqualified"
