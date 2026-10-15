#!/usr/bin/env bash
# Fresh, isolated build of one frozen worktree for the cuEST XC response validation.
# Runs inside a (CPU) allocation; no GPU calls. Dependencies are read-only.
# Arguments: worktree, environment, commit, output, variant (host|cuda).
#   host: LibXC from the environment (host LibXC derivative route)
#   cuda: CUDA LibXC 7.1.2 (device derivative route), sm_90
set -euo pipefail
root=$(realpath "$1")
env=$(realpath "$2")
pin=$3
out=$(realpath "$4")
variant=$5
[[ "$variant" == host || "$variant" == cuda ]]
test -n "${SLURM_JOB_ID:-}"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
mkdir -p "$out"
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1
jobs="${SLURM_CPUS_PER_TASK:-8}"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
# The CUDA12 module injects CPATH. GCC demotes an -Icuda13 directory that
# duplicates -isystem cuda13, so CPATH silently wins despite CUDAToolkit_ROOT.
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
deps=/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage
software=/storage/project/r-cs207-0/awallace43/software
export CUDA_HOME="$software/toolchains/cuda13" CUDA_PATH="$software/toolchains/cuda13"
libxc="$env"
cc="$env/bin/x86_64-conda-linux-gnu-cc"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
cuda_args=()
if [[ "$variant" == cuda ]]; then
    libxc="$software/libxc-7.1.2-cuda"
    cc="$software/toolchains/conda-gcc"
    cxx="$software/toolchains/conda-g++"
    cuda_args=(
        "-DCMAKE_CUDA_COMPILER=$software/toolchains/cuda13/bin/nvcc"
        "-DCMAKE_CUDA_HOST_COMPILER=$cxx"
        "-DCMAKE_CUDA_ARCHITECTURES=90"
        "-DCMAKE_CUDA_FLAGS=-I$software/toolchains/cuda13/targets/x86_64-linux/include"
    )
    libxc_config="$libxc/lib64/cmake/Libxc"
    export LD_LIBRARY_PATH="$libxc/lib64:$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    libxc_config="$libxc/lib/cmake/Libxc"
    export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
{
    "$cc" --version
    "$cxx" --version
    test -x "$("$cc" -print-prog-name=cc1)"
    test -x "$("$cxx" -print-prog-name=cc1plus)"
    printf 'cc=%s\ncxx=%s\nvariant=%s\nlibxc=%s\n' "$cc" "$cxx" "$variant" "$libxc"
} > "$out/compiler.txt"
printf '%s\n' "$pin" > "$out/commit.txt"
build="$root/build_cpks_$variant"
mkdir -p "$build"
if [[ -e "$build/cpks-commit.txt" ]]; then
    test "$( < "$build/cpks-commit.txt")" = "$pin"
else
    # Refuse an unrelated existing build rather than repointing its cache.
    test ! -e "$build/CMakeCache.txt"
    printf '%s\n' "$pin" > "$build/cpks-commit.txt"
fi
"$env/bin/cmake" -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_C_COMPILER=$cc" "-DCMAKE_CXX_COMPILER=$cxx" \
    "-DCUDAToolkit_ROOT=$software/toolchains/cuda13" \
    "-DCMAKE_C_FLAGS=-I$software/toolchains/cuda13/targets/x86_64-linux/include -DPSI4_CUDA13_HEADER_ENV=1" \
    "-DCMAKE_CXX_FLAGS=-I$software/toolchains/cuda13/targets/x86_64-linux/include -DPSI4_CUDA13_HEADER_ENV=1" \
    -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--enable-new-dtags \
    "-DCMAKE_PREFIX_PATH=$libxc;$env;$deps" \
    "-DPython_EXECUTABLE=$env/bin/python" \
    "-DLAPACK_LIBRARIES=$env/lib/libmkl_rt.so" "-DLAPACK_INCLUDE_DIRS=$env/include" \
    "-DLibxc_DIR=$libxc_config" "-DLibint2_DIR=$env/lib/cmake/libint2" \
    "-DEinsums_DIR=$deps/share/cmake/Einsums" \
    -DENABLE_Einsums=ON -DENABLE_OPENMP=ON -DENABLE_cuEST=ON \
    -DENABLE_dkh=OFF -DENABLE_XHOST=OFF \
    "-DcuEST_ROOT=$env" "-DcuEST_INCLUDE_DIR=$env/include" \
    "-DcuEST_LIBRARY=$env/lib/libcuest.so" \
    -DCMAKE_INSIST_FIND_PACKAGE_cuEST=ON -Dpsi4_CMAKE_UNITY_BUILD=ON \
    "${cuda_args[@]}" > "$out/configure.log" 2>&1
inner="$build/psi4-core-prefix/src/psi4-core-build"
"$env/bin/cmake" --build "$build" -j "$jobs" > "$out/build.log" 2>&1
# The GPU response kernel is compiled only when the SDK provides the nonsymmetric density.
grep -q '^PSI4_CUEST_HAS_NONSYMMETRIC_XC:INTERNAL=1$' "$inner/CMakeCache.txt"
if [[ "$variant" == cuda ]]; then
    grep -q 'USING_Libxc_CUDA' "$inner/compile_commands.json"
else
    ! grep -q 'USING_Libxc_CUDA' "$inner/compile_commands.json"
fi
"$env/bin/python" "$root/devtools/saptdft_cuda_abi_probe.py" \
    --compile-database "$inner/compile_commands.json" --runtime "$env/lib/libcudart.so" \
    --output "$out/abi-after-build" > "$out/abi-after-build.log" 2>&1
# A CUDA 12 header can otherwise silently compile against a CUDA 13 runtime.
if readelf --dyn-syms --wide "$build"/stage/lib/psi4/core*.so | grep ' UND cudaGetDeviceProperties_v2'; then
    echo 'CUDA header/runtime mismatch: core still requires the CUDA 12 device-properties ABI' >&2
    exit 1
fi
# Host self-check of the shared per-point algebra (no GPU).
"$cxx" -std=c++17 -O2 -I "$root/psi4/src" "$root/devtools/cuest_xc_response_math_check.cc" \
    -o "$out/math_check"
"$out/math_check" > "$out/math_check.log" 2>&1
sha256sum "$build"/stage/lib/psi4/core*.so > "$out/core.sha256"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
printf '%s %s\n' "$pin" "$variant" > "$out/BUILD_OK"
