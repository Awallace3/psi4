#!/usr/bin/env bash
# Fresh, isolated build of one frozen worktree for the cuEST XC response validation.
# Runs inside a (CPU) allocation; no GPU calls. Dependencies are read-only.
# Arguments: worktree, environment, commit, output, variant (host|cuda).
#   host: LibXC from the environment (host LibXC derivative route)
#   cuda: CUDA LibXC 7.1.2 (device derivative route)
# Optional environment overrides (defaults reproduce the original sm_90 / cuEST 0.2.2 builds):
#   CPKS_CUDA_ARCHS  CUDA architectures (default 90; must be embedded in CPKS_CUDA_LIBXC)
#   CPKS_CUDA_LIBXC  CUDA LibXC prefix (default software/libxc-7.1.2-cuda)
#   CPKS_CUEST_SDK   standalone cuEST SDK (lib/13, include); default: the environment's libcuest
#   CPKS_BUILD_DIR   build directory name in the worktree (default build_cpks_<variant>)
# The output gets runtime.env (LD_LIBRARY_PATH and expected prefixes) for the run scripts.
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
archs="${CPKS_CUDA_ARCHS:-90}"
cuest_sdk="${CPKS_CUEST_SDK:-}"
cuest_lib_dir="$env/lib"
cuest_args=("-DcuEST_ROOT=$env" "-DcuEST_INCLUDE_DIR=$env/include" "-DcuEST_LIBRARY=$env/lib/libcuest.so")
if [[ -n "$cuest_sdk" ]]; then
    cuest_sdk=$(realpath "$cuest_sdk")
    cuest_lib_dir="$cuest_sdk/lib/13"
    test -f "$cuest_lib_dir/libcuest.so"
    cuest_args=("-DcuEST_ROOT=$cuest_sdk" "-DcuEST_INCLUDE_DIR=$cuest_sdk/include"
                "-DcuEST_LIBRARY=$cuest_lib_dir/libcuest.so")
fi
libxc="$env"
cc="$env/bin/x86_64-conda-linux-gnu-cc"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
cuda_args=()
if [[ "$variant" == cuda ]]; then
    libxc=$(realpath "${CPKS_CUDA_LIBXC:-$software/libxc-7.1.2-cuda}")
    cc="$software/toolchains/conda-gcc"
    cxx="$software/toolchains/conda-g++"
    cuda_flags="-I$software/toolchains/cuda13/targets/x86_64-linux/include"
    cuda_archs="$archs"
    if [[ "$archs" == *";"* ]]; then
        # The superbuild forwards CMAKE_CUDA_ARCHITECTURES to psi4-core without a
        # LIST_SEPARATOR, so a list arrives as its first entry only. Spell the
        # architectures out as -gencode flags (one argument) and switch the CMake list off.
        for arch in ${archs//;/ }; do
            cuda_flags+=" -gencode=arch=compute_$arch,code=sm_$arch"
        done
        cuda_archs=OFF
    fi
    cuda_args=(
        "-DCMAKE_CUDA_COMPILER=$software/toolchains/cuda13/bin/nvcc"
        "-DCMAKE_CUDA_HOST_COMPILER=$cxx"
        "-DCMAKE_CUDA_ARCHITECTURES=$cuda_archs"
        "-DCMAKE_CUDA_FLAGS=$cuda_flags"
    )
    libxc_config="$libxc/lib64/cmake/Libxc"
    libxc_lib="$libxc/lib64"
else
    libxc_config="$libxc/lib/cmake/Libxc"
    libxc_lib="$env/lib"
fi
# The SDK libcuest and the environment's share the soname libcuest.so.0: the SDK goes first.
runtime_path="$cuest_lib_dir:$libxc_lib:$env/lib"
export LD_LIBRARY_PATH="$runtime_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
{
    printf 'export LD_LIBRARY_PATH=%q\n' "$runtime_path"
    printf 'CPKS_EXPECTED_LIBXC=%q\nCPKS_EXPECTED_CUEST=%q\nCPKS_CUDA_ARCHS=%q\n' \
        "$libxc" "$cuest_lib_dir" "$archs"
} > "$out/runtime.env"
{
    "$cc" --version
    "$cxx" --version
    test -x "$("$cc" -print-prog-name=cc1)"
    test -x "$("$cxx" -print-prog-name=cc1plus)"
    printf 'cc=%s\ncxx=%s\nvariant=%s\nlibxc=%s\ncuest=%s\narchs=%s\n' "$cc" "$cxx" "$variant" "$libxc" \
        "$cuest_lib_dir" "$archs"
} > "$out/compiler.txt"
printf '%s\n' "$pin" > "$out/commit.txt"
build="$root/${CPKS_BUILD_DIR:-build_cpks_$variant}"
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
    "-DCMAKE_PREFIX_PATH=$libxc;${cuest_sdk:+$cuest_sdk;}$env;$deps" \
    "-DPython_EXECUTABLE=$env/bin/python" \
    "-DLAPACK_LIBRARIES=$env/lib/libmkl_rt.so" "-DLAPACK_INCLUDE_DIRS=$env/include" \
    "-DLibxc_DIR=$libxc_config" "-DLibint2_DIR=$env/lib/cmake/libint2" \
    "-DEinsums_DIR=$deps/share/cmake/Einsums" \
    -DENABLE_Einsums=ON -DENABLE_OPENMP=ON -DENABLE_cuEST=ON \
    -DENABLE_dkh=OFF -DENABLE_XHOST=OFF \
    "${cuest_args[@]}" \
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
# Device code of the core must exist for every requested architecture.
if [[ "$variant" == cuda ]]; then
    cuobjdump -lelf "$build"/stage/lib/psi4/core*.so | grep -oE 'sm_[0-9]+' | sort -u > "$out/core_sass.txt"
    for arch in ${archs//;/ }; do
        grep -qx "sm_$arch" "$out/core_sass.txt"
    done
fi
# Host self-check of the shared per-point algebra (no GPU).
"$cxx" -std=c++17 -O2 -I "$root/psi4/src" "$root/devtools/cuest_xc_response_math_check.cc" \
    -o "$out/math_check"
"$out/math_check" > "$out/math_check.log" 2>&1
# The staged core must resolve the intended cuEST and LibXC under runtime.env.
ldd "$build"/stage/lib/psi4/core*.so > "$out/ldd.txt"
grep -q "libcuest.so.0 => $cuest_lib_dir/" "$out/ldd.txt"
grep -qE "libxc.so.[0-9]+ => $libxc/" "$out/ldd.txt"
sha256sum "$build"/stage/lib/psi4/core*.so > "$out/core.sha256"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
printf '%s %s\n' "$pin" "$variant" > "$out/BUILD_OK"
