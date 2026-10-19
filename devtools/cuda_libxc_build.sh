#!/usr/bin/env bash
# Build CUDA LibXC 7.1.2 for several GPU architectures into a NEW prefix (never an existing one).
# Runs inside a CPU allocation; same configuration as software/libxc-7.1.2-cuda except the
# architecture list. Arguments: LibXC source, build directory, install prefix, output,
#                               architectures (CMake list, e.g. "80;89;90;120").
set -euo pipefail
src=$(realpath "$1")
build=$2
prefix=$3
out=$4
archs=$5
test -n "${SLURM_JOB_ID:-}"
test -f "$src/CMakeLists.txt"
# Refuse to touch an existing install or an unrelated build tree.
test ! -e "$prefix"
test ! -e "$build/CMakeCache.txt"
mkdir -p "$build" "$out"
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
software=/storage/project/r-cs207-0/awallace43/software
export PATH="$env/bin:$PATH"
jobs="${SLURM_CPUS_PER_TASK:-8}"
source /etc/profile.d/z00_lmod_pace.sh
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
{
    "$env/bin/nvcc" --version
    printf 'src=%s\nprefix=%s\narchs=%s\njob=%s\n' "$src" "$prefix" "$archs" "$SLURM_JOB_ID"
} > "$out/config.txt"
"$env/bin/cmake" -S "$src" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_FPIC=ON -DBUILD_TESTING=ON \
    -DENABLE_CUDA=ON -DENABLE_FORTRAN=OFF -DENABLE_PYTHON=OFF -DDISABLE_FHC=OFF \
    "-DCMAKE_CUDA_ARCHITECTURES=$archs" \
    "-DCMAKE_CUDA_COMPILER=$env/bin/nvcc" \
    "-DCMAKE_CUDA_HOST_COMPILER=$software/toolchains/conda-g++" \
    "-DCMAKE_C_COMPILER=$software/toolchains/conda-gcc" \
    -DCMAKE_INSTALL_LIBDIR=lib64 "-DCMAKE_INSTALL_PREFIX=$prefix" > "$out/configure.log" 2>&1
SECONDS=0
"$env/bin/cmake" --build "$build" -j "$jobs" > "$out/build.log" 2>&1
"$env/bin/cmake" --install "$build" > "$out/install.log" 2>&1
echo "build_seconds=$SECONDS" >> "$out/config.txt"
# Every requested architecture must be embedded as SASS.
module load cuda/12.6.1 > /dev/null 2>&1
library=$(readlink -f "$prefix/lib64/libxc.so")
cuobjdump -lelf "$library" | grep -oE 'sm_[0-9]+' | sort -u > "$out/sass.txt"
for arch in ${archs//;/ }; do
    grep -qx "sm_$arch" "$out/sass.txt"
done
sha256sum "$library" > "$out/libxc.sha256"
printf '%s\n' "$archs" > "$out/LIBXC_OK"
