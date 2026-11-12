#!/usr/bin/env bash
set -eo pipefail
root=$1
campaign=$2
commit=$3
test "$(git -C "$root" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$root" status --porcelain)"
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate p4qca
set -u
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH PYTHONPATH PSI4_DATADIR
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CONDA_PREFIX/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CMAKE_BUILD_PARALLEL_LEVEL=8
runtime=/storage/scratch1/3/awallace43/psi4-pr44-283ae89a32-embers-v1/runtime-pydantic2
build="$campaign/build"
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then export TMPDIR=/scratch; fi
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
cmake -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DPython_EXECUTABLE="$runtime/bin/python" \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
    -DENABLE_XHOST=OFF -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=generic" \
    -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=generic" \
    -DCMAKE_INSTALL_PREFIX="$campaign/install" \
    -DCMAKE_UNITY_BUILD=ON -DCMAKE_UNITY_BUILD_BATCH_SIZE=8 \
    -DENABLE_Einsums=OFF -DBUILD_TESTING=ON \
    -DCMAKE_INSIST_FIND_PACKAGE_Libint2=ON -DCMAKE_INSIST_FIND_PACKAGE_Libxc=ON \
    -DCMAKE_INSIST_FIND_PACKAGE_qcelemental=ON -DCMAKE_INSIST_FIND_PACKAGE_qcengine=ON
cmake --build "$build" --parallel 8
export PR44_MEMORY_TRACE=1
ctest --test-dir "$build" -R '^sapt-dft(1|2|-fdds-ooc)$' --output-on-failure -j 1
PYTHONPATH="$build/stage/lib" PSI4_DATADIR="$build/stage/share/psi4" \
    "$runtime/bin/python" "$root/experiments/pr44/validate_build.py" "$build" "$commit"
