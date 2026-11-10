#!/bin/bash
# Whole candidate install using this private root's existing configured build.
# Preserve install/ (the measured baseline); never overlay selected modules.
set -euo pipefail
ROOT=$(realpath "${1:?private root}")
PIN=${2:?full clean Git pin}
[[ $PIN =~ ^[0-9a-f]{40}$ ]]
[[ $(git -C "$ROOT/source" rev-parse HEAD) == "$PIN" ]]
[[ -z $(git -C "$ROOT/source" status --porcelain) ]]
PREFIX="$ROOT/install-owned-${PIN:0:10}"
[[ ! -e $PREFIX ]]
ENV="$HOME/gits/psi4-disp0-cachy/environment-v2"
export PATH="$ENV/bin:/usr/bin:/bin" CONDA_PREFIX="$ENV" MATH_ROOT="$ENV"
export CUDA_HOME="$ENV" CUDA_PATH="$ENV" CMAKE_BUILD_PARALLEL_LEVEL=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
INNER="$ROOT/build/psi4-core-prefix/src/psi4-core-build"
[[ -f $ROOT/build/CMakeCache.txt && -f $INNER/CMakeCache.txt ]]
"$ENV/bin/python" -B -m unittest discover -s "$ROOT/source/devtools/disp0" \
    -p 'test_cuest_owned_runtime*.py'
"$ENV/bin/python" -B -m unittest discover -s "$ROOT/source/devtools/disp0" \
    -p 'test_cuda_owned_arguments.py'
cmake -S "$ROOT/source" -B "$ROOT/build" -DCMAKE_INSTALL_PREFIX="$PREFIX"
# ExternalProject stamps can otherwise skip Python/configured-version changes.
cmake -S "$ROOT/source/psi4" -B "$INNER"
cmake --build "$INNER" --parallel 1
cmake --install "$INNER"
cmake --build "$ROOT/build" --parallel 1
cmake --install "$ROOT/build"
[[ $(git -C "$ROOT/source" rev-parse HEAD) == "$PIN" ]]
[[ -z $(git -C "$ROOT/source" status --porcelain) ]]
printf '%s\n' "$PIN" > "$PREFIX/BUILT_FROM"
printf 'BUILD_OK %s %s\n' "$PIN" "$PREFIX"
