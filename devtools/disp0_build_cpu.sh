#!/usr/bin/env bash
# Isolated reference build. Reads dependencies; never installs into the env.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
env="${DISP0_ENV:-$HOME/miniconda3/envs/p4_einv2}"
build="$root/build_disp0_cpu"
export PATH="$env/bin:$PATH"
export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
cmake -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$build/install" \
    -DCMAKE_PREFIX_PATH="$env" \
    -DCMAKE_C_COMPILER="$env/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$env/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_Fortran_COMPILER="$env/bin/x86_64-conda-linux-gnu-gfortran" \
    -DPython_EXECUTABLE="$env/bin/python" \
    -DBLAS_LIBRARIES="$env/lib/libmkl_rt.so" \
    -DLAPACK_LIBRARIES="$env/lib/libmkl_rt.so" \
    -DENABLE_Einsums=OFF \
    -DENABLE_OpenOrbitalOptimizer=OFF \
    -DENABLE_PCMSolver=OFF \
    -DENABLE_dkh=OFF -DENABLE_gdma=OFF -DENABLE_libefp=OFF \
    -DENABLE_ambit=OFF -DENABLE_ecpint=OFF
cmake --build "$build" --parallel "${DISP0_BUILD_THREADS:-6}"
