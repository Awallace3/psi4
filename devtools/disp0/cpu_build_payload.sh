#!/usr/bin/env bash
# One-shot isolated dependency/build payload. Never resubmits or retries.
set -euo pipefail
umask 027
P=/storage/project/r-cs207-0/awallace43
source_root="${DISP0_SOURCE:?absolute immutable detached Psi4 source required}"
pin="${DISP0_PIN:?full immutable Psi4 commit required}"
run="${DISP0_RUN:?job-owned build root required}"
ein_pin=d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$source_root" == /* && "$run" == "$P/runs/saptdft-cuest-disp0/"* ]]
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$pin" ]]
git -C "$source_root" diff --quiet
git -C "$source_root" diff --cached --quiet
mkdir "$run"  # fail rather than reuse a partial/previous build
export XDG_CACHE_HOME="$run/cache" CONDA_PKGS_DIRS="$run/conda-pkgs" PIP_CACHE_DIR="$run/pip-cache"
export CONDARC="$source_root/devtools/disp0/condarc"
export TMPDIR="/storage/scratch1/3/awallace43/saptdft-cuest-disp0/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
export CUDA_HOME="$P/software/toolchains/cuda13" CUDA_PATH="$P/software/toolchains/cuda13"
env="$run/environment"
"$P/miniconda/bin/conda" env create --prefix "$env" --file "$source_root/devtools/disp0/environment.yml"
"$P/miniconda/bin/conda" list --prefix "$env" --explicit > "$run/environment-explicit.txt"
export PATH="$env/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$env/lib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cc="$env/bin/x86_64-conda-linux-gnu-cc"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
for entry in "$cc:cc1" "$cxx:cc1plus"; do
    compiler="${entry%:*}"
    frontend="$("$compiler" -print-prog-name="${entry##*:}")"
    [[ "$frontend" == /* && -x "$frontend" ]]
    "$compiler" -dumpfullversion
done
[[ "$("$cxx" -dumpfullversion)" == 14.3.0 ]]
nvcc_version="$("$CUDA_HOME/bin/nvcc" --version)"
printf '%s\n' "$nvcc_version" | tee "$run/nvcc-version.txt"
[[ "$nvcc_version" == *"V13.3.73"* ]]
"$cxx" "$source_root/devtools/disp0/cuda_header_abi.cc" -I"$CUDA_HOME/include" \
    -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib" -Wl,-rpath,"$CUDA_HOME/lib64" \
    -Wl,-rpath,"$CUDA_HOME/lib" -lcudart -o "$run/cuda_header_abi"
"$run/cuda_header_abi" | tee "$run/cuda-header-runtime.txt"
readelf -Ws "$run/cuda_header_abi" > "$run/cuda-header-symbols.txt"
ein_source="$run/einsums-source"
git clone --no-checkout https://github.com/jturney/Einsums.git "$ein_source"
git -C "$ein_source" checkout --detach "$ein_pin"
[[ "$(git -C "$ein_source" rev-parse HEAD)" == "$ein_pin" ]]
export DISP0_EIN_PREFIX="$run/einsums-stage"
cmake -S "$ein_source" -B "$run/einsums-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$DISP0_EIN_PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_PREFIX_PATH="$env" -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" -DEINSUMS_WITH_CXX_STANDARD=20 \
    -DEINSUMS_WITH_CUDA=ON -DEINSUMS_WITH_CUDA_ARCHITECTURES=90 \
    -DEINSUMS_WITH_HIP=OFF -DEINSUMS_WITH_MPS=OFF -DEINSUMS_WITH_GPU_MOCK_DISCRETE=OFF \
    -DEINSUMS_WITH_MALLOC=system -DEINSUMS_BUILD_PYTHON=ON \
    -DEINSUMS_PYBIND_NUM_TU=8 -DEINSUMS_INSTALL_PYMODDIR=naive \
    -DPython_EXECUTABLE="$env/bin/python" -DPython_ROOT_DIR="$env" \
    -DLAPACK_LIBRARIES="$env/lib/libmkl_rt.so" \
    -DEINSUMS_WITH_PROFILER=ON -DEINSUMS_WITH_TESTS=OFF \
    -DEINSUMS_WITH_EXAMPLES=OFF -DEINSUMS_WITH_DOCUMENTATION=OFF
cmake --build "$run/einsums-build" --parallel 4
cmake --install "$run/einsums-build"
export DISP0_ENV="$env" DISP0_BUILD_DIR="$run/psi4-build" DISP0_BUILD_THREADS=8
cd "$source_root"
bash devtools/disp0_build_cpu.sh
export LD_LIBRARY_PATH="$DISP0_EIN_PREFIX/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib:$env/lib"
export PYTHONPATH="$DISP0_EIN_PREFIX/lib"
eval "$("$DISP0_BUILD_DIR/stage/bin/psi4" --psiapi)"
cd "$run"
# Import failure on a CPU node is a real failed build gate, never masked.
"$env/bin/python" "$source_root/devtools/disp0/runtime_provenance.py" > "$run/runtime-provenance.json"
"$env/bin/python" "$source_root/devtools/disp0_einsums_api_probe.py" > "$run/api-probe.json"
cd "$source_root"
bash devtools/disp0_test_reference.sh \
    "$source_root/tests/pytests/test_saptdft_disp0.py" \
    "$source_root/tests/pytests/test_saptdft.py::test_einsum_terms" \
    "$source_root/tests/pytests/test_fisapt.py::test_fsapt_fdisp_blocked_matches_pair"
git diff --quiet
git diff --cached --quiet
printf 'CPU_BUILD_AND_API_PREPARATION_PASS source=%s einsums=%s\n' "$pin" "$ein_pin"
