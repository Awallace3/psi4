#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${DISP0_SOURCE:?}" "${DISP0_PIN:?}" "${DISP0_JOB_RUN:?}" "${DISP0_DEPENDENCIES:?}" \
  "${DISP0_E2_PREFIX:?}" "${DISP0_CUDA_SOURCE:?}" "${DISP0_CUEST_ROOT:?}" "${SLURM_JOB_ID:?}"
source_root="$DISP0_SOURCE"
run="$DISP0_JOB_RUN"
env="$DISP0_DEPENDENCIES"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$DISP0_PIN" ]]
[[ -z "$(git -C "$source_root" status --porcelain)" ]]
[[ "${SLURM_CPUS_PER_TASK:?}" == 8 && -z "${SLURM_JOB_GPUS:-}" ]]
export TMPDIR="/storage/scratch1/3/awallace43/disp0-phoenix-validation-$SLURM_JOB_ID"
mkdir -m 0700 "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT GCC_EXEC_PREFIX CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH \
      LIBRARY_PATH PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD LD_RUN_PATH \
      CMAKE_PREFIX_PATH CMAKE_TOOLCHAIN_FILE CMAKE_LIBRARY_PATH CMAKE_INCLUDE_PATH \
      DISP0_API_RUNTIME_PROOF CFLAGS CXXFLAGS CPPFLAGS LDFLAGS \
      CUDA_ROOT CUDA_HOME CUDA_PATH CUDACXX CUDAHOSTCXX CUDAToolkit_ROOT \
      CUDA_TOOLKIT_ROOT_DIR CUDAFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS \
      OMP_PROC_BIND OMP_PLACES
export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$env/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
export CONDA_PREFIX="$env" MKL_INTERFACE_LAYER=LP64
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1 CCACHE_DISABLE=1 SCCACHE_DISABLE=1
python="$env/bin/python"
helper="$source_root/devtools/disp0/phoenix_validation"
cd "$run"
"$python" -B "$source_root/devtools/disp0/optimized_build/source_version_guard.py" \
    --source "$source_root" --out "$run/source-version"
"$python" -B "$helper/validate.py" --prepare
export CUDA_HOME="$run/cuda13" CUDA_PATH="$run/cuda13"
export PATH="$env/bin:$CUDA_HOME/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$DISP0_CUEST_ROOT/lib/13:$env/lib"
cc="$env/bin/x86_64-conda-linux-gnu-cc"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
[[ "$("$cxx" -dumpfullversion)" == 14.3.0 ]]
for pair in "$cc:cc1" "$cxx:cc1plus"; do
    tool="${pair%:*}"; name="${pair##*:}"
    frontend="$(readlink -f "$("$tool" -print-prog-name="$name")")"
    [[ "$frontend" == "$env/"* && -x "$frontend" ]]
done
"$cxx" -I"$CUDA_HOME/include" "$source_root/devtools/disp0/cuda_header_abi.cc" \
    -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart -o "$run/header-abi"
"$run/header-abi" | tee "$run/header-abi.txt"
cmake -S "$source_root" -B "$run/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$run/install" \
    -DCMAKE_PREFIX_PATH="$env;$CUDA_HOME" \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_C_FLAGS=-march=x86-64-v3 -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \
    -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_XHOST=OFF \
    -DPython_EXECUTABLE="$python" \
    -DBLAS_LIBRARIES="$env/lib/libmkl_rt.so" -DLAPACK_LIBRARIES="$env/lib/libmkl_rt.so" \
    -DLibxc_DIR="$env/lib/cmake/Libxc" -DLibint2_DIR="$env/lib/cmake/libint2" \
    -DCMAKE_INSIST_FIND_PACKAGE_Libxc=ON -DCMAKE_INSIST_FIND_PACKAGE_Libint2=ON \
    -DENABLE_cuEST=ON -DCMAKE_INSIST_FIND_PACKAGE_cuEST=ON \
    -DcuEST_ROOT="$DISP0_CUEST_ROOT" -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_INSTALL_RPATH="$CUDA_HOME/lib64;$DISP0_CUEST_ROOT/lib/13;$env/lib" \
    -DENABLE_Einsums=OFF -DENABLE_OpenOrbitalOptimizer=OFF \
    -DENABLE_PCMSolver=OFF -DENABLE_dkh=OFF -DENABLE_gdma=OFF \
    -DENABLE_libefp=OFF -DENABLE_ambit=OFF -DENABLE_ecpint=OFF \
    2>&1 | tee "$run/configure.log"
cmake --build "$run/build" --parallel 6 2>&1 | tee "$run/build.log"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$DISP0_E2_PREFIX/lib:$DISP0_CUEST_ROOT/lib/13:$env/lib"
export PYTHONPATH="$DISP0_E2_PREFIX/lib"
test -x "$run/build/stage/bin/psi4"
eval "$("$run/build/stage/bin/psi4" --psiapi)"
"$python" -B -m unittest discover -s "$source_root/devtools/disp0" -p 'test_*.py' -v \
    > "$run/cpu-guards.stdout" 2> "$run/cpu-guards.stderr"
"$python" -B -m unittest discover -s "$helper" -p 'test_*.py' -v \
    > "$run/job-guards.stdout" 2> "$run/job-guards.stderr"
"$python" -B "$source_root/devtools/disp0_einsums_api_probe.py" > "$run/api-probe.json"
"$python" -B "$helper/validate.py"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$DISP0_PIN" ]]
git -C "$source_root" diff HEAD --exit-code
