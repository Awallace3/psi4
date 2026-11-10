#!/usr/bin/env bash
# CPU preparation only. CUDA enabled at compile time is NOT device validation.
set -euo pipefail
umask 077
P=/storage/project/r-cs207-0/awallace43
: "${DISP0_OPT_SOURCE:?}" "${DISP0_OPT_PIN:?}" "${DISP0_OPT_RUN:?}" "${SLURM_JOB_ID:?}"
source_root="$DISP0_OPT_SOURCE"
run="$DISP0_OPT_RUN"
step_suffix=
if [[ "${DISP0_OPT_INTERACTIVE:-0}" == 1 ]]; then
    [[ "${SLURM_STEP_ID:?}" =~ ^[0-9]+$ ]]
    step_suffix="-$SLURM_STEP_ID"
fi
E2=06404fa69ce9346443d7d4f19a250f6db9f0f62c
SCIENCE=587e172a33afaadb00eb7f574a5089533f7fc755
lock="$P/runs/saptdft-cuest-disp0/build-$SCIENCE-14196330/environment-explicit.txt"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$DISP0_OPT_PIN" ]]
[[ -z "$(git -C "$source_root" status --porcelain)" ]]
git -C "$source_root" diff --exit-code "$SCIENCE" HEAD -- psi4/
[[ "$run" == "$P/runs/saptdft-cuest-disp0/opt064-$DISP0_OPT_PIN-$SLURM_JOB_ID$step_suffix" ]]
[[ "${SLURM_CPUS_PER_TASK:?}" == 8 && -z "${SLURM_JOB_GPUS:-}" ]]
mkdir -m 0700 "$run"
export TMPDIR="/storage/scratch1/3/awallace43/disp0-opt064-$SLURM_JOB_ID$step_suffix"
mkdir -m 0700 "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
export CONDA_PKGS_DIRS="$run/conda-pkgs" PIP_CACHE_DIR="$run/pip-cache" XDG_CACHE_HOME="$run/cache"
export CONDARC="$source_root/devtools/disp0/condarc"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CCACHE_DISABLE=1 SCCACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=""
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT GCC_EXEC_PREFIX CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH \
    CUDA_ROOT CUDA_PATH CUDA_HOME CUDACXX CUDA_TOOLKIT_ROOT_DIR CUDAToolkit_ROOT CUDAHOSTCXX \
    CUDAFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CMAKE_PREFIX_PATH CMAKE_LIBRARY_PATH \
    CMAKE_INCLUDE_PATH CMAKE_TOOLCHAIN_FILE PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD \
    LD_RUN_PATH LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
unset OMP_PROC_BIND OMP_PLACES DISP0_API_RUNTIME_PROOF
export PATH="$P/miniconda/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$P/miniconda/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$source_root"
# Real source history/tags are build inputs. Fail before costly dependency work
# rather than install an undefined-version Psi4 that refuses its own import.
"$P/miniconda/bin/python" -B devtools/disp0/optimized_build/source_version_guard.py \
    --source "$source_root" --out "$run/psi4-source-version"
# Independent NEW input preparation within THIS build job; old input attempt
# remains UNKNOWN/RESERVED and is neither read as admission nor replayed.
timeout --signal=TERM --kill-after=5s 110s "$P/miniconda/bin/python" -B \
    devtools/disp0/optimized_build/prepare_inputs.py \
    --source "$P/software/toolchains/cuda13" --destination "$run/private-inputs" \
    --vendor-environment "$P/miniconda/envs/p4cuest_sapt" --lock "$lock" \
    --audit-log "$run/sdk-loader-audit.jsonl"
"$P/miniconda/bin/python" -B - "$run" "$source_root" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]) / "devtools/disp0"))
from runtime_contract import prepare_sdk
run = Path(sys.argv[1])
manifest = json.loads((run / "private-inputs/sdk-manifest.json").read_text())
prepare_sdk(manifest["source"], run / "cuda13", manifest)
PY
env="$run/environment"
clang="$run/clang23"
# Runtime dependency lock is unchanged. LLVM23 is isolated BUILD-ONLY tooling
# for Apiary1.2; no alteration of any original/shared environment or lock.
"$P/miniconda/bin/conda" create -y -p "$env" --file "$lock"
"$P/miniconda/bin/conda" create -y -p "$clang" --override-channels -c conda-forge \
    clangdev=23.1.2 clangxx=23.1.2 llvmdev=23.1.2
"$P/miniconda/bin/conda" list -p "$env" --explicit > "$run/environment-explicit.txt"
"$P/miniconda/bin/conda" list -p "$clang" --explicit > "$run/clang23-explicit.txt"
export CONDA_PREFIX="$env" PATH="$env/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$env/lib"
cc="$env/bin/x86_64-conda-linux-gnu-cc"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
[[ "$("$cxx" -dumpfullversion)" == 14.3.0 ]]
for tool in "$cc" "$cxx"; do
    "$tool" --version
done > "$run/compiler-versions.txt"
for pair in "$cc:cc1" "$cxx:cc1plus"; do
    tool="${pair%:*}"; name="${pair##*:}"
    actual="$(readlink -f "$("$tool" -print-prog-name="$name")")"
    [[ "$actual" == "$env/"* && -x "$actual" ]]
    sha256sum "$actual"
done > "$run/compiler-frontends.txt"
export CUDA_HOME="$run/cuda13" CUDA_PATH="$run/cuda13"
export PATH="$env/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$env/lib"
"$CUDA_HOME/bin/nvcc" --version > "$run/nvcc-version.txt"
cmake --version > "$run/cmake-version.txt"
"$cxx" -I"$CUDA_HOME/include" "$source_root/devtools/disp0/cuda_header_abi.cc" \
    -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart -o "$run/header-abi"
"$run/header-abi" > "$run/header-abi.txt"
readelf -d "$run/header-abi" > "$run/header-abi-elf.txt"
git clone --no-checkout https://github.com/Awallace3/Einsums.git "$run/einsums-source"
git -C "$run/einsums-source" checkout --detach "$E2"
[[ "$(git -C "$run/einsums-source" rev-parse HEAD)" == "$E2" ]]
git -C "$run/einsums-source" diff HEAD --quiet
# Explicit architecture avoids build-node device discovery. Existing upstream
# install rules supply the WHOLE Python package; no d17 packaging overlay.
CONDA_PREFIX="$clang" cmake -S "$run/einsums-source" -B "$run/einsums-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$run/einsums-stage" -DCMAKE_INSTALL_LIBDIR=lib \
    -DEINSUMS_INSTALL_PYMODDIR=lib -DCMAKE_PREFIX_PATH="$env;$CUDA_HOME" \
    -DClang_DIR="$clang/lib/cmake/clang" -DLLVM_DIR="$clang/lib/cmake/llvm" \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
    -DCMAKE_CUDA_FLAGS="-I$CUDA_HOME/include" \
    -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_CUDA_COMPILER_LAUNCHER= \
    -DEINSUMS_WITH_CCACHE=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEINSUMS_WITH_CXX_STANDARD=20 \
    -DCMAKE_PROJECT_Einsums_INCLUDE="$source_root/devtools/disp0/optimized_build/install_hook.cmake" \
    -DDISP0_SDK_CONTRACT="$CUDA_HOME/sdk-contract.json" \
    -DDISP0_DEPENDENCY_PREFIX="$env" \
    -DEINSUMS_WITH_CUDA=ON -DEINSUMS_WITH_CUDA_ARCHITECTURES=90 \
    -DEINSUMS_WITH_HIP=OFF -DEINSUMS_WITH_MPS=OFF -DEINSUMS_WITH_GPU_MOCK_DISCRETE=OFF \
    -DEINSUMS_WITH_FULL_RPATH=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF \
    -DEINSUMS_WITH_MALLOC=mimalloc -DEINSUMS_FFT_LIBRARY=mkl \
    -DLAPACK_LIBRARIES="$env/lib/libmkl_rt.so" \
    -DEINSUMS_SIMD_NATIVE_ARCH=OFF -DEINSUMS_WITH_SIMD_DISPATCH=ON \
    -DEINSUMS_BUILD_PYTHON=ON -DEINSUMS_PYBIND_NUM_TU=8 \
    -DPython_EXECUTABLE="$env/bin/python" -DPython_ROOT_DIR="$env" \
    -DEINSUMS_WITH_TESTS=OFF -DEINSUMS_WITH_TESTS_UNIT=OFF \
    -DEINSUMS_WITH_EXAMPLES=OFF -DEINSUMS_WITH_DOCUMENTATION=OFF \
    -DEINSUMS_WITH_PROFILER=ON 2>&1 | tee "$run/einsums-configure.log"
# No blanket -k0/ignored build errors. The install requires all default outputs.
CONDA_PREFIX="$clang" cmake --build "$run/einsums-build" --parallel 4 --verbose 2>&1 | tee "$run/einsums-build.log"
if grep -m 20 -E 'fatal error:|Error while processing' "$run/einsums-build.log"; then exit 1; fi
cmake --install "$run/einsums-build" 2>&1 | tee "$run/einsums-install.log"
export DISP0_ENV="$env" DISP0_BUILD_DIR="$run/psi4-build" DISP0_BUILD_THREADS=8 DISP0_VERBOSE_BUILD=1
bash devtools/disp0_build_cpu.sh 2>&1 | tee "$run/psi4-build.log"
"$env/bin/python" -B "$source_root/devtools/disp0/build_audit.py" \
    --run "$run" --source "$source_root" --scratch "$TMPDIR" --out "$run/effective-build-audit.json"
export LD_LIBRARY_PATH="$run/einsums-stage/lib:$CUDA_HOME/lib64:$env/lib"
export PYTHONPATH="$run/einsums-stage/lib"
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1
eval "$("$run/psi4-build/stage/bin/psi4" --psiapi)"
cd "$run"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$env/bin/python" -B \
    "$source_root/devtools/disp0/optimized_build/proof.py" > "$run/installed-proof.json"
export PYTHONPATH="$PYTHONPATH:$source_root/devtools/disp0"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$env/bin/python" -B \
    "$source_root/devtools/disp0_einsums_api_probe.py" > "$run/api-probe.json"
printf 'CPU_BUILD_AND_INSTALLED_API_ONLY_NO_DEVICE_CORRECTNESS\n'
