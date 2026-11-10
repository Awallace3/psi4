#!/usr/bin/env bash
# Executable proposal, not submission authority. One fresh CPU allocation only
# after a new explicit approval and pinned SDK/dependency input preflight.
set -euo pipefail
umask 077
P=/storage/project/r-cs207-0/awallace43
: "${DISP0_NEW_CPU_BUILD_APPROVAL_REFERENCE:?fresh approval required}"
: "${DISP0_SOURCE:?}" "${DISP0_PIN:?}" "${DISP0_RUN:?}" "${SLURM_JOB_ID:?}"
: "${DISP0_SDK_INPUT_MANIFEST:?}" "${DISP0_SDK_INPUT_SHA256:?}"
: "${DISP0_DEPENDENCY_LOCK:?}" "${DISP0_DEPENDENCY_LOCK_SHA256:?}"
source_root="$DISP0_SOURCE"
pin="$DISP0_PIN"
run="$DISP0_RUN"
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$(git -C "$source_root" rev-parse HEAD)" == "$pin" ]]
git -C "$source_root" diff HEAD --quiet
git -C "$source_root" diff --cached --quiet
git -C "$source_root" diff --exit-code 587e172a33afaadb00eb7f574a5089533f7fc755 "$pin" -- psi4/
[[ "$run" == "$P/runs/saptdft-cuest-disp0/revised-build-$pin-$SLURM_JOB_ID" ]]
[[ -z "${SLURM_JOB_GPUS:-}" && "${SLURM_CPUS_PER_TASK:?}" == 8 ]]
[[ "$(sha256sum "$DISP0_SDK_INPUT_MANIFEST" | awk '{print $1}')" == "$DISP0_SDK_INPUT_SHA256" ]]
[[ "$(sha256sum "$DISP0_DEPENDENCY_LOCK" | awk '{print $1}')" == "$DISP0_DEPENDENCY_LOCK_SHA256" ]]
mkdir "$run"
"$P/miniconda/bin/python" - "$run" "$DISP0_SDK_INPUT_MANIFEST" "$DISP0_DEPENDENCY_LOCK" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()
with (Path(sys.argv[1]) / "input-receipts.json").open("x") as stream:
    json.dump({name: {"path": str(Path(path).resolve()), "sha256": sha(path)}
               for name, path in zip(("sdk", "dependencies"), sys.argv[2:])}, stream, sort_keys=True)
PY
export TMPDIR="/storage/scratch1/3/awallace43/saptdft-cuest-disp0/revised-build-$SLURM_JOB_ID"
mkdir "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
export CONDA_PKGS_DIRS="$run/conda-pkgs" PIP_CACHE_DIR="$run/pip-cache" XDG_CACHE_HOME="$run/cache"
export CONDARC="$source_root/devtools/disp0/condarc"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CCACHE_DISABLE=1 SCCACHE_DISABLE=1
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT GCC_EXEC_PREFIX CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH \
      CUDA_ROOT CUDA_PATH CUDA_HOME CUDACXX CUDA_TOOLKIT_ROOT_DIR CUDAToolkit_ROOT \
      CUDAHOSTCXX CUDAFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS \
      CMAKE_PREFIX_PATH CMAKE_LIBRARY_PATH CMAKE_INCLUDE_PATH CMAKE_TOOLCHAIN_FILE \
      PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD LD_RUN_PATH LDFLAGS CFLAGS CXXFLAGS CPPFLAGS
env="$run/environment"
# Exact explicit input lock, no solver-driven version drift or shared env copy.
"$P/miniconda/bin/conda" create --yes --prefix "$env" --file "$DISP0_DEPENDENCY_LOCK"
"$P/miniconda/bin/conda" list --prefix "$env" --explicit > "$run/environment-explicit.txt"
export CONDA_PREFIX="$env" PATH="$env/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$env/lib"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
"$env/bin/python" - "$DISP0_SDK_INPUT_MANIFEST" "$run/cuda13" "$source_root" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[3]) / "devtools/disp0"))
from runtime_contract import prepare_sdk
manifest = json.loads(Path(sys.argv[1]).read_text())
prepare_sdk(manifest["source"], sys.argv[2], manifest)
PY
export CUDA_HOME="$run/cuda13" CUDA_PATH="$run/cuda13"
export PATH="$env/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$env/lib"
cc="$env/bin/x86_64-conda-linux-gnu-cc"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
[[ "$("$cxx" -dumpfullversion)" == 14.3.0 ]]
cc1="$(readlink -f "$("$cc" -print-prog-name=cc1)")"
cc1plus="$(readlink -f "$("$cxx" -print-prog-name=cc1plus)")"
[[ "$cc1" == "$env/"* && "$cc1plus" == "$env/"* && -x "$cc1" && -x "$cc1plus" ]]
sha256sum "$cc" "$cxx" "$cc1" "$cc1plus" > "$run/compiler-identities.txt"
"$cxx" --version > "$run/compiler-version.txt"
"$CUDA_HOME/bin/nvcc" --version > "$run/nvcc-version.txt"
cmake --version > "$run/cmake-version.txt"
sha256sum "$source_root/devtools/disp0/revised_einsums_hook.cmake" \
    "$source_root/devtools/disp0/runtime_contract.py" > "$run/configuration-source-sha256.txt"
cmake -S "$source_root/devtools/disp0/catch_config_probe" -B "$run/catch-config-probe" -G Ninja \
    -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_PREFIX_PATH="$env" -DCatch2_DIR="$env/lib/cmake/Catch2"
cmake -S "$source_root/devtools/disp0/apiary_config_probe" -B "$run/apiary-config-probe" -G Ninja \
    -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_PREFIX_PATH="$env"
git clone --no-checkout https://github.com/jturney/Einsums.git "$run/einsums-source"
git -C "$run/einsums-source" checkout --detach d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62
cmake -S "$run/einsums-source" -B "$run/einsums-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$run/einsums-stage" \
    -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_PREFIX_PATH="$env;$CUDA_HOME" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER= \
    -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_CUDA_COMPILER_LAUNCHER= \
    -DCMAKE_PROJECT_Einsums_INCLUDE="$source_root/devtools/disp0/revised_einsums_hook.cmake" \
    -DDISP0_SDK_CONTRACT="$CUDA_HOME/sdk-contract.json" \
    -DDISP0_SDK_CONTRACT_SHA256="$(sha256sum "$CUDA_HOME/sdk-contract.json" | awk '{print $1}')" \
    -DDISP0_DEPENDENCY_PREFIX="$env" -DCMAKE_MAKE_PROGRAM="$env/bin/ninja" \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
    -DEINSUMS_WITH_CXX_STANDARD=20 -DEINSUMS_WITH_CUDA=ON -DEINSUMS_WITH_CUDA_ARCHITECTURES=90 \
    -DEINSUMS_WITH_HIP=OFF -DEINSUMS_WITH_MPS=OFF -DEINSUMS_WITH_GPU_MOCK_DISCRETE=OFF \
    -DEINSUMS_WITH_MALLOC=system -DEINSUMS_BUILD_PYTHON=ON -DEINSUMS_PYBIND_NUM_TU=8 \
    -DPython_EXECUTABLE="$env/bin/python" -DPython_ROOT_DIR="$env" \
    -DCatch2_DIR="$env/lib/cmake/Catch2" -DLAPACK_LIBRARIES="$env/lib/libmkl_rt.so" \
    -DEINSUMS_WITH_PROFILER=ON -DEINSUMS_WITH_TESTS=OFF \
    -DEINSUMS_WITH_EXAMPLES=OFF -DEINSUMS_WITH_DOCUMENTATION=OFF
cmake --build "$run/einsums-build" --parallel 4 --verbose 2>&1 | tee "$run/einsums-build.log"
if grep -m 20 -E 'fatal error:|Error while processing' "$run/einsums-build.log"; then exit 1; fi
cmake --build "$run/einsums-build" --target PyEinsums PyEinsumsStubs --parallel 4 --verbose \
    2>&1 | tee -a "$run/einsums-build.log"
if grep -m 20 -E 'fatal error:|Error while processing' "$run/einsums-build.log"; then exit 1; fi
cmake --install "$run/einsums-build"
export DISP0_ENV="$env" DISP0_BUILD_DIR="$run/psi4-build" DISP0_BUILD_THREADS=8
export DISP0_VERBOSE_BUILD=1
cd "$source_root"
bash devtools/disp0_build_cpu.sh 2>&1 | tee "$run/psi4-build.log"
"$env/bin/python" "$source_root/devtools/disp0/build_audit.py" \
    --run "$run" --source "$source_root" --scratch "$TMPDIR" --out "$run/effective-build-audit.json"
export LD_LIBRARY_PATH="$run/einsums-stage/lib:$CUDA_HOME/lib64:$env/lib"
export PYTHONPATH="$run/einsums-stage/lib"
eval "$("$run/psi4-build/stage/bin/psi4" --psiapi)"
cd "$run"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$env/bin/python" \
    "$source_root/devtools/disp0/revised_runtime_probe.py" > "$run/runtime-provenance.json"
export DISP0_API_RUNTIME_PROOF=1
export PYTHONPATH="$PYTHONPATH:$source_root/devtools/disp0"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$env/bin/python" \
    "$source_root/devtools/disp0_einsums_api_probe.py" > "$run/api-probe.json"
unset DISP0_API_RUNTIME_PROOF
cd "$source_root"
export DISP0_PHYSICAL_PROCESS_PROOF=1
bash devtools/disp0_test_reference.sh \
    "$source_root/tests/pytests/test_saptdft_disp0.py" \
    "$source_root/tests/pytests/test_saptdft.py::test_einsum_terms" \
    "$source_root/tests/pytests/test_fisapt.py::test_fsapt_fdisp_blocked_matches_pair"
"$env/bin/python" - "$run" "$source_root" "$pin" <<'PY'
import sys
from pathlib import Path
run, source, pin = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(source / "devtools/disp0"))
from revised_admission import collect
from runtime_contract import write_new
xml = list((run / "psi4-build").glob("reference-check-*/results.xml"))
assert len(xml) == 1, xml
write_new(run / "preparation-result.json", collect(run, source, pin, xml[0]))
PY
printf 'REVISED_CPU_PREPARATION_ONLY_PENDING_CLEAN_JOB_EXIT\n'
