#!/bin/bash
# Run on a compute node under explicit QoS, not on a login node.
# Usage: /bin/bash build_grid_export.sh ABS_REPO COMMIT ABS_RUN embers|inferno ABS_CUEST_SDK ABS_CUDA_RUNTIME_ROOT
# inferno is permitted only for a specifically user-authorized submission.
# The historical CUDA-LibXC stage is read-only and is never rebuilt here.
set -euo pipefail
repo=${1:?absolute comparison worktree}
commit=${2:?pinned commit}
run=${3:?absolute unique build run}
expected_qos=${4:-embers}
[[ "$expected_qos" == embers || "$expected_qos" == inferno ]]
[[ "$repo" == /* && "$run" == /* && "$run" != "$repo" ]]
export SLURM_JOB_QOS
SLURM_JOB_QOS=$(squeue -h -j "${SLURM_JOB_ID:?}" -o '%q')
[[ "$SLURM_JOB_QOS" == "$expected_qos" ]] || { echo "Unexpected allocation QoS" >&2; exit 2; }
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
[[ ! -e "$run/build" && ! -e "$run/build-manifest.json" && ! -e "$run/cuest-sdk-identity.json" ]]

env_root=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
sdk=${5:?explicit canonical cuEST 0.2.50 SDK required}
[[ "$sdk" == /* && -r "$sdk/include/cuest.h" ]]
sdk_lib="$sdk/lib/13"
test -r "$sdk_lib/libcuest.so"
export CUEST_SELECTED_SDK="$sdk" CUEST_SELECTED_LIBRARY="$sdk_lib/libcuest.so"
toolchains=/storage/project/r-cs207-0/awallace43/software/toolchains
cuda="$toolchains/cuda13"
# Toolkit paths are symlinks into the dependency environment. Admission binds
# resolved CUDA files, not the toolkit directory's version-like label.
cuda_runtime=${6:?explicit resolved CUDA runtime closure prefix required}
[[ "$cuda_runtime" == /* && -d "$cuda_runtime" ]]
xc=/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda
source /etc/profile.d/z00_lmod_pace.sh
module load cmake
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
unset LD_PRELOAD
export PATH="$env_root/bin:$cuda/bin:$PATH" PYTHONNOUSERSITE=1
export CUDA_HOME="$cuda" CUDA_PATH="$cuda"
export LD_LIBRARY_PATH="$sdk_lib:$xc/lib64:$cuda/lib64:$env_root/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/storage/scratch1/3/awallace43/gpu-dft-grid-build/${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR" "$run"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
"$env_root/bin/python" "$repo/devtools/benchmarks/gpu_compare/cuest_sdk.py" \
    --sdk "$sdk" --output "$run/cuest-sdk-identity.json"

cc="$toolchains/conda-gcc"
cxx="$toolchains/conda-g++"
test -x "$("$cc" -print-prog-name=cc1)"
test -x "$("$cxx" -print-prog-name=cc1plus)"
cmake -S "$repo" -B "$run/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_CUDA_HOST_COMPILER="$cxx" -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
    -DCUDAToolkit_ROOT="$cuda" -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_PREFIX_PATH="$sdk;$env_root;$xc" \
    -DPython_EXECUTABLE="$env_root/bin/python" \
    -DLibint2_DIR="$env_root/lib/cmake/libint2" \
    -DLibxc_DIR="$xc/lib64/cmake/Libxc" \
    -DcuEST_ROOT="$sdk" -DcuEST_CUDA_RUNTIME_ROOT="$cuda_runtime" \
    -DcuEST_INCLUDE_DIR="$sdk/include" \
    -DcuEST_LIBRARY="$sdk_lib/libcuest.so" -DENABLE_cuEST=ON \
    -DBLAS_LIBRARIES="$env_root/lib/libmkl_rt.so" \
    -DLAPACK_LIBRARIES="$env_root/lib/libmkl_rt.so" \
    -DCMAKE_INSTALL_PREFIX="$run/install" \
    -DENABLE_ambit=OFF -DENABLE_dkh=OFF -DENABLE_BrianQC=OFF

# Background child + wait lets SLURM's TERM interrupt compilation promptly.
child=
terminated=0
on_term() {
    terminated=1
    [[ -z "$child" ]] || kill -TERM -- "-$child" 2>/dev/null || true
}
trap on_term TERM
setsid cmake --build "$run/build" --parallel "${SLURM_CPUS_PER_TASK:?}" &
child=$!
set +e
wait "$child"
status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
(( terminated == 0 )) || exit 143
(( status == 0 )) || exit "$status"

export PYTHONPATH="$run/build/stage/lib"
"$env_root/bin/python" - "$repo" "$commit" "$run" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import psi4

repo, commit, run = sys.argv[1:]
sys.path.insert(0, str(Path(repo) / "devtools/benchmarks/gpu_compare"))
from cuest_sdk import checked_provider, cuda_linkage_identity, verify_sdk
run = Path(run)
package = run / "build/stage/lib/psi4"
assert Path(psi4.__file__).resolve().parent == package.resolve()
assert hasattr(psi4.core.VBase, "export_cuest_grid")
assert hasattr(psi4.core.VBase, "cuest_grid_points")
checked_provider(psi4.core)  # No hidden handle initialization or version call.
core = next(package.glob("core*.so"))
linkage = subprocess.check_output(["ldd", str(core)], text=True)
dynamic = subprocess.check_output(["readelf", "-d", str(core)], text=True)
assert "not found" not in linkage, linkage
xc_line = next(line for line in linkage.splitlines() if line.strip().startswith("libxc.so"))
assert "/software/libxc-7.1.2-cuda/" in xc_line, linkage
assert "libcudart.so.13" in linkage, linkage
sdk = Path(os.environ["CUEST_SELECTED_SDK"]).resolve()
library = Path(os.environ["CUEST_SELECTED_LIBRARY"]).resolve()
cuest_line = next(line for line in linkage.splitlines() if line.strip().startswith("libcuest.so"))
assert Path(cuest_line.split()[2]).resolve() == library, linkage
sdk_identity = verify_sdk(sdk)
assert sdk_identity == json.loads((run / "cuest-sdk-identity.json").read_text()), "SDK changed during build"
for candidate in (run / "build").rglob("CMakeCache.txt"):
    values = dict(line.split("=", 1) for line in candidate.read_text().splitlines()
                  if line.startswith(("cuEST_INCLUDE_DIR:", "cuEST_LIBRARY:")) and "=" in line)
    for key, value in values.items():
        target = sdk / "include" if key.startswith("cuEST_INCLUDE_DIR:") else library
        assert Path(value).resolve() == target, (candidate, key, value)
cache = (run / "build/psi4-core-prefix/src/psi4-core-build/CMakeCache.txt")
if not cache.exists():
    matches = list((run / "build").rglob("CMakeCache.txt"))
    direct_available = any("cuEST_HAS_DIRECT_JK:INTERNAL=1" in p.read_text() for p in matches)
else:
    direct_available = "cuEST_HAS_DIRECT_JK:INTERNAL=1" in cache.read_text()
assert direct_available, "Strict build must retain compiled/linked four-center direct JK"
# Keep comparison SDK JSON separate from the CMake contract's identity schema.
sys.path.insert(0, str(Path(repo) / "cmake"))
from cuest_sdk_contract import compare_identity, snapshot_core, validate_sdk
cmake_receipts = sorted((run / "build").rglob("cuest-sdk-identity.json"))
assert len(cmake_receipts) == 2, ("outer/inner admission receipts required", cmake_receipts)
cmake_sdk = json.loads(cmake_receipts[0].read_text())
for path in cmake_receipts:
    compare_identity(json.loads(path.read_text()), cmake_sdk)
compare_identity(validate_sdk(cmake_sdk["sdk_root"], cmake_sdk["cuda_major"],
                              cmake_sdk["cuda_root"], cmake_sdk["cudart"]), cmake_sdk)
header_receipts = list((run / "build").rglob("cuest-header-dependencies.json"))
assert len(header_receipts) == 1, ("consumed-header audit required", header_receipts)
compiled_headers = json.loads(header_receipts[0].read_text())
assert compiled_headers and any(Path(path).name == "cuest.h" for path in compiled_headers)
for path, digest in compiled_headers.items():
    assert Path(path).resolve().is_relative_to(sdk / "include")
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
native_build_identity = snapshot_core(core, cmake_sdk)
assert subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip() == commit
assert not subprocess.check_output(
    ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
manifest = dict(
    commit=commit, version=psi4.__version__, job_id=os.environ["SLURM_JOB_ID"],
    qos=os.environ["SLURM_JOB_QOS"], package=str(package), core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
    linkage=linkage, dynamic_section=dynamic, gpu_test_run=False,
    cuest_sdk=str(sdk), cuest_library=str(library),
    cuest_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    cuest_sdk_identity=sdk_identity, cuda_linkage_identity=cuda_linkage_identity(linkage),
    cuest_cmake_identity=cmake_sdk, cuest_native_build_identity=native_build_identity,
    cuest_consumed_headers=compiled_headers,
    cuest_direct_jk_available=direct_available,
    note="New instrumented comparison implementation; historical stage unchanged. CUDA grid export still requires allocated-GPU validation.",
)
with (run / "build-manifest.json").open("x") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
print("BUILD_READY", package)
print("GPU_EXPORT_TEST_NOT_RUN")
PY
