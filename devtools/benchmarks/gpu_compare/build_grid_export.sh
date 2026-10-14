#!/bin/bash
# Run on a compute node under explicit embers QoS, not on a login node.
# Usage: /bin/bash build_grid_export.sh ABS_REPO COMMIT ABS_RUN
# The historical CUDA-LibXC stage is read-only and is never rebuilt here.
set -euo pipefail
repo=${1:?absolute comparison worktree}
commit=${2:?pinned commit}
run=${3:?absolute unique build run}
[[ "$repo" == /* && "$run" == /* && "$run" != "$repo" ]]
export SLURM_JOB_QOS
SLURM_JOB_QOS=$(squeue -h -j "${SLURM_JOB_ID:?}" -o '%q')
[[ "$SLURM_JOB_QOS" == embers ]] || { echo "Expected an embers compute allocation" >&2; exit 2; }
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
[[ ! -e "$run/build" && ! -e "$run/build-manifest.json" ]]

env_root=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
toolchains=/storage/project/r-cs207-0/awallace43/software/toolchains
cuda="$toolchains/cuda13"
xc=/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda
source /etc/profile.d/z00_lmod_pace.sh
module load cmake
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
export PATH="$env_root/bin:$cuda/bin:$PATH" PYTHONNOUSERSITE=1
export CUDA_HOME="$cuda" CUDA_PATH="$cuda"
export LD_LIBRARY_PATH="$xc/lib64:$cuda/lib64:$env_root/lib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/storage/scratch1/3/awallace43/gpu-dft-grid-build/${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR" "$run"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"

cc="$toolchains/conda-gcc"
cxx="$toolchains/conda-g++"
test -x "$("$cc" -print-prog-name=cc1)"
test -x "$("$cxx" -print-prog-name=cc1plus)"
cmake -S "$repo" -B "$run/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_CUDA_HOST_COMPILER="$cxx" -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
    -DCUDAToolkit_ROOT="$cuda" -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_PREFIX_PATH="$env_root;$xc" \
    -DPython_EXECUTABLE="$env_root/bin/python" \
    -DLibint2_DIR="$env_root/lib/cmake/libint2" \
    -DLibxc_DIR="$xc/lib64/cmake/Libxc" \
    -DcuEST_ROOT="$env_root" -DENABLE_cuEST=ON \
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
run = Path(run)
package = run / "build/stage/lib/psi4"
assert Path(psi4.__file__).resolve().parent == package.resolve()
assert hasattr(psi4.core.VBase, "export_cuest_grid")
assert hasattr(psi4.core.VBase, "cuest_grid_points")
core = next(package.glob("core*.so"))
linkage = subprocess.check_output(["ldd", str(core)], text=True)
assert "not found" not in linkage, linkage
assert "/software/libxc-7.1.2-cuda/" in linkage, linkage
assert "libcudart.so.13" in linkage, linkage
assert subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip() == commit
assert not subprocess.check_output(
    ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
manifest = dict(
    commit=commit, version=psi4.__version__, job_id=os.environ["SLURM_JOB_ID"],
    qos=os.environ["SLURM_JOB_QOS"], package=str(package), core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
    linkage=linkage, gpu_test_run=False,
    note="New instrumented comparison implementation; historical stage unchanged. CUDA grid export still requires allocated-GPU validation.",
)
with (run / "build-manifest.json").open("x") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
print("BUILD_READY", package)
print("GPU_EXPORT_TEST_NOT_RUN")
PY
