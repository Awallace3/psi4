#!/usr/bin/env bash
# One frozen worktree, two independent fresh builds. Dependencies are read-only.
# Arguments: worktree, environment, commit, output, variant (host|cuda).
set -euo pipefail
root=$(realpath "$1")
env=$(realpath "$2")
pin=$3
out=$(realpath "$4")
variant=$5
[[ "$variant" == host || "$variant" == cuda ]]
test -n "${SLURM_JOB_ID:-}"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
mkdir -p "$out"
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    export TMPDIR="/scratch/$USER/grid-$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
fi
TMPDIR=$(mktemp -d "$TMPDIR/grid-$variant-$SLURM_JOB_ID.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT
deps=/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage
software=/storage/project/r-cs207-0/awallace43/software
libxc="$env"
cc="$env/bin/x86_64-conda-linux-gnu-cc"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
cuda_args=()
if [[ "$variant" == cuda ]]; then
    libxc="$software/libxc-7.1.2-cuda"
    cc="$software/toolchains/conda-gcc"
    cxx="$software/toolchains/conda-g++"
    cuda_args=(
        "-DCMAKE_CUDA_COMPILER=$software/toolchains/cuda13/bin/nvcc"
        "-DCMAKE_CUDA_HOST_COMPILER=$cxx"
        "-DCUDAToolkit_ROOT=$software/toolchains/cuda13"
        "-DCMAKE_CUDA_ARCHITECTURES=90"
        "-DCMAKE_CUDA_FLAGS=-I$software/toolchains/cuda13/targets/x86_64-linux/include"
    )
    libxc_config="$libxc/lib64/cmake/Libxc"
    export LD_LIBRARY_PATH="$libxc/lib64:$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    libxc_config="$libxc/lib/cmake/Libxc"
    export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
{
    "$cc" --version
    "$cxx" --version
    test -x "$("$cc" -print-prog-name=cc1)"
    test -x "$("$cxx" -print-prog-name=cc1plus)"
    printf 'cc=%s\ncxx=%s\nvariant=%s\n' "$cc" "$cxx" "$variant"
} > "$out/compiler.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader | tee "$out/gpu.txt"
grep -q H200 "$out/gpu.txt"
printf '%s\n' "$pin" > "$out/commit.txt"
build="$root/build_grid_$variant"
mkdir -p "$build"
if [[ -e "$build/grid-commit.txt" ]]; then
    test "$( < "$build/grid-commit.txt")" = "$pin"
else
    # Refuse an unrelated existing build rather than repointing its cache.
    test ! -e "$build/CMakeCache.txt"
    printf '%s\n' "$pin" > "$build/grid-commit.txt"
fi
"$env/bin/cmake" -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_C_COMPILER=$cc" "-DCMAKE_CXX_COMPILER=$cxx" \
    "-DCUDAToolkit_ROOT=$software/toolchains/cuda13" \
    "-DCMAKE_C_FLAGS=-I$software/toolchains/cuda13/targets/x86_64-linux/include" \
    "-DCMAKE_CXX_FLAGS=-I$software/toolchains/cuda13/targets/x86_64-linux/include" \
    "-DCMAKE_PREFIX_PATH=$libxc;$env;$deps" \
    "-DPython_EXECUTABLE=$env/bin/python" \
    "-DLAPACK_LIBRARIES=$env/lib/libmkl_rt.so" "-DLAPACK_INCLUDE_DIRS=$env/include" \
    "-DLibxc_DIR=$libxc_config" "-DLibint2_DIR=$env/lib/cmake/libint2" \
    "-DEinsums_DIR=$deps/share/cmake/Einsums" \
    -DENABLE_Einsums=ON -DENABLE_OPENMP=ON -DENABLE_cuEST=ON \
    -DENABLE_dkh=OFF -DENABLE_XHOST=OFF \
    "-DcuEST_ROOT=$env" "-DcuEST_INCLUDE_DIR=$env/include" \
    "-DcuEST_LIBRARY=$env/lib/libcuest.so" \
    -DCMAKE_INSIST_FIND_PACKAGE_cuEST=ON -Dpsi4_CMAKE_UNITY_BUILD=ON \
    "${cuda_args[@]}" > "$out/configure.log" 2>&1
"$env/bin/cmake" --build "$build" -j "$OMP_NUM_THREADS" > "$out/build.log" 2>&1
# A CUDA 12 header can otherwise silently compile against a CUDA 13 runtime.
# The latter no longer exports cudaGetDeviceProperties_v2. Test before import.
if readelf --dyn-syms --wide "$build"/stage/lib/psi4/core*.so | grep ' UND cudaGetDeviceProperties_v2'; then
    echo 'CUDA header/runtime mismatch: core still requires the CUDA 12 device-properties ABI' >&2
    exit 1
fi
eval "$("$build/stage/bin/psi4" --psiapi)"
export EXPECTED_PSI4_PACKAGE="$build/stage/lib/psi4"
export CUEST_DEPENDENCY_ENV="$env" EXPECTED_LIBXC_PREFIX="$libxc" GRID_VARIANT="$variant"
cd "$out"
"$env/bin/python" - <<'PY' > import.json
import hashlib, json, os, subprocess
from pathlib import Path
import psi4
package = Path(psi4.__file__).resolve().parent
assert package == Path(os.environ["EXPECTED_PSI4_PACKAGE"]).resolve(), package
assert hasattr(psi4.core, "cuest_chain_gemm")
core = Path(psi4.core.__file__)
ldd = subprocess.check_output(["ldd", str(core)], text=True)
assert "not found" not in ldd, ldd
libraries = {}
for line in ldd.splitlines():
    words = line.split()
    if "=>" in words and any(k in words[0].lower() for k in ("cuest", "libxc", "cublas", "cudart")):
        libraries[words[0]] = str(Path(words[words.index("=>") + 1]).resolve())
xc = [Path(v) for k, v in libraries.items() if k.startswith("libxc.")]
assert xc and all(p.is_relative_to(Path(os.environ["EXPECTED_LIBXC_PREFIX"])) for p in xc), libraries
packages = [json.loads(p.read_text()) for p in
            (Path(os.environ["CUEST_DEPENDENCY_ENV"]) / "conda-meta").glob("libcuest*.json")]
assert any(p["name"] == "libcuest" and p["version"] == "0.2.2.2" for p in packages), packages
print(json.dumps(dict(package=str(package), core=str(core), version=psi4.__version__,
                     core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),
                     linked_libraries=libraries, variant=os.environ["GRID_VARIANT"],
                     cuest_packages=[{k:p[k] for k in ("name","version","build")} for p in packages]), indent=2))
PY
"$env/bin/python" -m pytest "$root/tests/pytests/test_dft_grid_switch.py" \
    "$root/tests/pytests/test_scf_lifecycle.py" -q --junitxml="$out/regressions.xml" \
    > "$out/regressions.log" 2>&1
"$env/bin/python" - <<'PY'
import json
from pathlib import Path
from xml.etree import ElementTree
suite = ElementTree.parse("regressions.xml").find("testsuite")
counts = dict(suite.attrib)
assert int(suite.get("tests")) == 34, counts
assert all(int(suite.get(k, "0")) == 0 for k in ("errors", "failures", "skipped")), counts
Path("REGRESSIONS_VERIFIED.json").write_text(json.dumps(counts, indent=2) + "\n")
PY
# Fresh processes keep scratch, options, timers and grid controls independent.
"$env/bin/python" "$root/devtools/saptdft_xc_profile.py" --output "$out/profile" \
    --threads "$OMP_NUM_THREADS" --use-cuest > "$out/profile.log" 2>&1
"$env/bin/python" - <<'PY'
import json, os
from pathlib import Path
data = json.loads(Path("profile/timers.json").read_text())
cuda = any("cuEST XC: CUDA LibXC Functional" in key for key in data["timers"])
assert cuda == (os.environ["GRID_VARIANT"] == "cuda"), "Wrong LibXC execution route"
assert any("xc_gga_exc_vxc" in key for key in data["microtimers"])
PY
"$env/bin/python" "$root/devtools/saptdft_grid_sweep.py" --output "$out/water" \
    --system water --threads "$OMP_NUM_THREADS" --use-cuest > "$out/water.log" 2>&1
"$env/bin/python" "$root/devtools/saptdft_grid_sweep.py" --output "$out/water-iterative" \
    --system water --threads "$OMP_NUM_THREADS" --use-cuest --iterative-grac \
    > "$out/water-iterative.log" 2>&1
"$env/bin/python" "$root/devtools/saptdft_grid_sweep.py" --output "$out/water-full" \
    --system water --threads "$OMP_NUM_THREADS" --use-cuest --full-sapt --compact \
    > "$out/water-full.log" 2>&1
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
printf 'build, regressions, profiling and sweeps finished; see numerical gates in summary.json\n' > "$out/VERIFIED"
