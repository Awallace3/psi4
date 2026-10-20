#!/usr/bin/env bash
# Isolated FDDS build/validation. Never reuse a staged response/SCF binary.
# Arguments: source checkout, read-only dependency environment, commit, output.
set -euo pipefail
root=$(realpath "$1")
env=$(realpath "$2")
pin=$3
out=$(realpath "$4")
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
test -n "${SLURM_JOB_ID:-}"
export PATH="$env/bin:$PATH"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    export TMPDIR="/scratch/$USER/fdds-$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
fi
TMPDIR=$(mktemp -d "$TMPDIR/cuest-fdds-$SLURM_JOB_ID.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
build_root="${CUEST_FDDS_READONLY_BUILD_ROOT:-$root}"
build="$build_root/build_fdds_validation"
if [[ -n "${CUEST_FDDS_READONLY_BUILD_ROOT:-}" ]]; then
    : "${CUEST_FDDS_BINARY_PIN:?pin the existing read-only scientific binary}"
    : "${CUEST_FDDS_EXPECTED_CORE_SHA256:?pin the existing core hash}"
    test "$(git -C "$build_root" rev-parse HEAD)" = "$CUEST_FDDS_BINARY_PIN"
    test -z "$(git -C "$build_root" status --porcelain --untracked-files=no)"
    # Only test-driver/docs/harness differences are allowed when reusing a core.
    git -C "$root" diff --exit-code "$CUEST_FDDS_BINARY_PIN" "$pin" -- \
        psi4 cmake CMakeLists.txt external
    printf '%s\n' "$CUEST_FDDS_BINARY_PIN" > "$out/binary-commit.txt"
fi
mkdir -p "$out"
# A CPU allocation may compile/link the CUDA backend ahead of GPU scheduling.
# It must never write VERIFIED.json or a GPU-validation completion artifact.
compile_only="${CUEST_FDDS_COMPILE_ONLY:-0}"
[[ "$compile_only" == 0 || "$compile_only" == 1 ]]
if [[ "$compile_only" == 0 ]]; then
    nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader | tee "$out/gpu.txt"
    grep -q H200 "$out/gpu.txt"
fi
git -C "$root" rev-parse HEAD > "$out/commit.txt"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
# Site GCC_ROOT overrides the conda GCC driver's relocated cc1/sysroot lookup.
# This is job-local only; never change either installed compiler environment.
unset GCC_ROOT
# Conda compiler hooks read optional backup variables without defaults.
# Relax nounset only during activation; do not change installed environments.
set +u
source "$(dirname "$(dirname "$env")")/etc/profile.d/conda.sh"
conda activate "$env"
set -u
unset GCC_ROOT
# Site CUDA12 CPATH otherwise wins over the selected conda CUDA13 imported
# targets. This is job-local header/linker isolation, never an env upgrade.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="$env" CUDA_PATH="$env"
test "$(realpath "$(command -v nvcc)")" = "$(realpath "$env/bin/nvcc")"
cuda_include="$env/targets/x86_64-linux/include"
test -f "$cuda_include/cuda_runtime_api.h"
payload=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
{
    for pair in "cc:cc1" "c++:cc1plus"; do
        compiler="$env/bin/x86_64-conda-linux-gnu-${pair%:*}"
        backend=$("$compiler" "-print-prog-name=${pair#*:}")
        [[ "$backend" = /* && -x "$backend" ]]
        printf '%s -> %s\n' "$compiler" "$(realpath "$backend")"
        "$compiler" --version
    done
    # Also check assembler/linker/sysroot, not just frontend discovery.
    printf 'int main(void) { return 0; }\n' |
        "$env/bin/x86_64-conda-linux-gnu-cc" -x c -o "$out/compiler-c-probe" -
    printf 'int main() { return 0; }\n' |
        "$env/bin/x86_64-conda-linux-gnu-c++" -x c++ -o "$out/compiler-cxx-probe" -
    "$out/compiler-c-probe"
    "$out/compiler-cxx-probe"
} > "$out/compiler.txt" 2>&1
"$env/bin/x86_64-conda-linux-gnu-c++" \
    -I"$cuda_include" "$payload/fdds_cuda_probe.cc" \
    -L"$env/lib" -Wl,-rpath,"$env/lib" -lcudart -o "$out/cuda-probe"
if [[ "$compile_only" == 1 ]]; then
    "$env/bin/x86_64-conda-linux-gnu-c++" -I"$cuda_include" -dM -E \
        "$payload/fdds_cuda_probe.cc" | grep '^#define CUDART_VERSION ' > "$out/cuda-probe.txt"
else
    "$out/cuda-probe" > "$out/cuda-probe.txt" 2>&1
fi
ldd "$out/cuda-probe" >> "$out/cuda-probe.txt"
# Existing installed Einsums dependency only; this directory is read-only.
deps=/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage
if [[ -z "${CUEST_FDDS_READONLY_BUILD_ROOT:-}" ]]; then
"$env/bin/cmake" -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$env/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$env/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_PREFIX_PATH="$env;$deps" -DPython_EXECUTABLE="$env/bin/python" \
    -DLAPACK_LIBRARIES="$env/lib/libmkl_rt.so" -DLAPACK_INCLUDE_DIRS="$env/include" \
    -DLibxc_DIR="$env/lib/cmake/Libxc" -DLibint2_DIR="$env/lib/cmake/libint2" \
    -DEinsums_DIR="$deps/share/cmake/Einsums" \
    -DENABLE_Einsums=ON -DENABLE_OPENMP=ON -DENABLE_cuEST=ON \
    -DENABLE_dkh=OFF -DENABLE_XHOST=OFF \
    -DcuEST_ROOT="$env" -DcuEST_INCLUDE_DIR="$env/include" \
    -DcuEST_LIBRARY="$env/lib/libcuest.so" -DCMAKE_INSIST_FIND_PACKAGE_cuEST=ON \
    -DCUDAToolkit_ROOT="$env" -DCUDAToolkit_NVCC_EXECUTABLE="$env/bin/nvcc" \
    -Dpsi4_CMAKE_UNITY_BUILD=ON > "$out/configure.log" 2>&1
"$env/bin/cmake" --build "$build" -j "${SLURM_CPUS_PER_TASK:-4}" > "$out/build.log" 2>&1
fi
eval "$("$build/stage/bin/psi4" --psiapi)"
export EXPECTED_PSI4_PACKAGE="$build/stage/lib/psi4"
export CUEST_DEPENDENCY_ENV="$env"
cd "$out"
"$env/bin/python" - <<'PY' > import.json
import hashlib, json, os, subprocess
from pathlib import Path
import psi4
p = Path(psi4.__file__).resolve().parent
assert p == Path(os.environ["EXPECTED_PSI4_PACKAGE"]).resolve(), p
assert hasattr(psi4.core, "cuest_chain_gemm")
assert hasattr(psi4.core.FDDS_Dispersion, "gpu_contraction_blocks")
core = Path(psi4.core.__file__)
core_hash = hashlib.sha256(core.read_bytes()).hexdigest()
expected_hash = os.environ.get("CUEST_FDDS_EXPECTED_CORE_SHA256")
if expected_hash:
    assert core_hash == expected_hash, (core_hash, expected_hash)
ldd = subprocess.check_output(["ldd", str(core)], text=True)
assert "not found" not in ldd, ldd
libraries = {}
for line in ldd.splitlines():
    words = line.split()
    if "=>" in words and any(k in words[0].lower() for k in ("cuest", "libxc", "cublas", "cudart")):
        libraries[words[0]] = str(Path(words[words.index("=>") + 1]).resolve())
packages = []
for file in (Path(os.environ["CUEST_DEPENDENCY_ENV"]) / "conda-meta").glob("libcuest*.json"):
    data = json.loads(file.read_text())
    packages.append({k: data[k] for k in ("name", "version", "build")})
assert any(p["name"] == "libcuest" and p["version"] == "0.2.2.2" for p in packages), packages
print(json.dumps(dict(package=str(p), core=str(core), version=psi4.__version__,
                     core_sha256=core_hash,
                     linked_libraries=libraries,
                     linked_library_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                            for name, path in libraries.items()},
                     binary_pin=os.environ.get("CUEST_FDDS_BINARY_PIN"),
                     cuest_packages=packages), indent=2))
PY
if [[ "$compile_only" == 1 ]]; then
    printf 'CUDA-enabled compile/link/import only; zero GPU tests executed.\n' > "$out/BUILD_ONLY.txt"
    exit 0
fi
test_selection=()
if [[ "${CUEST_FDDS_FOCUSED:-0}" == 1 ]]; then
    test_selection=(-k 'test_fdds_contribution_fixture_budget or test_fdds_gpu_contributions or test_fdds_native_exchange_preprocessing or test_fdds_gpu_exchange_preprocessing')
    # Preserve both focused outcomes even if the contribution process fails.
    set +e
fi
"$env/bin/python" -m pytest "$root/tests/pytests/test_cuest_fdds.py" -q -s \
    "${test_selection[@]}" --junitxml="$out/fdds.xml" > "$out/fdds.log" 2>&1
contribution_status=$?
if [[ "${CUEST_FDDS_FOCUSED:-0}" == 1 ]]; then
    # Native faults must stop at GDB, not wait indefinitely for an attachment.
    # A successful GDB process alone is not a successful pytest run.
    export PYTHONFAULTHANDLER=1
    ulimit -c 0
    timeout --signal=TERM --kill-after=15s 240s /usr/bin/gdb --batch \
        -ex 'set pagination off' -ex 'set confirm off' -ex 'set breakpoint pending on' \
        -ex 'handle SIGSEGV stop print nopass' \
        -ex 'break einsums::util::attach_debugger()' \
        -ex run -ex 'thread apply all bt 12' -ex quit \
        --args "$env/bin/python" -X faulthandler -m pytest \
        "$root/tests/pytests/test_cuest_fdds.py::test_saptdft_fdds_native_xc_gpu" \
        -vv -x -s --basetemp="$out/sapt-tmp" --junitxml="$out/sapt.xml" \
        > "$out/sapt-gdb.log" 2>&1
    gdb_status=$?
    set -e
    export CUEST_FDDS_CONTRIBUTION_STATUS="$contribution_status" CUEST_FDDS_GDB_STATUS="$gdb_status"
fi
"$env/bin/python" - <<'PY'
from pathlib import Path
from xml.etree import ElementTree
import json, os, re
suite = ElementTree.parse("fdds.xml").find("testsuite")
counts = dict(suite.attrib)
focused = os.environ.get("CUEST_FDDS_FOCUSED", "0") == "1"
if focused:
    assert int(os.environ["CUEST_FDDS_CONTRIBUTION_STATUS"]) == 0, "contribution subprocess failed"
    assert int(os.environ["CUEST_FDDS_GDB_STATUS"]) == 0, "GDB subprocess failed or timed out"
assert int(suite.get("tests")) == (20 if focused else 69), counts
assert all(int(suite.get(k, "0")) == 0 for k in ("errors", "failures", "skipped")), counts
if focused:
    log = Path("sapt-gdb.log").read_text()
    assert not re.search(r"received signal|hit Breakpoint|^Breakpoint \d+,", log, re.MULTILINE), "native fault/debugger breakpoint"
    assert "exited normally" in log, "GDB did not observe a normal inferior exit"
    sapt = ElementTree.parse("sapt.xml").find("testsuite")
    sapt_counts = dict(sapt.attrib)
    assert int(sapt.get("tests")) == 1, sapt_counts
    assert all(int(sapt.get(k, "0")) == 0 for k in ("errors", "failures", "skipped")), sapt_counts
    counts = {"mode": "focused_only_not_full_suite", "contribution_tests": counts,
              "sapt_gpu_test": sapt_counts, "executed_tests": 21}
Path("VERIFIED.json").write_text(json.dumps(counts, indent=2) + "\n")
print(json.dumps(counts, indent=2))
PY
# Use the same allocation for the full FDDS suite after both focused processes
# pass, including when this allocation built its own new scientific stage.
if [[ "${CUEST_FDDS_FOCUSED:-0}" == 1 &&
      "${CUEST_FDDS_FULL_AFTER_FOCUSED:-1}" == 1 ]]; then
    mv "$out/VERIFIED.json" "$out/FOCUSED_VERIFIED.json"
    timeout --signal=TERM --kill-after=15s 300s /usr/bin/gdb --batch \
        -ex 'set pagination off' -ex 'set confirm off' -ex 'set print thread-events off' \
        -ex 'set breakpoint pending on' -ex 'handle SIGSEGV stop print nopass' \
        -ex 'break einsums::util::attach_debugger()' \
        -ex run -ex 'thread apply all bt 12' -ex quit \
        --args "$env/bin/python" -X faulthandler -m pytest \
        "$root/tests/pytests/test_cuest_fdds.py" -vv -s \
        --basetemp="$out/full-tmp" --junitxml="$out/full-fdds.xml" \
        > "$out/full-fdds-gdb.log" 2>&1
    "$env/bin/python" - <<'PY'
from pathlib import Path
from xml.etree import ElementTree
import json, re
log = Path("full-fdds-gdb.log").read_text()
assert not re.search(r"received signal|hit Breakpoint|^Breakpoint \d+,", log, re.MULTILINE), "full suite native fault"
assert "exited normally" in log, "full suite inferior did not exit normally"
suite = ElementTree.parse("full-fdds.xml").find("testsuite")
counts = dict(suite.attrib)
assert int(suite.get("tests")) == 69, counts
assert all(int(suite.get(k, "0")) == 0 for k in ("errors", "failures", "skipped")), counts
focused = json.loads(Path("FOCUSED_VERIFIED.json").read_text())
result = {"mode": "focused_then_full", "focused": focused, "full": counts,
          "executed_tests_including_repeats": 90, "unique_full_suite_tests": 69}
Path("VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
fi
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
if [[ -n "${CUEST_FDDS_READONLY_BUILD_ROOT:-}" ]]; then
    test "$(git -C "$build_root" rev-parse HEAD)" = "$CUEST_FDDS_BINARY_PIN"
    test -z "$(git -C "$build_root" status --porcelain --untracked-files=no)"
fi
