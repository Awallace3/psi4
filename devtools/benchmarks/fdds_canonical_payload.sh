#!/usr/bin/env bash
# Compute-node-only payload for fdds_canonical_job.sh.
set -euo pipefail
[[ $# == 4 ]]
source_root=$1 pin=$2 dep=$3 out=$4
# No inherited reuse/compile-only escape hatch may weaken this fresh build.
for variable in CUEST_FDDS_READONLY_BUILD_ROOT CUEST_FDDS_BINARY_PIN \
    CUEST_FDDS_EXPECTED_CORE_SHA256 CUEST_FDDS_BINARY_IDENTITY; do
    [[ ! -v "$variable" ]] || { echo "Refusing inherited $variable" >&2; exit 2; }
done
[[ ! -e "$source_root/build_fdds_validation" ]]
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
set +u
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate "$dep"
set -u
export PATH="$dep/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$dep/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
# Validation fixtures are small. Avoid the long-lived /tmp bind mount and
# network-intensive protein timings in this first canonical-version gate.
scratch="/storage/scratch1/3/$USER/fdds-canonical/$SLURM_JOB_ID"
mkdir -p "$scratch"
export TMPDIR="$scratch" SCRATCH="$scratch" PSI_SCRATCH="$scratch"
mkdir "$out/native"
native="$source_root/build_fdds_canonical_native"
[[ ! -e "$native" ]]
for spec in 'cc:cc1' 'c++:cc1plus'; do
    compiler="$dep/bin/x86_64-conda-linux-gnu-${spec%:*}"
    frontend=$("$compiler" -print-prog-name="${spec##*:}")
    [[ "$frontend" = /* && -x "$frontend" ]]
    printf '%s %s\n' "$compiler" "$frontend"
done
"$dep/bin/cmake" -S "$source_root" -B "$native" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$dep/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$dep/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_PREFIX_PATH="$dep" -DPython_EXECUTABLE="$dep/bin/python" \
    -DLAPACK_LIBRARIES="$dep/lib/libmkl_rt.so" -DLAPACK_INCLUDE_DIRS="$dep/include" \
    -DLibxc_DIR="$dep/lib/cmake/Libxc" -DLibint2_DIR="$dep/lib/cmake/libint2" \
    -DENABLE_Einsums=OFF -DENABLE_OPENMP=ON -DENABLE_cuEST=OFF \
    -DENABLE_dkh=OFF -DENABLE_XHOST=OFF -Dpsi4_CMAKE_UNITY_BUILD=ON \
    > "$out/native/configure.log" 2>&1
"$dep/bin/cmake" --build "$native" -j8 > "$out/native/build.log" 2>&1
(
    eval "$("$native/stage/bin/psi4" --psiapi)"
    cd "$out/native"
    EXPECTED_PACKAGE="$native/stage/lib/psi4" "$dep/bin/python" - <<'PY' > provenance.json
import hashlib, json, os, pathlib, subprocess, psi4
p = pathlib.Path(psi4.__file__).resolve().parent
assert p == pathlib.Path(os.environ["EXPECTED_PACKAGE"]).resolve()
core = pathlib.Path(psi4.core.__file__).resolve()
deps = subprocess.check_output(["ldd", str(core)], text=True)
assert "libcuest" not in deps and "libcudart" not in deps
print(json.dumps({"package": str(p), "core": str(core),
                  "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
                  "linked_libraries": deps}, indent=2))
PY
    timeout --signal=TERM --kill-after=30s 900 "$dep/bin/python" -m pytest \
        "$source_root/tests/pytests/test_cuest_fdds.py" \
        "$source_root/tests/pytests/test_cuest_cphf.py" \
        "$source_root/tests/pytests/test_scf_lifecycle.py" \
        -m 'not cuest and not cuda_cc8' -q -s --junitxml=tests.xml > tests.log 2>&1
    "$dep/bin/python" - <<'PY'
import json, pathlib, xml.etree.ElementTree as ET
root = ET.parse("tests.xml").getroot()
suites = list(root.iter("testsuite"))
counts = {k: sum(int(s.get(k, "0")) for s in suites)
          for k in ("tests", "failures", "errors", "skipped")}
assert counts["tests"] > 0 and not any(counts[k] for k in ("failures", "errors", "skipped")), counts
pathlib.Path("verified.json").write_text(json.dumps(counts, indent=2) + "\n")
PY
)
export CUEST_SDK_ROOT=/storage/project/r-cs207-0/awallace43/software/cuest-0.2.50-a06e8148
export CUEST_FDDS_COMPILE_ONLY=0 CUEST_FDDS_FOCUSED=1 CUEST_FDDS_FULL_AFTER_FOCUSED=1
# Existing driver performs compiler/CUDA probes, canonical outer/inner CMake
# admission, original core receipt, same-process runtime admission and GDB gates.
bash "$source_root/devtools/benchmarks/fdds_validate.sh" \
    "$source_root" "$dep" "$pin" "$out/gpu"
