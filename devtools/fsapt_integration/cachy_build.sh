#!/bin/bash
# Private coherent sm_86 build, followed by CPU CPKS numerical regressions.
# GPU F-SAPT/runtime qualification is a separate gate, not implied by this.
set -euo pipefail
ROOT=${1:?private root}
PIN=${2:?exact source commit}
ENV=$HOME/gits/psi4-disp0-cachy/environment-v2
test "$(git -C "$ROOT/source" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$ROOT/source" status --porcelain)"
export PATH="$ENV/bin:/usr/bin:/bin" CONDA_PREFIX="$ENV" MATH_ROOT="$ENV"
export CUDA_HOME="$ENV" CUDA_PATH="$ENV"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
# --psiapi emits an expansion of PYTHONPATH; keep it defined under set -u.
export PYTHONPATH=""
export LD_LIBRARY_PATH="$ROOT/runtime/lib:$ENV/lib"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RP="$ROOT/runtime/lib;$ENV/lib"
cmake -S "$ROOT/source" -B "$ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ROOT/install" -DCMAKE_INSTALL_LIBDIR=lib \
    -DPYMOD_INSTALL_LIBDIR=/python3.13/site-packages \
    -DCMAKE_C_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_PREFIX_PATH="$ENV" -DPython_EXECUTABLE="$ENV/bin/python" \
    -DCUDAToolkit_ROOT="$ENV" -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER="$ENV/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-c++" \
    -DBLAS_TYPE=MKL -DLAPACK_TYPE=MKL -DCMAKE_INSIST_FIND_PACKAGE_Libxc=ON \
    -DLibxc_DIR="$ENV/lib/cmake/Libxc" -DENABLE_cuEST=ON \
    -DcuEST_ROOT="$ROOT/runtime/cuest" -DcuEST_INCLUDE_DIR="$ROOT/runtime/cuest/include" \
    -DcuEST_LIBRARY="$ROOT/runtime/cuest/lib/13/libcuest.so" \
    -DCMAKE_BUILD_RPATH="$RP" -DCMAKE_INSTALL_RPATH="$RP" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Dpsi4_CMAKE_UNITY_BUILD_BATCH_SIZE=4
cmake --build "$ROOT/build" -j 4
cmake --install "$ROOT/build"
test "$(git -C "$ROOT/source" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$ROOT/source" status --porcelain)"
printf '%s\n' "$PIN" > "$ROOT/install/BUILT_FROM"
echo BUILD_OK

export ROOT PIN
RUN=$(mktemp -d "$ROOT/runs/cpu-cpks-$PIN-XXXXXX")
mkdir "$RUN/scratch"
export TMPDIR="$RUN/scratch" SCRATCH="$RUN/scratch" PSI_SCRATCH="$RUN/scratch"
eval "$("$ROOT/install/bin/psi4" --psiapi)"
python - <<'PY'
import os
from pathlib import Path
import psi4
root = Path(os.environ["ROOT"])
assert Path(psi4.__file__).resolve().is_relative_to(root / "install")
assert Path(psi4.core.__file__).resolve().is_relative_to(root / "install")
assert (root / "install/BUILT_FROM").read_text().strip() == os.environ["PIN"]
print("TEST_IMPORT", psi4.__file__, psi4.core.__file__, flush=True)
PY
# A new working directory prevents timer.dat and F-SAPT outputs overwriting runs.
cd "$RUN"
python -m pytest "$ROOT/source/tests/pytests/test_saptdft.py" \
    -k fsapt_xc_response -v --junitxml="$RUN/junit.xml"
python - "$RUN/junit.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = list(root.iter("testcase"))
assert len(cases) == 4, len(cases)
assert not any(list(c.iter(tag)) for c in cases for tag in ("skipped", "failure", "error"))
print("CPU_CPKS_NUMERICAL_OK")
PY
