#!/usr/bin/env bash
# Read-only reproduction using an existing staged GPU core, never a rebuild.
# Arguments: frozen scientific checkout, diagnostic checkout, output directory.
set -euo pipefail
science=$(realpath "$1")
driver=$(realpath "$2")
out=$(realpath "$3")
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
pin=8810da5743cc7350e7620ecb708cdd3e47baa739
test "$(git -C "$science" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$science" status --porcelain --untracked-files=no)"
test -z "$(git -C "$driver" status --porcelain --untracked-files=no)"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT
set +u
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate "$env"
set -u
unset GCC_ROOT
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export TMPDIR="$out/tmp" SCRATCH="$out/tmp" PSI_SCRATCH="$out/tmp"
mkdir -p "$TMPDIR"
ulimit -c 0
stage="$science/build_fdds_validation/stage"
eval "$("$stage/bin/psi4" --psiapi)"
export EXPECTED_PSI4_PACKAGE="$stage/lib/psi4"
export SCIENCE_ROOT="$science" DRIVER_ROOT="$driver"
cd "$out"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > gpu.txt
grep -q H200 gpu.txt
"$env/bin/python" - <<'PY' > provenance.json
import hashlib, json, os, subprocess
from pathlib import Path
import psi4
package = Path(psi4.__file__).resolve().parent
assert package == Path(os.environ["EXPECTED_PSI4_PACKAGE"]).resolve(), package
core = Path(psi4.core.__file__).resolve()
core_hash = hashlib.sha256(core.read_bytes()).hexdigest()
assert core_hash == "3fe5a446db657ecb82c6557973eab85f08c161729ab16d047df7c16fb9d61c73", core_hash
ldd = subprocess.check_output(["ldd", str(core)], text=True)
assert "not found" not in ldd, ldd
libs = {}
for line in ldd.splitlines():
    words = line.split()
    if "=>" in words and any(s in words[0].lower() for s in ("cuest", "cublas", "cudart", "libxc")):
        p = Path(words[words.index("=>") + 1]).resolve()
        libs[words[0]] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
rev = lambda root: subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
print(json.dumps({"scientific_pin": rev(os.environ["SCIENCE_ROOT"]),
                  "diagnostic_driver_pin": rev(os.environ["DRIVER_ROOT"]),
                  "original_build_harness_pin": "e60b32507b3069192032705b1d4161dd38886511",
                  "package": str(package), "version": psi4.__version__,
                  "core": str(core), "core_sha256": core_hash, "libraries": libs}, indent=2))
PY
# Separate processes: contribution failure cannot suppress the native stack.
set +e
timeout --signal=TERM --kill-after=15s 120s "$env/bin/python" -X faulthandler -m pytest \
    "$science/tests/pytests/test_cuest_fdds.py" \
    -k 'test_fdds_gpu_contributions and unequal and False' -vv -x -s \
    --basetemp="$out/contributions-tmp" --junitxml="$out/contributions.xml" \
    > "$out/contributions.log" 2>&1
contrib_status=$?
timeout --signal=TERM --kill-after=15s 240s /usr/bin/gdb --batch \
    -ex 'set pagination off' -ex 'set confirm off' -ex 'set breakpoint pending on' \
    -ex 'handle SIGSEGV stop print nopass' \
    -ex 'break einsums::util::attach_debugger()' \
    -ex run -ex 'thread apply all bt 12' -ex quit \
    --args "$env/bin/python" -X faulthandler -m pytest \
    "$science/tests/pytests/test_cuest_fdds.py::test_saptdft_fdds_native_xc_gpu" \
    -vv -x -s --basetemp="$out/sapt-tmp" --junitxml="$out/sapt.xml" \
    > "$out/sapt-gdb.log" 2>&1
gdb_status=$?
set -e
export CONTRIB_STATUS="$contrib_status" GDB_STATUS="$gdb_status"
"$env/bin/python" - <<'PY' > DIAGNOSED.json
import json, os
from pathlib import Path
s = Path("sapt-gdb.log").read_text()
print(json.dumps({"contribution_process_exit": int(os.environ["CONTRIB_STATUS"]),
                  "gdb_process_exit": int(os.environ["GDB_STATUS"]),
                  "native_fault_or_breakpoint": "SIGSEGV" in s or "Breakpoint 1," in s,
                  "note": "Diagnostic completion only; GDB exit 0 after a fault is NOT a test pass."}, indent=2))
PY
test "$(git -C "$science" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$science" status --porcelain --untracked-files=no)"
