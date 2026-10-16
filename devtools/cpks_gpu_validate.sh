#!/usr/bin/env bash
# GPU validation of one cpks_gpu_build.sh build: regression tests, then the probe.
# GPUs: H100/H200 for the CUDA LibXC build, also A100 for the host-LibXC build.
# Arguments: worktree, environment, commit, build output (with BUILD_OK), output,
#            variant (host|cuda), probe sections (e.g. "contract timing sapt").
set -euo pipefail
root=$(realpath "$1")
env=$(realpath "$2")
pin=$3
built=$(realpath "$4")
out=$(realpath "$5")
variant=$6
sections=$7
[[ "$variant" == host || "$variant" == cuda ]]
test -n "${SLURM_JOB_ID:-}"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
# The build link must have finished for this exact commit and variant.
test "$( < "$built/BUILD_OK")" = "$pin $variant"
build="$root/build_cpks_$variant"
(cd "$built" && sha256sum -c core.sha256)
mkdir -p "$out"
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Job-specific Lustre scratch: private /tmp has vanished mid-allocation.
scratch_root=/storage/scratch1/3/awallace43/cpks-gpu
mkdir -p "$scratch_root"
TMPDIR=$(mktemp -d "$scratch_root/$variant-$SLURM_JOB_ID.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
software=/storage/project/r-cs207-0/awallace43/software
export CUDA_HOME="$software/toolchains/cuda13" CUDA_PATH="$software/toolchains/cuda13"
libxc="$env"
if [[ "$variant" == cuda ]]; then
    libxc="$software/libxc-7.1.2-cuda"
    export LD_LIBRARY_PATH="$libxc/lib64:$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader | tee "$out/gpu.txt"
# CUDA LibXC is built for sm_90 only; the host-LibXC build has no device code of its own.
if [[ "$variant" == cuda ]]; then
    grep -Eq 'H100|H200' "$out/gpu.txt"
else
    grep -Eq 'A100|H100|H200' "$out/gpu.txt"
fi
printf '%s\n' "$pin" > "$out/commit.txt"
eval "$("$build/stage/bin/psi4" --psiapi)"
export EXPECTED_PSI4_PACKAGE="$build/stage/lib/psi4"
export CUEST_DEPENDENCY_ENV="$env" EXPECTED_LIBXC_PREFIX="$libxc" CPKS_VARIANT="$variant"
# Never run Python from the source root (it shadows the staged package).
cd "$out"
"$env/bin/python" -P - <<'PY' > import.json
import hashlib, json, os, subprocess
from pathlib import Path
import psi4
package = Path(psi4.__file__).resolve().parent
assert package == Path(os.environ["EXPECTED_PSI4_PACKAGE"]).resolve(), package
assert hasattr(psi4.core.VBase, "compute_Vx_factored")
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
                     linked_libraries=libraries, variant=os.environ["CPKS_VARIANT"],
                     cuest_packages=[{k: p[k] for k in ("name", "version", "build")} for p in packages]),
                indent=2))
PY
# Regression tests: keep going on failure so the probe still records numbers.
set +e
"$env/bin/python" -P -m pytest "$root/tests/pytests/test_cuest_cphf.py" -v -s -p no:cacheprovider \
    --junitxml="$out/cphf.xml" > "$out/cphf.log" 2>&1
cphf_status=$?
"$env/bin/python" -P -m pytest "$root/tests/pytests/test_cuest_grac.py" \
    "$root/tests/pytests/test_dft_grid_switch.py" -q -p no:cacheprovider \
    --junitxml="$out/regressions.xml" > "$out/regressions.log" 2>&1
regressions_status=$?
"$env/bin/python" -P "$root/devtools/cuest_xc_response_probe.py" "$out/probe" $sections \
    > "$out/probe.log" 2>&1
probe_status=$?
set -e
printf 'cphf=%s regressions=%s probe=%s\n' "$cphf_status" "$regressions_status" "$probe_status" \
    | tee "$out/status.txt"
"$env/bin/python" -P - <<'PY'
import json
from pathlib import Path
from xml.etree import ElementTree
summary = {}
for name in ("cphf", "regressions"):
    suite = ElementTree.parse(f"{name}.xml").find("testsuite")
    summary[name] = dict(suite.attrib)
    summary[name]["failed_cases"] = [
        f"{case.get('classname')}::{case.get('name')}" for case in suite.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None]
    summary[name]["skipped_cases"] = [
        f"{case.get('classname')}::{case.get('name')}" for case in suite.iter("testcase")
        if case.find("skipped") is not None]
Path("tests_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: {f: v[f] for f in ("tests", "failures", "errors", "skipped")}
                  for k, v in summary.items()}, indent=2))
PY
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
rm -rf "$TMPDIR"
# Truthful completion: every stage must have passed. (An `a && b && c` list does not trip
# set -e when a or b fails, which wrote VERIFIED for the failed 14085249; use an if.)
if [[ "$cphf_status" != 0 || "$regressions_status" != 0 || "$probe_status" != 0 ]]; then
    exit 1
fi
printf 'tests and probe passed for %s %s\n' "$pin" "$variant" > "$out/VERIFIED"
