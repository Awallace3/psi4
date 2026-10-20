#!/usr/bin/env bash
# Small diagnostic-only compile, before requesting a GPU. Never rebuilds E2/Psi4.
set -euo pipefail
P=/storage/project/r-cs207-0/awallace43
source_root="${DISP0_GATE_SOURCE:?immutable validation-harness source required}"
pin="${DISP0_GATE_PIN:?full validation-harness commit required}"
build="${DISP0_BUILD_ROOT:?completed immutable corrected build required}"
build_pin="${DISP0_BUILD_PIN:?full successful scientific build commit required}"
build_job="${DISP0_BUILD_JOB_ID:?successful CPU preparation job ID required}"
science="${DISP0_SCIENCE_SOURCE:?immutable scientific build source required}"
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$(git -C "$source_root" rev-parse HEAD)" == "$pin" ]]
[[ "$build_pin" =~ ^[0-9a-f]{40}$ && "$build_job" =~ ^[0-9]+$ ]]
[[ "$build" == "$P/runs/saptdft-cuest-disp0/build-$build_pin-$build_job" ]]
[[ "$(git -C "$science" rev-parse HEAD)" == "$build_pin" ]]
git -C "$science" diff --quiet
git -C "$science" diff --cached --quiet
git -C "$source_root" diff --quiet
git -C "$source_root" diff --cached --quiet
# Standalone harness only: scientific files must remain exactly the successful build pin.
git -C "$source_root" diff --exit-code "$build_pin" "$pin" -- psi4/
python3 - "$build" "$build_job" <<'PY'
import json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
r = Path(sys.argv[1])
status = (r.parent / f"logs/{sys.argv[2]}.status").read_text()
assert "exit_code=0\n" in status and "term_received=0\n" in status and "CPU_PREPARATION_EXIT_ZERO" in status
for name in ("runtime-provenance.json", "api-probe.json"):
    assert json.loads((r / name).read_text())["status"] == "PASS", name
assert (r / "psi4-build/stage/bin/psi4").is_file()
results = list((r / "psi4-build").glob("reference-check-*/results.xml"))
assert len(results) == 1, results
cases = list(ET.parse(results[0]).getroot().iter("testcase"))
assert len(cases) == 26, len(cases)
assert not any(c.find("failure") is not None or c.find("error") is not None for c in cases)
skipped = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
assert all(name == "test_disp0_cuda_request_rejects_mock" for name in skipped), skipped
records = []
for c in cases:
    for p in c.findall("./properties/property"):
        if p.attrib["name"] == "signed_comparisons":
            records.extend(json.loads(p.attrib["value"]))
assert len(records) == 64, len(records)
print(json.dumps({"cpu_tests": len(cases), "allowed_skips": skipped,
                  "retained_signed_comparisons": len(records), "result_xml": str(results[0])}))
PY
prep="$P/runs/saptdft-cuest-disp0/gate-prep-$pin"
mkdir "$prep"  # never reuse/overwrite a partial preparation
export TMPDIR="$prep/tmp"
mkdir "$TMPDIR"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH PYTHONPATH PYTHONHOME LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="$P/software/toolchains/cuda13" CUDA_PATH="$P/software/toolchains/cuda13"
env="$build/environment"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$env/lib"
export PATH="$env/bin:$CUDA_HOME/bin:$PATH"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
[[ "$("$cxx" -dumpfullversion)" == 14.3.0 ]]
[[ "$("$CUDA_HOME/bin/nvcc" --version)" == *"V13.3.73"* ]]
"$cxx" -std=c++20 -O2 -shared -fPIC "$source_root/devtools/disp0/cublas_trace.cc" \
    -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib" \
    -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib" \
    -Wl,--no-as-needed -lcublas -Wl,--as-needed -lcudart -ldl -o "$prep/libdisp0_cublas_trace.so"
readelf -d "$prep/libdisp0_cublas_trace.so" > "$prep/tracer-elf.txt"
nm -D "$prep/libdisp0_cublas_trace.so" > "$prep/tracer-symbols.txt"
sha256sum "$prep/libdisp0_cublas_trace.so" | tee "$prep/tracer-sha256.txt"
# Load and resolve only; no CUDA device/API execution on the login node.
"$env/bin/python" - "$prep/libdisp0_cublas_trace.so" <<'PY'
import ctypes, json, sys
from pathlib import Path
library = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
function = library.disp0_cublas_forwarding_identity
function.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
function.restype = ctypes.c_int
buffer = ctypes.create_string_buffer(8192)
assert function(buffer, len(buffer)) == 0, buffer.value
result = json.loads(buffer.value)
assert result["status"] == 0 and "libcublas.so.13" in Path(result["library"]).resolve().name, result
print("NATIVE_FORWARDING_RESOLUTION_ONLY", json.dumps(result))
PY
printf 'GATE_PREP=%s\nHARNESS_PIN=%s\nBUILD_ROOT=%s\nBUILD_PIN=%s\nBUILD_JOB=%s\n' \
    "$prep" "$pin" "$build" "$build_pin" "$build_job"
