#!/usr/bin/env bash
# Separate preparation route. NEVER feeds failed14196330 to the successful-build admission.
# Requires explicit clearance for this small forwarding-diagnostic compilation;
# artifact-only deployment clearance alone is not sufficient.
set -euo pipefail
umask 077
if [[ "${1:-}" != --bounded-child ]]; then
    # Includes module setup, compiler/preprocessor, ELF and load/resolve checks.
    export DISP0_DIAGNOSTIC_START_EPOCH="$(date +%s)"
    exec timeout --signal=TERM --kill-after=5s 110s bash "${BASH_SOURCE[0]}" --bounded-child
fi
ulimit -v 4194304  # 4GiB virtual-address ceiling includes large read-only NVIDIA ELF mappings.
ulimit -f 65536    # 64MiB maximum output file; no packages/build trees
ulimit -t 110
: "${DISP0_RECOVERY_DIAGNOSTIC_COMPILE_CLEARANCE:?coordinator checkpoint reference required}"
P=/storage/project/r-cs207-0/awallace43
compile_pin=587e172a33afaadb00eb7f574a5089533f7fc755
origin="$P/runs/saptdft-cuest-disp0/build-$compile_pin-14196330"
env="$origin/environment"
source_root="${DISP0_GATE_SOURCE:?immutable reviewed harness source required}"
pin="${DISP0_GATE_PIN:?exact harness commit required}"
prefix="${DISP0_BUILD_ROOT:?new recovered artifact prefix required}"
science="${DISP0_SCIENCE_SOURCE:?original immutable compilation source required}"
[[ "${DISP0_BUILD_PIN:?}" == "$compile_pin" && "${DISP0_BUILD_JOB_ID:?}" == 14196330 ]]
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$(git -C "$source_root" rev-parse HEAD)" == "$pin" ]]
git -C "$source_root" diff HEAD --quiet
git -C "$source_root" diff --cached --quiet
git -C "$source_root" diff --exit-code "$compile_pin" "$pin" -- psi4/
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH \
      PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME="$P/software/toolchains/cuda13" CUDA_PATH="$P/software/toolchains/cuda13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$env/lib"
export PATH="$env/bin:$CUDA_HOME/bin:$PATH"
admission="${DISP0_RECOVERED_ADMISSION:?separately verified recovery admission required}"
expected="${DISP0_RECOVERED_ADMISSION_SHA256:?}"
# Read-only check here. Full admission creation/revalidation is a separate
# operation; diagnostic preparation NEVER writes the deployed prefix.
"$env/bin/python" - "$admission" "$expected" "$prefix" "$pin" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, checksum, prefix, pin = sys.argv[1:]
raw = Path(path).read_bytes()
assert hashlib.sha256(raw).hexdigest() == checksum
document = json.loads(raw)
assert document["status"] == "RECOVERED_RUNTIME_PREPARATION_ONLY"
assert document["actual_gpu_correctness_verified"] is False
assert document["admission_pin"] == pin and document["prefix"] == str(Path(prefix).resolve())
assert document["compile_pin"] == "587e172a33afaadb00eb7f574a5089533f7fc755"
assert document["origin_compilation_job"] == "14196330"
assert Path(path).resolve().parent == Path(prefix).resolve() / "recovery-proofs"
PY
prep="$P/runs/saptdft-cuest-disp0/recovered-gate-prep-$pin"
mkdir "$prep"
export TMPDIR="$prep/tmp"
mkdir "$TMPDIR"
cxx="$env/bin/x86_64-conda-linux-gnu-c++"
[[ "$("$cxx" -dumpfullversion)" == 14.3.0 ]]
[[ "$("$CUDA_HOME/bin/nvcc" --version)" == *"V13.3.73"* ]]
[[ -x "$("$cxx" -print-prog-name=cc1plus)" ]]
sha256sum "$source_root/devtools/disp0/cublas_trace.cc" \
    "$source_root/devtools/disp0/prepare_recovered_gpu_gate.sh" > "$prep/helper-source-sha256.txt"
sha256sum "$cxx" "$("$cxx" -print-prog-name=cc1plus)" \
    "$CUDA_HOME/bin/nvcc" "$CUDA_HOME/include/cuda_runtime.h" \
    "$CUDA_HOME/include/cuda_runtime_api.h" "$CUDA_HOME/include/cublas_v2.h" \
    > "$prep/compiler-header-identities.txt"
"$cxx" --version > "$prep/compiler-version.txt"
"$CUDA_HOME/bin/nvcc" --version > "$prep/nvcc-version.txt"
"$cxx" -dM -E -x c++ -I"$CUDA_HOME/include" -include cuda_runtime_api.h /dev/null \
    > "$prep/selected-header-macros.txt"
grep -Fx '#define CUDART_VERSION 13030' "$prep/selected-header-macros.txt"
# Existing generic cuBLAS forwarding/identity diagnostic only, no scientific algorithm.
command=("$cxx" -std=c++20 -O2 -shared -fPIC "$source_root/devtools/disp0/cublas_trace.cc" \
    -MD -MF "$prep/tracer-dependencies.d" \
    -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -L"$CUDA_HOME/lib" \
    -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib" \
    -Wl,--no-as-needed -lcublas -Wl,--as-needed -lcudart -ldl -o "$prep/libdisp0_cublas_trace.so")
printf '%q ' "${command[@]}" > "$prep/compile-command.txt"
printf '\n' >> "$prep/compile-command.txt"
/usr/bin/time -v -o "$prep/compiler-resource.txt" "${command[@]}"
readelf -d "$prep/libdisp0_cublas_trace.so" > "$prep/tracer-elf.txt"
nm -D "$prep/libdisp0_cublas_trace.so" > "$prep/tracer-symbols.txt"
sha256sum "$prep/libdisp0_cublas_trace.so" | tee "$prep/tracer-sha256.txt"
"$env/bin/python" - "$prep" "$pin" "$DISP0_RECOVERY_DIAGNOSTIC_COMPILE_CLEARANCE" <<'PY'
import ctypes, hashlib, json, os, resource, sys, time
from pathlib import Path
prep, pin, clearance = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
library = ctypes.CDLL(str(prep / "libdisp0_cublas_trace.so"), mode=ctypes.RTLD_GLOBAL)
function = library.disp0_cublas_forwarding_identity
function.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
function.restype = ctypes.c_int
buffer = ctypes.create_string_buffer(8192)
assert function(buffer, len(buffer)) == 0, buffer.value
result = json.loads(buffer.value)
assert result["status"] == 0 and "libcublas.so.13" in Path(result["library"]).resolve().name
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
maps = sorted({str(Path(line.split()[-1]).resolve())
               for line in Path("/proc/self/maps").read_text().splitlines()
               if "/" in line and any(n in line for n in ("libcublas", "libcudart", "libstdc++"))})
assert any("libcudart.so.13" in p for p in maps)
assert any("libcublas.so.13" in p for p in maps)
files = {str(path): digest(path) for path in prep.iterdir() if path.is_file()}
report = {"status": "DIAGNOSTIC_ELF_LOAD_RESOLUTION_ONLY", "recipe_pin": pin,
          "clearance": clearance, "scope": "No CUDA device/API/numerical execution or readiness claim",
          "limits": {"wall_seconds": 110, "address_space_bytes": 4 * 1024**3,
                     "output_file_bytes": 64 * 1024**2, "threads": 1},
          "aggregate_elapsed_seconds_upper_resolution1s":
              time.time() - float(os.environ["DISP0_DIAGNOSTIC_START_EPOCH"]),
          "resolver_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "forwarding_identity": result, "output_sha256": files,
          "mapped_library_sha256": {p: digest(p) for p in maps}}
with (prep / "diagnostic-preparation.json").open("x") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
print("RESOLUTION_ONLY_NO_CUDA_EXECUTION", json.dumps(result))
PY
printf 'RECOVERED_GATE_PREP=%s\nADMISSION=%s\nADMISSION_SHA256=%s\nHARNESS_PIN=%s\nCLEARANCE=%s\n' \
    "$prep" "$admission" "$(sha256sum "$admission" | awk '{print $1}')" \
    "$pin" "$DISP0_RECOVERY_DIAGNOSTIC_COMPILE_CLEARANCE"
