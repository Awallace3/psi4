#!/usr/bin/env bash
# Explicit recovered-artifact route; original failed-build admission is untouched.
set -euo pipefail
umask 077
P=/storage/project/r-cs207-0/awallace43
compile_pin=587e172a33afaadb00eb7f574a5089533f7fc755
origin="$P/runs/saptdft-cuest-disp0/build-$compile_pin-14196330"
source_root="${DISP0_GATE_SOURCE:?}"
prefix="${DISP0_BUILD_ROOT:?}"
pin="${DISP0_GATE_PIN:?}"
run="${DISP0_GATE_RUN:?}"
prep="${DISP0_GATE_PREP:?}"
: "${DISP0_RECOVERED_ADMISSION:?}" "${DISP0_RECOVERED_ADMISSION_SHA256:?}"
[[ "${DISP0_BUILD_PIN:?}" == "$compile_pin" && "${DISP0_BUILD_JOB_ID:?}" == 14196330 ]]
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$(git -C "$source_root" rev-parse HEAD)" == "$pin" ]]
git -C "$source_root" diff HEAD --quiet
git -C "$source_root" diff --cached --quiet
git -C "$source_root" diff --exit-code "$compile_pin" "$pin" -- psi4/
mkdir "$run"
export TMPDIR="/storage/scratch1/3/awallace43/saptdft-cuest-disp0/gpu-$SLURM_JOB_ID"
mkdir "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH \
      PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD
export CUDA_HOME="$P/software/toolchains/cuda13" CUDA_PATH="$P/software/toolchains/cuda13"
env="$origin/environment"  # Read-only dependency environment; never relabel/rebuild it.
export PATH="$env/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/einsums-stage/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib:$env/lib"
export PYTHONPATH="$prefix/einsums-stage/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
eval "$("$prefix/psi4-build/stage/bin/psi4" --psiapi)"
export DISP0_TRACE_LIB="$prep/libdisp0_cublas_trace.so"
export DISP0_TRACE_SHA256
DISP0_TRACE_SHA256="$(awk '{print $1}' "$prep/tracer-sha256.txt")"
[[ "$(sha256sum "$DISP0_TRACE_LIB" | awk '{print $1}')" == "$DISP0_TRACE_SHA256" ]]
export DISP0_CUBLAS_TRACE="$run/cublas-api-trace.jsonl"
export DISP0_TRACE_PHASE=initialization
cd "$run"
# This gate includes small native/eager/defaultCPU signed physical oracles in
# the ACTUAL recovered stage before accepting GPU results. No builds/repairs.
LD_PRELOAD="$DISP0_TRACE_LIB" "$env/bin/python" "$source_root/devtools/disp0/gpu_gate.py" \
    --out "$run" "--einsums:profile:save=$run/einsums-auto-profile.json" \
    --einsums:profile:disable=false --einsums:profile:report=false \
    --einsums:debug:no-attach-debugger
"$env/bin/python" - "$run/gpu-gate-result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["status"] == "PASS" and not r["failures"], r
assert r["native_cublas_trace_evidence"]["actual_production_cublas_calls"] > 0
assert r["recovered_runtime_admission"]["origin_compilation_status"].startswith("FAILED1:0")
assert all(any(c["mode"] == mode and c["pass"] for c in r["comparisons"])
           for mode in ("cpu-None", "cpu-cpu"))
print("RECOVERED_STAGE_GPU_SCALAR_NUMERICAL_GATE_PASS_NOT_FSAPT_COVERAGE")
PY
