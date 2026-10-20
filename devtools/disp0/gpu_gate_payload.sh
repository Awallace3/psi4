#!/usr/bin/env bash
set -euo pipefail
umask 027
P=/storage/project/r-cs207-0/awallace43
source_root="${DISP0_GATE_SOURCE:?}"
pin="${DISP0_GATE_PIN:?}"
build="${DISP0_BUILD_ROOT:?}"
build_pin="${DISP0_BUILD_PIN:?}"
build_job="${DISP0_BUILD_JOB_ID:?}"
science="${DISP0_SCIENCE_SOURCE:?}"
prep="${DISP0_GATE_PREP:?}"
run="${DISP0_GATE_RUN:?}"
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$(git -C "$source_root" rev-parse HEAD)" == "$pin" ]]
[[ "$build_pin" =~ ^[0-9a-f]{40}$ && "$build_job" =~ ^[0-9]+$ ]]
[[ "$build" == "$P/runs/saptdft-cuest-disp0/build-$build_pin-$build_job" ]]
[[ "$(git -C "$science" rev-parse HEAD)" == "$build_pin" ]]
git -C "$science" diff --quiet
git -C "$science" diff --cached --quiet
git -C "$source_root" diff --quiet
git -C "$source_root" diff --cached --quiet
git -C "$source_root" diff --exit-code "$build_pin" "$pin" -- psi4/
mkdir "$run"
export TMPDIR="/storage/scratch1/3/awallace43/saptdft-cuest-disp0/gpu-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0 cuda/12.6.1
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH PYTHONPATH PYTHONHOME
export CUDA_HOME="$P/software/toolchains/cuda13" CUDA_PATH="$P/software/toolchains/cuda13"
env="$build/environment"
export PATH="$env/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$build/einsums-stage/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib:$env/lib"
export PYTHONPATH="$build/einsums-stage/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
eval "$("$build/psi4-build/stage/bin/psi4" --psiapi)"
export DISP0_TRACE_LIB="$prep/libdisp0_cublas_trace.so"
export DISP0_TRACE_SHA256
DISP0_TRACE_SHA256="$(awk '{print $1}' "$prep/tracer-sha256.txt")"
[[ "$(sha256sum "$DISP0_TRACE_LIB" | awk '{print $1}')" == "$DISP0_TRACE_SHA256" ]]
export DISP0_CUBLAS_TRACE="$run/cublas-api-trace.jsonl"
export DISP0_TRACE_PHASE=initialization
cd "$run"
# No installs or scientific builds inside this GPU allocation. The diagnostic
# interposer only forwards cuBLAS and records real API calls/statuses.
LD_PRELOAD="$DISP0_TRACE_LIB" "$env/bin/python" "$source_root/devtools/disp0/gpu_gate.py" \
    --out "$run" "--einsums:profile:save=$run/einsums-auto-profile.json" \
    --einsums:profile:disable=false --einsums:profile:report=false \
    --einsums:debug:no-attach-debugger
"$env/bin/python" - "$run/gpu-gate-result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["status"] == "PASS" and not r["failures"], r
assert r["native_cublas_trace_evidence"]["actual_production_cublas_calls"] > 0
print("GPU_SCALAR_DISPERSION_NUMERICAL_GATE_PASS_NOT_FSAPT_COVERAGE")
PY
