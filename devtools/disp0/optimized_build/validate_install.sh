#!/usr/bin/env bash
# Validation continuation only: read the completed installation from step0;
# never configure/build/install, mutate it, or relabel its failed audit status.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${DISP0_VALIDATOR_SOURCE:?}" "${DISP0_VALIDATOR_PIN:?}"
[[ "$SLURM_JOB_ID" == 14254131 && "${SLURM_CPUS_PER_TASK:?}" == 8 && -z "${SLURM_JOB_GPUS:-}" ]]
P=/storage/project/r-cs207-0/awallace43
compiled_source="$P/gits/psi4.disp0-opt064-20261021-04/source"
compile_pin=770ecf90108aa7783da766bf0402d8c267dcbbed
run="$P/runs/saptdft-cuest-disp0/opt064-$compile_pin-$SLURM_JOB_ID-0"
validator="$DISP0_VALIDATOR_SOURCE"
out="$P/runs/saptdft-cuest-disp0/validation-$DISP0_VALIDATOR_PIN-$SLURM_JOB_ID-$SLURM_STEP_ID"
[[ "$(git -C "$compiled_source" rev-parse HEAD)" == "$compile_pin" ]]
[[ "$(git -C "$validator" rev-parse HEAD)" == "$DISP0_VALIDATOR_PIN" ]]
[[ -z "$(git -C "$compiled_source" status --porcelain)" && -z "$(git -C "$validator" status --porcelain)" ]]
git -C "$validator" diff --exit-code 587e172a33afaadb00eb7f574a5089533f7fc755 HEAD -- psi4/
mkdir -m 0700 "$out"
status="$out/status"
signalled=0
child=
term() { signalled=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
finish() {
    code=$?
    (( signalled == 0 )) || code=143
    printf 'exit_code=%s\nsignalled=%s\ncompiled_run=%s\nvalidator_pin=%s\n' \
        "$code" "$signalled" "$run" "$DISP0_VALIDATOR_PIN" > "$status" || code=1
    trap - EXIT
    exit "$code"
}
trap term TERM INT
trap finish EXIT
env="$run/environment"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$env/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$run/einsums-stage/lib:$run/cuda13/lib64:$env/lib"
unset PYTHONHOME PYTHONPATH LD_PRELOAD DISP0_API_RUNTIME_PROOF
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
while IFS= read -r name; do unset "$name"; done < <(compgen -v EINSUMS_ || true)
export EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1
export XDG_CACHE_HOME="$out/cache" PIP_CACHE_DIR="$out/pip-cache"
export TMPDIR="/storage/scratch1/3/awallace43/disp0-validation-$SLURM_JOB_ID-$SLURM_STEP_ID"
mkdir -m 0700 "$TMPDIR"
export PSI_SCRATCH="$TMPDIR" SCRATCH="$TMPDIR"
export DISP0_OPT_RUN="$run" DISP0_OPT_PIN="$compile_pin"
export PYTHONPATH="$run/einsums-stage/lib"
eval "$("$run/psi4-build/stage/bin/psi4" --psiapi)"
snapshot() {
    "$env/bin/python" -B - "$run" "$1" <<'PY'
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1])
files=set()
for folder,pattern in ((run/'einsums-stage/lib','*.so*'),
                       (run/'einsums-stage/lib/einsums','_core*.so'),
                       (run/'einsums-stage/lib/waggle','_core*.so'),
                       (run/'psi4-build/stage/lib/psi4','core*.so')):
    files.update(p.resolve(strict=True) for p in folder.glob(pattern) if p.is_file())
assert len(files)>=5, files
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()
with Path(sys.argv[2]).open('x') as stream:
    json.dump({str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)},
              stream,indent=2,sort_keys=True)
PY
}
snapshot "$out/binaries-before.json"
"$env/bin/python" -B "$validator/devtools/disp0/build_audit.py" \
    --run "$run" --source "$compiled_source" \
    --scratch "/storage/scratch1/3/awallace43/disp0-opt064-$SLURM_JOB_ID-0" \
    --artifact-dir "$out/effective-build-audit" --out "$out/effective-build-audit.json"
cd "$out"
"$env/bin/python" -B "$validator/devtools/disp0/optimized_build/proof.py" > "$out/installed-proof.json"
"$env/bin/python" -B "$validator/devtools/disp0_einsums_api_probe.py" > "$out/api-probe.json"
snapshot "$out/binaries-after.json"
cmp "$out/binaries-before.json" "$out/binaries-after.json"
printf 'COMPLETED_INSTALL_VALIDATION_ONLY_NO_DEVICE_OR_PHYSICAL_PASS out=%s\n' "$out"
