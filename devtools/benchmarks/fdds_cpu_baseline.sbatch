#!/usr/bin/env bash
#SBATCH --job-name=fdds-real-cpu-baseline
#SBATCH --account=gts-cs207
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@60

# Required absolute paths/pins are supplied with sbatch --export.
set -euo pipefail
: "${FDDS_PAYLOAD_ROOT:?}" "${FDDS_PAYLOAD_PIN:?}" "${FDDS_RUN_ROOT:?}"
: "${FDDS_STAGE_ROOT:?}" "${FDDS_BINARY_PIN:?}" "${FDDS_CORE_SHA256:?}"
test "$(git -C "$FDDS_PAYLOAD_ROOT" rev-parse HEAD)" = "$FDDS_PAYLOAD_PIN"
test -z "$(git -C "$FDDS_PAYLOAD_ROOT" status --porcelain --untracked-files=no)"
test "$(git -C "$FDDS_STAGE_ROOT" rev-parse HEAD)" = "$FDDS_BINARY_PIN"
test -z "$(git -C "$FDDS_STAGE_ROOT" status --porcelain --untracked-files=no)"
git -C "$FDDS_PAYLOAD_ROOT" diff --exit-code "$FDDS_BINARY_PIN" "$FDDS_PAYLOAD_PIN" -- \
    psi4 cmake CMakeLists.txt external
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$env/lib:$env/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$FDDS_STAGE_ROOT/build_fdds_validation/stage/lib"
export PSIDATADIR="$FDDS_STAGE_ROOT/build_fdds_validation/stage/share/psi4"
export TMPDIR="/storage/scratch1/3/$USER/fdds-perf/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
out="$FDDS_RUN_ROOT/cpu-$SLURM_JOB_ID"
mkdir "$out"
cd "$out"
{
    printf 'payload=%s\nbinary=%s\ncore_sha256=%s\n' \
        "$FDDS_PAYLOAD_PIN" "$FDDS_BINARY_PIN" "$FDDS_CORE_SHA256"
    hostname
    lscpu
    scontrol show job "$SLURM_JOB_ID"
} > provenance.txt
child=
terminated=0
on_term() {
    terminated=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
trap on_term TERM INT
for system in peptide nanotube protein83; do
    timeout --signal=TERM --kill-after=30s 4500s \
        "$env/bin/python" "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_benchmark.py" \
        --prepare --suite "$FDDS_RUN_ROOT/geometries" --system "$system" \
        --cache "$FDDS_RUN_ROOT/caches/$system" --output "$out/$system" \
        --threads "$SLURM_CPUS_PER_TASK" --memory "96 GiB" \
        --binary-pin "$FDDS_BINARY_PIN" --expected-core-sha256 "$FDDS_CORE_SHA256" \
        --expected-package "$FDDS_STAGE_ROOT/build_fdds_validation/stage/lib/psi4" \
        > "$out/$system.log" 2>&1 &
    child=$!
    rc=0
    wait "$child" || rc=$?
    child=
    if ((terminated)); then exit 143; fi
    if ((rc)); then exit "$rc"; fi
done
"$env/bin/python" - <<'PY'
import json
from pathlib import Path
results = [json.loads(Path(s, "result.json").read_text()) for s in ("peptide", "nanotube", "protein83")]
assert all(r["ok"] and r["arm"] == "cpu" and r["gpu_timer_calls"] == 0 for r in results)
Path("VERIFIED.json").write_text(json.dumps(results, indent=2) + "\n")
PY
test "$terminated" = 0
printf '%s\n' "$FDDS_PAYLOAD_PIN" > COMPLETED
