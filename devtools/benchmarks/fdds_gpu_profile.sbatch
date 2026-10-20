#!/usr/bin/env bash
#SBATCH --job-name=fdds-matched-gpu-profile
#SBATCH --account=gts-cs207
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@60

set -euo pipefail
: "${FDDS_PAYLOAD_ROOT:?}" "${FDDS_PAYLOAD_PIN:?}" "${FDDS_RUN_ROOT:?}" "${FDDS_CPU_JOB_ID:?}"
: "${FDDS_STAGE_ROOT:?}" "${FDDS_BINARY_PIN:?}" "${FDDS_CORE_SHA256:?}"
test -s "$FDDS_RUN_ROOT/cpu-$FDDS_CPU_JOB_ID/COMPLETED"
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
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" PYTHONFAULTHANDLER=1
out="$FDDS_RUN_ROOT/matched-$SLURM_JOB_ID"
mkdir "$out"
cd "$out"
{
    printf 'payload=%s\nbinary=%s\ncore_sha256=%s\n' \
        "$FDDS_PAYLOAD_PIN" "$FDDS_BINARY_PIN" "$FDDS_CORE_SHA256"
    hostname
    lscpu
    scontrol show job "$SLURM_JOB_ID"
    nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > provenance.txt
child=
terminated=0
on_term() {
    terminated=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
trap on_term TERM INT
for system in peptide nanotube protein83; do
    # ABBA controls: each arm is a new process, same physical cache/host/threads.
    for control in cpu-0 gpu-0 gpu-1 cpu-1; do
        arm=${control%-*}
        timeout --signal=TERM --kill-after=30s 3000s \
            "$env/bin/python" "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_benchmark.py" \
            --system "$system" --cache "$FDDS_RUN_ROOT/caches/$system" \
            --output "$out/$system-$control" --arm "$arm" \
            --threads "$SLURM_CPUS_PER_TASK" --memory "96 GiB" \
            --binary-pin "$FDDS_BINARY_PIN" --expected-core-sha256 "$FDDS_CORE_SHA256" \
            --expected-package "$FDDS_STAGE_ROOT/build_fdds_validation/stage/lib/psi4" \
            > "$out/$system-$control.log" 2>&1 &
        child=$!
        rc=0
        wait "$child" || rc=$?
        child=
        if ((terminated)); then exit 143; fi
        if ((rc)); then exit "$rc"; fi
    done
done
"$env/bin/python" - <<'PY'
import json
from pathlib import Path
from statistics import median
summary = {}
for system in ("peptide", "nanotube", "protein83"):
    rows = [json.loads(Path(f"{system}-{arm}-{i}", "result.json").read_text())
            for arm in ("cpu", "gpu") for i in range(2)]
    assert all(r["ok"] for r in rows)
    for key in ("cache_sha256", "metadata_sha256", "options", "binary_pin",
                "core_sha256", "threads", "memory", "hostname"):
        assert all(r[key] == rows[0][key] for r in rows), key
    errors = {k: max(abs(r["energies_Eh"][k] - rows[0]["energies_Eh"][k]) for r in rows)
              for k in rows[0]["energies_Eh"]}
    assert max(errors.values()) <= 1.e-5, errors
    assert all((r["gpu_timer_calls"] > 0) == (r["arm"] == "gpu") for r in rows)
    times = {arm: [r["fdds_seconds"] for r in rows if r["arm"] == arm] for arm in ("cpu", "gpu")}
    summary[system] = {"seconds": times, "cpu_over_gpu": median(times["cpu"]) / median(times["gpu"]),
                       "max_errors_Eh": errors}
Path("VERIFIED.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
test "$terminated" = 0
printf '%s\n' "$FDDS_PAYLOAD_PIN" > COMPLETED
