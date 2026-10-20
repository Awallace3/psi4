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
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@60

set -euo pipefail
: "${FDDS_PAYLOAD_ROOT:?}" "${FDDS_PAYLOAD_PIN:?}" "${FDDS_RUN_ROOT:?}"
: "${FDDS_STAGE_ROOT:?}" "${FDDS_BINARY_PIN:?}" "${FDDS_CORE_SHA256:?}"
if [[ -n "${FDDS_CPU_JOB_ID:-}" ]]; then
    test -s "$FDDS_RUN_ROOT/cpu-$FDDS_CPU_JOB_ID/COMPLETED"
fi
cache_root="${FDDS_CACHE_ROOT:-$FDDS_RUN_ROOT/caches}"
for system in peptide nanotube protein83; do
    test -s "$cache_root/$system/metadata.json"
    test -s "$cache_root/$system/orbitals.npz"
done
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
grep -q 'NVIDIA H200' provenance.txt
extras=()
if [[ "${FDDS_GPU_EXCHANGE_REQUIRED:-0}" == 1 ]]; then
    extras=(--require-gpu-exchange)
fi
child=
terminated=0
on_term() {
    terminated=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
trap on_term TERM INT
for pass in 0 1; do
    # Save one complete CPU/GPU pair for every system before costly repeats.
    # For each system the overall order remains CPU/GPU/GPU/CPU (ABBA).
    if [[ "$pass" == 0 ]]; then controls=(cpu-0 gpu-0); else controls=(gpu-1 cpu-1); fi
    for system in peptide nanotube protein83; do
      for control in "${controls[@]}"; do
        arm=${control%-*}
        timeout --signal=TERM --kill-after=30s 10800s \
            "$env/bin/python" "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_benchmark.py" \
            --system "$system" --cache "$cache_root/$system" \
            --output "$out/$system-$control" --arm "$arm" \
            --threads "$SLURM_CPUS_PER_TASK" --memory "96 GiB" \
            --binary-pin "$FDDS_BINARY_PIN" --expected-core-sha256 "$FDDS_CORE_SHA256" \
            --expected-package "$FDDS_STAGE_ROOT/build_fdds_validation/stage/lib/psi4" \
            "${extras[@]}" \
            > "$out/$system-$control.log" 2>&1 &
        child=$!
        rc=0
        wait "$child" || rc=$?
        child=
        if ((terminated)); then exit 143; fi
        if ((rc)); then exit "$rc"; fi
      done
      if [[ "$pass" == 0 ]]; then
          "$env/bin/python" "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_compare.py" \
              --root "$out" --first-pair --system "$system"
      fi
    done
    if [[ "$pass" == 0 ]]; then
        "$env/bin/python" "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_compare.py" \
            --root "$out" --first-pair
    fi
done
"$env/bin/python" "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_compare.py" --root "$out"
test "$terminated" = 0
printf '%s\n' "$FDDS_PAYLOAD_PIN" > COMPLETED
