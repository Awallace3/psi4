#!/usr/bin/env bash
# Run only inside the owner's serialized/exclusive shared GPU step.
# Existing scientific stage, caches and benchmark code stay frozen/read-only.
set -euo pipefail
system=${1:?system}
phase=${2:?first or repeat}
case "$system" in peptide|nanotube|protein83) ;; *) exit 2;; esac
case "$phase" in first) controls=(cpu-0 gpu-0);; repeat) controls=(gpu-1 cpu-1);; *) exit 2;; esac
: "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
test "$SLURM_JOB_ID" = 14214205
test "$SLURM_CPUS_PER_TASK" = 8
payload=/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-opt-gpu-recovery-harness-20261020-v1
stage=/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-opt-cuda-compile-20261020-v2
payload_pin=c289c502d55627eabf99acdc127a28334416704d
binary_pin=fd2c09cd0cfbfce61a311508ba6c53039025179a
core_hash=92e5bff6641dbb2f4790938e9886fc8038d6d03527eb26c6934ddeecabac5a7f
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
cache=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-optimization-01/caches
run=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-interactive-01
tag=${FDDS_BENCH_TAG:-}
if [[ -n "$tag" ]]; then
    [[ "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]
    tag="-$tag"
fi
out=$run/matched-$SLURM_JOB_ID$tag
for pair in "$payload:$payload_pin" "$stage:$binary_pin"; do
    tree=${pair%:*} pin=${pair##*:}
    test "$(git -C "$tree" rev-parse HEAD)" = "$pin"
    test -z "$(git -C "$tree" status --porcelain --untracked-files=no)"
done
test -s "$run/validation-$SLURM_JOB_ID/VERIFIED.json"
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$env/lib:$env/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$stage/build_fdds_validation/stage/lib"
export PSIDATADIR="$stage/build_fdds_validation/stage/share/psi4"
export TMPDIR="${FDDS_BENCH_SCRATCH:-/storage/scratch1/3/$USER/fdds-interactive/$SLURM_JOB_ID}"
if [[ -n "${FDDS_BENCH_SCRATCH:-}" ]]; then
    [[ "$TMPDIR" == "/var/tmp/awallace43-fdds-$SLURM_JOB_ID-v2" ]]
    test -s "$TMPDIR/STORAGE_VERIFIED.json"
fi
mkdir -p "$TMPDIR" "$out"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" PYTHONFAULTHANDLER=1
cd "$out"
# Unique per-step provenance; never overwrite an earlier attempted phase.
test ! -e "$out/$system-$phase-provenance.txt"
{
    printf 'binary=%s\npayload=%s\ncore_sha256=%s\nphase=%s\n' \
        "$binary_pin" "$payload_pin" "$core_hash" "$phase"
    printf 'co_tenancy=serialized shared allocation; exclusive whole-step timing\n'
    printf 'scratch=%s\n' "$TMPDIR"
    df -PT "$TMPDIR"
    hostname
    scontrol show job -o "$SLURM_JOB_ID"
    nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "$system-$phase-provenance.txt"
for control in "${controls[@]}"; do
    test ! -e "$out/$system-$control"
    arm=${control%-*}
    timeout --signal=TERM --kill-after=30s 10500s \
        "$env/bin/python" "$payload/devtools/benchmarks/fdds_benchmark.py" \
        --system "$system" --cache "$cache/$system" \
        --output "$out/$system-$control" --arm "$arm" \
        --threads 8 --memory "96 GiB" --binary-pin "$binary_pin" \
        --expected-core-sha256 "$core_hash" \
        --expected-package "$stage/build_fdds_validation/stage/lib/psi4" \
        --require-gpu-exchange \
        > "$out/$system-$control.log" 2>&1
done
if [[ "$phase" == first ]]; then
    extra=(--first-pair)
else
    extra=()
fi
"$env/bin/python" "$payload/devtools/benchmarks/fdds_compare.py" \
    --root "$out" --system "$system" "${extra[@]}"
