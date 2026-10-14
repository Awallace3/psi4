#!/usr/bin/env bash
# Runtime-only matched CPU/GPU experiment. No scientific rebuild or environment mutation.
set -euo pipefail
root=$(realpath "$1")
pin=$2
science=$(realpath "$3")
science_pin=$4
validation=$(realpath "$5")
out=$(realpath "$6")
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
test -n "${SLURM_JOB_ID:-}"
for pair in "$root:$pin" "$science:$science_pin"; do
    test "$(git -C "${pair%:*}" rev-parse HEAD)" = "${pair##*:}"
    test -z "$(git -C "${pair%:*}" status --porcelain --untracked-files=no)"
done
mkdir -p "$out"
"$env/bin/python" "$root/devtools/benchmarks/grid_workload_gate.py" \
    --validation "$validation" --science "$science" --pin "$science_pin" \
    --output "$out/supported-workload.json"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# Phoenix's private /tmp bind mounts have disappeared during multi-hour jobs.
# Use one persistent job-owned Lustre directory for all three routes.
scratch_base="/storage/scratch1/3/$USER/saptdft-cpu-gpu-grids/$SLURM_JOB_ID"
mkdir -p "$scratch_base"
TMPDIR=$(mktemp -d "$scratch_base/runtime.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "$out/gpu.txt"
grep -q H200 "$out/gpu.txt"
lscpu > "$out/cpu.txt"
scontrol show job -o "$SLURM_JOB_ID" > "$out/allocation.txt"
grep -q 'Account=gts-cs207 ' "$out/allocation.txt"
grep -q 'QOS=inferno ' "$out/allocation.txt"
cd "$out"
"$env/bin/python" "$root/devtools/benchmarks/cpu_gpu_grids.py" \
    --output "$out/timing" --receipt "$out/supported-workload.json" --science "$science" \
    --scientific-commit "$science_pin" --environment "$env" \
    --threads "$OMP_NUM_THREADS" --repeats 3 --memory "112 GiB" > "$out/timing.log" 2>&1
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test "$(git -C "$science" rev-parse HEAD)" = "$science_pin"
printf 'All108 timings completed. Accuracy gates are recorded separately.\n' > "$out/VERIFIED"
