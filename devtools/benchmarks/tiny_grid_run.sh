#!/usr/bin/env bash
# Execute in one bounded CPU allocation; leave existing science/builds untouched.
set -euo pipefail
root=$(realpath "$1")
pin=$2
science=$(realpath "$3")
runbase=$4
science_pin=c28950041bbe8bb10617264a7b6ddc56a436cda1
environment=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
test -n "${SLURM_JOB_ID:-}"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test "$(git -C "$science" rev-parse HEAD)" = "$science_pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
test -z "$(git -C "$science" status --porcelain --untracked-files=no)"
out="$runbase/pilot-$SLURM_JOB_ID"
mkdir -p "$out/metadata"
test ! -e "$out/results"
runtime="$out/metadata/runtime.env"
printf 'job_id=%s\nharness_commit=%s\nscientific_commit=%s\nstarted_at=%s\n' \
    "$SLURM_JOB_ID" "$pin" "$science_pin" "$(date -Is)" > "$runtime"
term_received=0
child=
finish() {
    rc=$?
    trap - EXIT
    printf 'exit_code=%s\nterm_received=%s\nfinished_at=%s\n' \
        "$rc" "$term_received" "$(date -Is)" >> "$runtime"
    if [[ $rc == 0 && $term_received == 0 ]]; then
        printf 'Execution finished; inspect failed/accuracy-gated arms in summary.json.\n' > "$out/metadata/COMPLETED"
    else
        printf 'exit=%s term_received=%s\n' "$rc" "$term_received" > "$out/metadata/FAILED"
    fi
    exit "$rc"
}
terminate() {
    term_received=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
    exit 143
}
trap finish EXIT
trap terminate TERM INT
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$science/build_grid_host/stage/lib"
export LD_LIBRARY_PATH="$environment/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$environment/bin:$PATH"
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
scratch="/storage/scratch1/3/$USER/saptdft-tiny-grid/$SLURM_JOB_ID"
mkdir -p "$scratch"
export TMPDIR="$scratch" SCRATCH="$scratch" PSI_SCRATCH="$scratch"
lscpu > "$out/cpu.txt"
scontrol show job -o "$SLURM_JOB_ID" > "$out/allocation.txt"
grep -q 'QOS=embers ' "$out/allocation.txt"
grep -q 'Account=gts-cs207-chemx ' "$out/allocation.txt"
cd "$out"
"$environment/bin/python" -u "$root/devtools/benchmarks/tiny_grid_pilot.py" campaign \
    --expected-package "$science/build_grid_host/stage/lib/psi4" --variant cpu \
    --systems water peptide protein83 --threads "$OMP_NUM_THREADS" --memory "80 GiB" \
    --budget 6600 --case-timeout 1500 --output "$out/results" > "$out/pilot.log" 2>&1 &
child=$!
wait "$child"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test "$(git -C "$science" rev-parse HEAD)" = "$science_pin"
