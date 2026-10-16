#!/usr/bin/env bash
# Compare both policies in fresh serial workers, reusing the exact pinned core.
set -euo pipefail
root=$(realpath "$1")
pin=$2
science=/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft-cuest-grid-correctness-v1
environment=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
test "$(git -C "$science" rev-parse HEAD)" = c28950041bbe8bb10617264a7b6ddc56a436cda1
out="/storage/project/r-cs207-0/awallace43/runs/saptdft-grid-policy-v1/policy-$SLURM_JOB_ID"
mkdir -p "$out/metadata"
test ! -e "$out/results"
child=
term_received=0
printf 'job_id=%s\nharness_commit=%s\nbase_science=c28950041bbe8bb10617264a7b6ddc56a436cda1\nstarted_at=%s\n' \
    "$SLURM_JOB_ID" "$pin" "$(date -Is)" > "$out/metadata/runtime.env"
finish() {
    rc=$?
    trap - EXIT
    printf 'exit_code=%s\nterm_received=%s\nfinished_at=%s\n' "$rc" "$term_received" "$(date -Is)" >> "$out/metadata/runtime.env"
    if [[ $rc == 0 && $term_received == 0 ]]; then
        printf 'Execution complete; inspect accuracy/failures, not only allocation state.\n' > "$out/metadata/COMPLETED"
    else printf 'Execution incomplete.\n' > "$out/metadata/FAILED"; fi
    exit "$rc"
}
terminate() { term_received=1; if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi; exit 143; }
trap finish EXIT
trap terminate TERM INT
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$science/build_grid_host/stage/lib"
export LD_LIBRARY_PATH="$environment/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
scratch="/storage/scratch1/3/$USER/saptdft-grid-policy/$SLURM_JOB_ID"
mkdir -p "$scratch"
export TMPDIR="$scratch" SCRATCH="$scratch" PSI_SCRATCH="$scratch"
scontrol show job -o "$SLURM_JOB_ID" > "$out/allocation.txt"
grep -q 'QOS=embers ' "$out/allocation.txt"
grep -q 'Account=gts-cs207-chemx ' "$out/allocation.txt"
lscpu > "$out/cpu.txt"
cd "$out"
"$environment/bin/python" "$root/devtools/benchmarks/grid_policy_probe.py" --output "$out/probe" > "$out/probe.log" 2>&1 &
child=$!
wait "$child"
"$environment/bin/python" -u "$root/devtools/benchmarks/grid_policy_pilot.py" campaign \
    --expected-package "$science/build_grid_host/stage/lib/psi4" --variant cpu \
    --systems water peptide --threads "$OMP_NUM_THREADS" --memory "80 GiB" \
    --arms fixed75 fixed99 \
        tiny10_early1 tiny10_full2 tiny12_early1 tiny12_full2 \
        tiny19_early1 tiny19_full2 coarse75_early1 coarse75_full2 \
        tiny08_early1 tiny08_full2 tiny05_early1 tiny05_full2 \
        tiny12_full1 tiny12_early2 \
    --budget 6600 --case-timeout 360 --output "$out/results" > "$out/pilot.log" 2>&1 &
child=$!
wait "$child"
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test "$(git -C "$science" rev-parse HEAD)" = c28950041bbe8bb10617264a7b6ddc56a436cda1
