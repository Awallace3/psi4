#!/usr/bin/env bash
# Immutable runtime harness; scientific libraries/environments are never rebuilt.
set -euo pipefail
root=$(realpath "$1")
pin=$2
science=$(realpath "$3")
science_pin=$4
validation=$(realpath "$5")
out=$(realpath "$6")
mode=$7
selection=${8:-}
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
test -n "${SLURM_JOB_ID:-}"
for pair in "$root:$pin" "$science:$science_pin"; do
    test "$(git -C "${pair%:*}" rev-parse HEAD)" = "${pair##*:}"
    test -z "$(git -C "${pair%:*}" status --porcelain --untracked-files=no)"
done
mkdir -p "$out"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
scratch_base="/storage/scratch1/3/$USER/saptdft-protein157-grids/$SLURM_JOB_ID"
mkdir -p "$scratch_base"
TMPDIR=$(mktemp -d "$scratch_base/runtime.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
export PYTHONPATH="$science/build_grid_cuda/stage/lib"
export LD_LIBRARY_PATH="/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda/lib64:$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$env/bin:$PATH"
unset GCC_ROOT CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$out/gpu.txt"
grep -q H200 "$out/gpu.txt"
lscpu > "$out/cpu.txt"
scontrol show job -o "$SLURM_JOB_ID" > "$out/allocation.txt"
grep -q 'Account=gts-cs207 ' "$out/allocation.txt"
grep -q 'QOS=inferno ' "$out/allocation.txt"
cd "$out"
"$env/bin/python" "$root/devtools/benchmarks/grid_workload_gate.py" \
    --validation "$validation" --science "$science" --pin "$science_pin" \
    --output "$out/supported-workload.json"
"$env/bin/python" -c 'import hashlib,json,pathlib,sys,psi4
p=pathlib.Path(psi4.__file__).resolve().parent
assert p == pathlib.Path(sys.argv[1]).resolve()
h=hashlib.sha256(pathlib.Path(psi4.core.__file__).read_bytes()).hexdigest()
r=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert h == r["backends"]["cuda"]["core_sha256"]
print(json.dumps(dict(package=str(p),core_sha256=h)))' \
    "$science/build_grid_cuda/stage/lib/psi4" "$out/supported-workload.json" > "$out/binary-cuda.json"
common=(--expected-package "$science/build_grid_cuda/stage/lib/psi4"
        --scientific-commit "$science_pin" --threads "$OMP_NUM_THREADS")
if [[ "$mode" == query ]]; then
    # A standalone SDK diagnostic, not a rebuilt or modified Psi4 scientific core.
    "$env/bin/x86_64-conda-linux-gnu-c++" -std=c++17 -O2 \
        -I"$env/include" -I"$env/targets/x86_64-linux/include" \
        "$root/devtools/benchmarks/cuest_df_workspace_query.cc" \
        -L"$env/lib" -L"$env/targets/x86_64-linux/lib" \
        -Wl,-rpath,"$env/lib" -Wl,-rpath,"$env/targets/x86_64-linux/lib" \
        -lcuest -lcudart -lcublas -lcusolver -o "$out/cuest_df_workspace_query"
    "$env/bin/python" "$root/devtools/benchmarks/protein157_grids.py" query \
        --output "$out/capacity" --query-executable "$out/cuest_df_workspace_query" \
        --headroom-gib 16 "${common[@]}" > "$out/capacity.log" 2>&1
elif [[ "$mode" == campaign ]]; then
    test -f "$selection"
    readarray -t selected < <("$env/bin/python" -c \
        'import json,sys; r=json.load(open(sys.argv[1]))["selected"]; print(r["basis"]); print(r["nbf"])' "$selection")
    for arm in fixed99 fixed75 coarse75; do
        "$env/bin/python" "$root/devtools/benchmarks/grid_memory.py" run \
            --output "$out/memory-$arm" --interval 0.2 --timeout 3600 \
            --script "$root/devtools/benchmarks/protein157_grids.py" -- \
            worker --output "$out/diagnostic-$arm" --basis "${selected[0]}" \
            --expected-nbf "${selected[1]}" --arm "$arm" --variant cuda \
            --memory "112 GiB" --diagnostic "${common[@]}" > "$out/memory-$arm.log" 2>&1
    done
    "$env/bin/python" "$root/devtools/benchmarks/protein157_grids.py" campaign \
        --output "$out/timing" --selection "$selection" --repeats 3 \
        --memory "112 GiB" --variant cuda "${common[@]}" > "$out/timing.log" 2>&1
else
    printf 'Unknown mode: %s\n' "$mode" >&2
    exit 2
fi
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test "$(git -C "$science" rev-parse HEAD)" = "$science_pin"
printf '%s execution complete; inspect numerical accuracy separately.\n' "$mode" > "$out/VERIFIED"
