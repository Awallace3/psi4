#!/usr/bin/env bash
# Fresh build/gates followed by matched controls on the existing shared grant.
set -euo pipefail
source_root=$(realpath "${1:?scientific source}")
source_pin=${2:?scientific pin}
out=$(realpath "${3:?fresh run}")
harness_pin=${4:?harness pin}
payload=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
harness_root=$(git -C "$payload" rev-parse --show-toplevel)
test "$(git -C "$harness_root" rev-parse HEAD)" = "$harness_pin"
test "$(git -C "$source_root" rev-parse HEAD)" = "$source_pin"
test -z "$(git -C "$harness_root" status --porcelain --untracked-files=no)"
test -z "$(git -C "$source_root" status --porcelain --untracked-files=no)"
test "$SLURM_JOB_ID" = 14214205
test "$SLURM_CPUS_PER_TASK" = 8
P=/storage/project/r-cs207-0/awallace43
env=$P/miniconda/envs/p4cuest_sapt
base=$P/gits/psi4.fdds-metric-fold-ada1a1d212-v1
base_pin=ada1a1d2128d6d6f3f5b89952b08f6ffd4ca5917
base_hash=6f8294594134277ac55ae1a3b87d9003367a451b4f8a506a16e1a2b8a823de68
scratch=/var/tmp/awallace43-fdds-14214205-v2
test -s "$scratch/STORAGE_VERIFIED.json"
test -s "$P/runs/psi4-fdds/20261020-product-profile-02/VERIFIED.json"
test "$(git -C "$base" rev-parse HEAD)" = "$base_pin"
test -z "$(git -C "$base" status --porcelain --untracked-files=no)"
export TMPDIR="$scratch" SCRATCH="$scratch" PSI_SCRATCH="$scratch"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
mkdir "$out/validation" "$out/pairs"
export CUEST_FDDS_FOCUSED=1 CUEST_FDDS_FULL_AFTER_FOCUSED=1
unset CUEST_FDDS_READONLY_BUILD_ROOT CUEST_FDDS_BINARY_PIN CUEST_FDDS_EXPECTED_CORE_SHA256
bash "$source_root/devtools/benchmarks/fdds_validate.sh" "$source_root" "$env" "$source_pin" "$out/validation" \
    > "$out/validation-driver.log" 2>&1
test -s "$out/validation/VERIFIED.json"
candidate_hash=$("$env/bin/python" - "$out/validation/import.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["core_sha256"])
PY
)
export PATH="$env/bin:$PATH"
export LD_LIBRARY_PATH="$env/lib:$env/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
runner="$payload/fdds_benchmark.py"
cache=$P/runs/psi4-fdds/20261020-optimization-01/caches
cd "$out"
for system in peptide nanotube protein83; do
    for control in base-gpu-0 candidate-gpu-0 candidate-cpu-0 candidate-gpu-1 base-gpu-1; do
        if [[ "$control" == base-* ]]; then
            build=$base; pin=$base_pin; hash=$base_hash; require_products=()
        else
            build=$source_root; pin=$source_pin; hash=$candidate_hash
            require_products=(--require-gpu-response-products)
        fi
        arm=gpu
        if [[ "$control" == *-cpu-* ]]; then arm=cpu; require_products=(); fi
        export PYTHONPATH="$build/build_fdds_validation/stage/lib"
        export PSIDATADIR="$build/build_fdds_validation/stage/share/psi4"
        printf 'START_BENCHMARK system=%s control=%s source=%s core=%s\n' "$system" "$control" "$pin" "$hash"
        timeout --signal=TERM --kill-after=30s 3600s "$env/bin/python" "$runner" \
            --system "$system" --cache "$cache/$system" --output "$out/pairs/$system-$control" \
            --arm "$arm" --threads 8 --memory "96 GiB" --binary-pin "$pin" \
            --expected-core-sha256 "$hash" --expected-package "$build/build_fdds_validation/stage/lib/psi4" \
            --require-gpu-exchange --require-gpu-metric "${require_products[@]}" \
            > "$out/pairs/$system-$control.log" 2>&1
    done
    "$env/bin/python" "$payload/fdds_metric_compare.py" "$out/pairs" "$system" response
done
test -z "$(git -C "$source_root" status --porcelain --untracked-files=no)"
test -z "$(git -C "$harness_root" status --porcelain --untracked-files=no)"
printf 'RESPONSE_CANDIDATE_VALIDATION_AND_MATCHED_BENCHMARKS_FINISHED\n'
