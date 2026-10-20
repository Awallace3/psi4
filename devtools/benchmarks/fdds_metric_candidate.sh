#!/usr/bin/env bash
# Run only through the owner's whole-allocation exclusive shared helper.
# Fresh source/build/output; accepted fd2 stage/cache/results stay read-only.
set -euo pipefail
source_root=$(realpath "${1:?candidate scientific source}")
source_pin=${2:?candidate scientific source pin}
out=$(realpath "${3:?new output root}")
harness_pin=${4:?frozen controller and benchmark pin}
payload=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
harness_root=$(git -C "$payload" rev-parse --show-toplevel)
test "$(git -C "$harness_root" rev-parse HEAD)" = "$harness_pin"
test -z "$(git -C "$harness_root" status --porcelain --untracked-files=no)"
test "$SLURM_JOB_ID" = 14214205
test "$SLURM_CPUS_PER_TASK" = 8
test "$(git -C "$source_root" rev-parse HEAD)" = "$source_pin"
test -z "$(git -C "$source_root" status --porcelain --untracked-files=no)"
P=/storage/project/r-cs207-0/awallace43
env=$P/miniconda/envs/p4cuest_sapt
base=$P/gits/psi4.fdds-opt-cuda-compile-20261020-v2
base_pin=fd2c09cd0cfbfce61a311508ba6c53039025179a
base_hash=92e5bff6641dbb2f4790938e9886fc8038d6d03527eb26c6934ddeecabac5a7f
scratch=/var/tmp/awallace43-fdds-14214205-v2
test -s "$scratch/STORAGE_VERIFIED.json"
test -s "$P/runs/psi4-fdds/20261020-interactive-01/matched-14214205-var-local-v2/protein83-VERIFIED.json"
test "$(git -C "$base" rev-parse HEAD)" = "$base_pin"
test -z "$(git -C "$base" status --porcelain --untracked-files=no)"
export TMPDIR="$scratch" SCRATCH="$scratch" PSI_SCRATCH="$scratch"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
mkdir "$out/validation" "$out/pairs"
# Exact candidate's existing compiler isolation/provenance/gdb+zero-skip gate.
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
export PSIDATADIR="$source_root/build_fdds_validation/stage/share/psi4"
runner="$payload/fdds_benchmark.py"
cache=$P/runs/psi4-fdds/20261020-optimization-01/caches
cd "$out"
for system in peptide nanotube protein83; do
    # Freeze all candidate/base inputs before first timing. Fresh processes;
    # baseline GPU0/candidate GPU0/CPU0/candidate GPU1/baseline GPU1.
    for control in base-gpu-0 candidate-gpu-0 candidate-cpu-0 candidate-gpu-1 base-gpu-1; do
        if [[ "$control" == base-* ]]; then
            build=$base
            pin=$base_pin
            hash=$base_hash
            require_metric=()
        else
            build=$source_root
            pin=$source_pin
            hash=$candidate_hash
            require_metric=(--require-gpu-metric)
        fi
        arm=gpu
        if [[ "$control" == *-cpu-* ]]; then arm=cpu; require_metric=(); fi
        export PYTHONPATH="$build/build_fdds_validation/stage/lib"
        export PSIDATADIR="$build/build_fdds_validation/stage/share/psi4"
        printf 'START_BENCHMARK system=%s control=%s source=%s core=%s\n' "$system" "$control" "$pin" "$hash"
        timeout --signal=TERM --kill-after=30s 3600s \
            "$env/bin/python" "$runner" --system "$system" --cache "$cache/$system" \
            --output "$out/pairs/$system-$control" --arm "$arm" \
            --threads 8 --memory "96 GiB" --binary-pin "$pin" \
            --expected-core-sha256 "$hash" --expected-package "$build/build_fdds_validation/stage/lib/psi4" \
            --require-gpu-exchange "${require_metric[@]}" \
            > "$out/pairs/$system-$control.log" 2>&1
    done
    "$env/bin/python" "$payload/fdds_metric_compare.py" "$out/pairs" "$system"
done
test -z "$(git -C "$source_root" status --porcelain --untracked-files=no)"
test "$(git -C "$harness_root" rev-parse HEAD)" = "$harness_pin"
test -z "$(git -C "$harness_root" status --porcelain --untracked-files=no)"
printf 'CANDIDATE_VALIDATION_AND_MATCHED_BENCHMARKS_FINISHED\n'
