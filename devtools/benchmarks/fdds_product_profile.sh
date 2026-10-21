#!/usr/bin/env bash
# Diagnostic only, via the existing exclusive shared allocation helper.
set -euo pipefail
root=$(realpath "${1:?frozen diagnostic harness}")
pin=${2:?diagnostic harness pin}
out=$(realpath "${3:?fresh diagnostic output}")
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
test "$SLURM_JOB_ID" = 14214205
test "$SLURM_CPUS_PER_TASK" = 8
P=/storage/project/r-cs207-0/awallace43
source=$P/gits/psi4.fdds-metric-fold-ada1a1d212-v1
source_pin=ada1a1d2128d6d6f3f5b89952b08f6ffd4ca5917
core_hash=6f8294594134277ac55ae1a3b87d9003367a451b4f8a506a16e1a2b8a823de68
test "$(git -C "$source" rev-parse HEAD)" = "$source_pin"
test -z "$(git -C "$source" status --porcelain --untracked-files=no)"
env=$P/miniconda/envs/p4cuest_sapt
scratch=/var/tmp/awallace43-fdds-14214205-v2
test -s "$scratch/STORAGE_VERIFIED.json"
export TMPDIR="$scratch" SCRATCH="$scratch" PSI_SCRATCH="$scratch"
export PATH="$env/bin:$PATH"
export LD_LIBRARY_PATH="$env/lib:$env/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$source/build_fdds_validation/stage/lib"
export PSIDATADIR="$source/build_fdds_validation/stage/share/psi4"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
driver_sha=$(git -C "$source" show "$source_pin:psi4/driver/procrouting/sapt/sapt_mp2_terms.py" | sha256sum)
driver_sha=${driver_sha%% *}
cd "$out"
timeout --signal=TERM --kill-after=30s 840s "$env/bin/python" \
    "$root/devtools/benchmarks/fdds_product_profile.py" \
    "$root/devtools/benchmarks/fdds_benchmark.py" "$driver_sha" "$out/profile.json" \
    --system protein83 --cache "$P/runs/psi4-fdds/20261020-optimization-01/caches/protein83" \
    --output "$out/replay" --arm gpu --threads 8 --memory "96 GiB" \
    --binary-pin "$source_pin" --expected-core-sha256 "$core_hash" \
    --expected-package "$source/build_fdds_validation/stage/lib/psi4" \
    --require-gpu-exchange --require-gpu-metric > "$out/replay.log" 2>&1
"$env/bin/python" - "$out" "$P/runs/psi4-fdds/20261020-metric-fold-01/pairs/protein83-candidate-gpu-0/result.json" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
profile = json.loads((out / "profile.json").read_text())
assert profile["valid"] and profile["sites"] == 8 and len(profile["records"]) == 80
actual = json.loads((out / "replay/result.json").read_text())
reference = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert actual["ok"] and reference["ok"]
for key in ("cache_sha256", "metadata_sha256", "options", "hostname", "job_id",
            "core_sha256", "binary_pin", "threads", "memory", "libraries_sha256"):
    assert actual[key] == reference[key], key
errors = {k: abs(actual["energies_Eh"][k] - v) for k, v in reference["energies_Eh"].items()}
assert max(errors.values()) <= 1.e-5, errors
groups = {}
for r in profile["records"]:
    group = groups.setdefault(r["expression"], {"count": 0, "seconds": 0, "shapes": []})
    group["count"] += 1
    group["seconds"] += r["seconds"]
    if r["shape"] not in group["shapes"]:
        group["shapes"].append(r["shape"])
summary = {"diagnostic_only": True, "groups": groups, "errors_Eh": errors,
           "total_product_seconds": profile["total_product_seconds"],
           "bookkeeping_seconds_lower_bound": profile["bookkeeping_seconds_lower_bound"],
           "no_optimized_speedup_claim": True}
with (out / "VERIFIED.json").open("x") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary))
PY
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$source" status --porcelain --untracked-files=no)"
