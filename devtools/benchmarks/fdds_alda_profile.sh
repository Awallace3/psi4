#!/usr/bin/env bash
# Attribution only: unchanged accepted scientific build, no candidate timing.
set -euo pipefail
root=$(realpath "${1:?frozen diagnostic harness}")
pin=${2:?diagnostic pin}
out=$(realpath "${3:?new diagnostic output}")
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
test "$SLURM_JOB_ID" = 14214205
test "$SLURM_CPUS_PER_TASK" = 8
P=/storage/project/r-cs207-0/awallace43
source=$P/gits/psi4.fdds-response-products-dcc8ac9713-v1
source_pin=dcc8ac971310530688920efd0b2ba587a6e5cfd7
core_hash=c4579a401c5c7fb81d009694c569124636f9d7205668e3a6da20597bb649293f
test "$(git -C "$source" rev-parse HEAD)" = "$source_pin"
test -z "$(git -C "$source" status --porcelain --untracked-files=no)"
env=$P/miniconda/envs/p4cuest_sapt
scratch=/var/tmp/awallace43-fdds-14214205-v2
test -s "$scratch/STORAGE_VERIFIED.json"
test -s "$P/runs/psi4-fdds/20261020-response-products-01/validation/VERIFIED.json"
test -s "$P/runs/psi4-fdds/20261020-response-products-01/pairs/protein83-RESPONSE_CANDIDATE_VERIFIED.json"
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
    "$root/devtools/benchmarks/fdds_alda_profile.py" \
    "$root/devtools/benchmarks/fdds_benchmark.py" "$driver_sha" "$out/profile.json" \
    --system protein83 --cache "$P/runs/psi4-fdds/20261020-optimization-01/caches/protein83" \
    --output "$out/replay" --arm gpu --threads 8 --memory "96 GiB" \
    --binary-pin "$source_pin" --expected-core-sha256 "$core_hash" \
    --expected-package "$source/build_fdds_validation/stage/lib/psi4" \
    --require-gpu-exchange --require-gpu-metric --require-gpu-response-products \
    > "$out/replay.log" 2>&1
"$env/bin/python" - "$out" "$P/runs/psi4-fdds/20261020-response-products-01/pairs/protein83-candidate-gpu-0/result.json" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
p = json.loads((out / "profile.json").read_text())
assert p["valid"] and p["sites"] == 13 and len(p["records"]) == 19
actual = json.loads((out / "replay/result.json").read_text())
reference = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert actual["ok"] and reference["ok"]
for k in ("cache_sha256", "metadata_sha256", "runner_sha256", "options", "hostname",
          "job_id", "core_sha256", "binary_pin", "threads", "memory",
          "libraries_sha256", "numpy_version", "numpy_configuration"):
    assert actual[k] == reference[k], k
assert (actual["gpu_timer_calls"], actual["gpu_exchange_timer_calls"],
        actual["gpu_metric_timer_calls"], actual["gpu_response_timer_calls"]) == (20, 4, 12, 60)
assert len(actual["pinv_calls"]) == 22
errors = {k: abs(actual["energies_Eh"][k] - v) for k, v in reference["energies_Eh"].items()}
assert max(errors.values()) <= 1.e-5, errors
groups = {}
for r in p["records"]:
    g = groups.setdefault(r["label"], {"count": 0, "seconds": 0, "argument_shapes": []})
    g["count"] += 1
    g["seconds"] += r["seconds"]
    if r["argument_shapes"] not in g["argument_shapes"]:
        g["argument_shapes"].append(r["argument_shapes"])
summary = {"diagnostic_only": True, "groups": groups, "errors_Eh": errors,
           "no_optimized_speedup_claim": True,
           "inclusive_groups": ["alda_A_inclusive", "alda_B_inclusive"],
           "bookkeeping_seconds_lower_bound": p["bookkeeping_seconds_lower_bound"]}
with (out / "AUDIT_PENDING.json").open("x") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary))
PY
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
test "$(git -C "$source" rev-parse HEAD)" = "$source_pin"
test -z "$(git -C "$source" status --porcelain --untracked-files=no)"
# A late provenance failure must never leave a success marker.
"$env/bin/python" - "$out" <<'PY'
import os, pathlib, sys
out = pathlib.Path(sys.argv[1])
assert not (out / "VERIFIED.json").exists()
os.rename(out / "AUDIT_PENDING.json", out / "VERIFIED.json")
PY
