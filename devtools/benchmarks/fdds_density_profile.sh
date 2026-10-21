#!/usr/bin/env bash
# Diagnostic only: runtime gates plus one physical replay of a timer-only build.
set -euo pipefail
root=$(realpath "${1:?fresh timer-only source}")
pin=${2:?source/harness pin}
out=$(realpath "${3:?fresh output}")
P=/storage/project/r-cs207-0/awallace43
env=$P/miniconda/envs/p4cuest_sapt
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
test "$SLURM_JOB_ID" = 14214205
test "$SLURM_CPUS_PER_TASK" = 8
scratch=/var/tmp/awallace43-fdds-14214205-v2
test -s "$scratch/STORAGE_VERIFIED.json"
export TMPDIR="$scratch" SCRATCH="$scratch" PSI_SCRATCH="$scratch"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
compile="$P/runs/psi4-fdds/20261021-density-profile-compile-01"
test -s "$compile/BUILD_ONLY.txt"
hash=$("$env/bin/python" - "$compile/import.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["core_sha256"])
PY
)
export CUEST_FDDS_READONLY_BUILD_ROOT="$root" CUEST_FDDS_BINARY_PIN="$pin"
export CUEST_FDDS_EXPECTED_CORE_SHA256="$hash" CUEST_FDDS_COMPILE_ONLY=0
export CUEST_FDDS_FOCUSED=1 CUEST_FDDS_FULL_AFTER_FOCUSED=1
mkdir "$out/validation"
bash "$root/devtools/benchmarks/fdds_validate.sh" "$root" "$env" "$pin" "$out/validation" \
    > "$out/validation-driver.log" 2>&1
test -s "$out/validation/VERIFIED.json"
export PATH="$env/bin:$PATH"
export LD_LIBRARY_PATH="$env/lib:$env/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$root/build_fdds_validation/stage/lib"
export PSIDATADIR="$root/build_fdds_validation/stage/share/psi4"
driver_sha=$(git -C "$root" show "$pin:psi4/driver/procrouting/sapt/sapt_mp2_terms.py" | sha256sum)
driver_sha=${driver_sha%% *}
cd "$out"
timeout --signal=TERM --kill-after=30s 840s "$env/bin/python" \
    "$root/devtools/benchmarks/fdds_alda_profile.py" \
    "$root/devtools/benchmarks/fdds_benchmark.py" "$driver_sha" "$out/profile.json" \
    --system protein83 --cache "$P/runs/psi4-fdds/20261020-optimization-01/caches/protein83" \
    --output "$out/replay" --arm gpu --threads 8 --memory "96 GiB" \
    --binary-pin "$pin" --expected-core-sha256 "$hash" \
    --expected-package "$root/build_fdds_validation/stage/lib/psi4" \
    --require-gpu-exchange --require-gpu-metric --require-gpu-response-products \
    > "$out/replay.log" 2>&1
"$env/bin/python" - "$out" "$P/runs/psi4-fdds/20261021-alda-profile-01/replay/result.json" <<'PY'
from pathlib import Path
import json,math,sys
out=Path(sys.argv[1]);a=json.loads((out/"replay/result.json").read_text())
b=json.loads(Path(sys.argv[2]).read_text());p=json.loads((out/"profile.json").read_text())
assert a["ok"] and b["ok"] and p["valid"] and p["sites"]==13 and len(p["records"])==19
for k in ("cache_sha256","metadata_sha256","runner_sha256","options","threads",
          "memory","hostname","job_id","numpy_version","numpy_configuration"):
    assert a[k]==b[k],k
deps=lambda d:{p:h for p,h in d["libraries_sha256"].items()
               if not (Path(p).parent==Path(d["psi4_package"]).parent and Path(p).name.startswith("core.cpython"))}
assert deps(a)==deps(b)
assert len(a["pinv_calls"])==22
assert (a["gpu_timer_calls"],a["gpu_exchange_timer_calls"],a["gpu_metric_timer_calls"],a["gpu_response_timer_calls"])==(20,4,12,60)
phases={}
for label in ("AO integrals and collapse","inverse metric","auxiliary overlap and collapse"):
    found=[v for k,v in a["timers"].items() if v["timer_name"]=="FDDS density: "+label]
    assert len(found)==1 and found[0]["n_calls"]==1 and math.isfinite(found[0]["wall_time"])
    assert found[0]["wall_time"]>=0
    phases[label]=found[0]["wall_time"]
errors={k:abs(a["energies_Eh"][k]-v) for k,v in b["energies_Eh"].items()}
assert max(errors.values())<=1e-5,errors
with (out/"AUDIT_PENDING.json").open("x") as f:
    json.dump({"diagnostic_only":True,"no_speedup_claim":True,"projection_phases_seconds":phases,
               "errors_Eh":errors,"note":"Three nested projection regions; includes integrals/packing/contractions, not pure BLAS"},f,indent=2)
PY
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -z "$(git -C "$root" status --porcelain --untracked-files=no)"
"$env/bin/python" - "$out" <<'PY'
import os,pathlib,sys
p=pathlib.Path(sys.argv[1]);assert not (p/"VERIFIED.json").exists()
os.rename(p/"AUDIT_PENDING.json",p/"VERIFIED.json")
PY
