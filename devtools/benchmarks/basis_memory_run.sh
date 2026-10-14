#!/usr/bin/env bash
# Runtime-only JK isolation, 6 same-geometry basis controls x 2 fresh processes.
set -euo pipefail
root=$(realpath "$1"); harness_pin=$2
science=$(realpath "$3"); science_pin=$4
validation=$(realpath "$5"); out=$(realpath "$6")
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
for pair in "$root:$harness_pin" "$science:$science_pin"; do
    test "$(git -C "${pair%:*}" rev-parse HEAD)" = "${pair##*:}"
    test -z "$(git -C "${pair%:*}" status --porcelain --untracked-files=no)"
done
mkdir -p "$out"
cd "$out"
"$env/bin/python" "$root/devtools/benchmarks/grid_workload_gate.py" \
    --validation "$validation" --science "$science" --pin "$science_pin" \
    --output "$out/supported-workload.json"
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
test -n "${SLURM_JOB_ID:-}"
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    export TMPDIR="/scratch/$USER/basis-memory-$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
fi
TMPDIR=$(mktemp -d "$TMPDIR/basis-memory.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits > "$out/gpu.txt"
grep -q H200 "$out/gpu.txt"
"$env/bin/python" - "$out/gpu.txt" <<'PY'
import sys
from pathlib import Path
rows=Path(sys.argv[1]).read_text().splitlines()
assert len(rows)==1 and float(rows[0].rsplit(",",1)[1])>=120000, "Require one large-memory H200"
PY
build="$science/build_grid_host"
eval "$("$build/stage/bin/psi4" --psiapi)"
export EXPECTED_PSI4_PACKAGE="$build/stage/lib/psi4"
export VALIDATION_RECEIPT="$validation/host/import.json"
"$env/bin/python" - <<'PY' > "$out/binary-host.json"
import hashlib,json,os
from pathlib import Path
import psi4
assert Path(psi4.__file__).resolve().parent==Path(os.environ["EXPECTED_PSI4_PACKAGE"]).resolve()
r=json.loads(Path(os.environ["VALIDATION_RECEIPT"]).read_text())
assert hashlib.sha256(Path(psi4.core.__file__).read_bytes()).hexdigest()==r["core_sha256"]
print(json.dumps(r,indent=2))
PY
for repeat in 1 2; do
    cases=(control922 primary634 primary766 primary1293 primary1786 aux4762)
    [[ "$repeat" == 2 ]] && cases=(aux4762 primary1786 primary1293 primary766 primary634 control922)
    for name in "${cases[@]}"; do
        directory="$out/$name-$repeat"
        mkdir -p "$directory"
        echo "START $name repeat=$repeat"
        "$env/bin/python" "$root/devtools/benchmarks/grid_memory.py" run \
            --output "$directory/memory" --interval 0.2 --timeout 900 \
            --script "$root/devtools/benchmarks/basis_memory.py" -- \
            --output "$directory/case" --case "$name" --threads "$OMP_NUM_THREADS" \
            --scientific-commit "$science_pin" --expected-package "$EXPECTED_PSI4_PACKAGE" \
            > "$directory/supervisor.log" 2>&1
        echo "END $name repeat=$repeat"
    done
done
"$env/bin/python" - "$out" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);cases=[]
for p in sorted(r.glob("*/case/result.json")):
    d=json.loads(p.read_text());m=json.loads((p.parent.parent/"memory/summary.json").read_text())
    assert d["ok"] and not d["cpu_smoke"] and not d["xc_grid_constructed"]
    assert m["ok"] and m["gpu_observed"] and m["failed_samples"]==m["unavailable_samples"]==0
    cases.append(dict(name=p.parent.parent.name,worker=d,memory=m))
assert len(cases)==12
report=dict(complete=True,cases=cases,diagnostic_only=True,eligible_for_speedup_bars=False,
            scope="Same protein83 geometry; vary primary at fixed Naux or auxiliary at fixed NBF. "
                  "Synthetic occupied coefficients, no XC grid or SCF. Queried sizes rounded to "
                  "0.001 GiB exclude construction temporaries; NVML peaks remain lower bounds.")
(r/"basis-memory-evidence.json").write_text(json.dumps(report,indent=2)+"\n")
PY
printf 'Twelve JK-only basis-memory diagnostics complete.\n' > "$out/VERIFIED"
