#!/usr/bin/env bash
# Separate perturbative diagnostics, never timing evidence. No builds or installs.
# args: harness-root harness-pin scientific-root scientific-pin validation-output output
set -euo pipefail
root=$(realpath "$1")
harness_pin=$2
science=$(realpath "$3")
science_pin=$4
validation=$(realpath "$5")
out=$(realpath "$6")
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
test -n "${SLURM_JOB_ID:-}"
for pair in "$root:$harness_pin" "$science:$science_pin"; do
    test "$(git -C "${pair%:*}" rev-parse HEAD)" = "${pair##*:}"
    test -z "$(git -C "${pair%:*}" status --porcelain --untracked-files=no)"
done
test -f "$validation/metadata/COMPLETED"
test -f "$validation/host/VERIFIED"
test -f "$validation/cuda/VERIFIED"
mkdir -p "$out"
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    export TMPDIR="/scratch/$USER/grid-memory-$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
fi
TMPDIR=$(mktemp -d "$TMPDIR/grid-memory-$SLURM_JOB_ID.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "$out/gpu.txt"
grep -q H200 "$out/gpu.txt"
scontrol show job -o "$SLURM_JOB_ID" > "$out/allocation.txt"
grep -q 'Account=gts-cs207 ' "$out/allocation.txt"
grep -q 'QOS=inferno ' "$out/allocation.txt"
printf 'scientific_commit=%s\nharness_commit=%s\nvalidation=%s\ndiagnostic_only=true\n' \
    "$science_pin" "$harness_pin" "$validation" > "$out/provenance.txt"
overall=0
for variant in host cuda; do
    (
        libxc=
        [[ "$variant" == cuda ]] && libxc=/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda/lib64:
        export LD_LIBRARY_PATH="$libxc$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        build="$science/build_grid_$variant"
        eval "$("$build/stage/bin/psi4" --psiapi)"
        export EXPECTED_PSI4_PACKAGE="$build/stage/lib/psi4"
        export VALIDATION_RECEIPT="$validation/$variant/import.json"
        "$env/bin/python" - <<'PY' > "$out/binary-$variant.json"
import hashlib, json, os
from pathlib import Path
import psi4
p = Path(psi4.__file__).resolve().parent
assert p == Path(os.environ["EXPECTED_PSI4_PACKAGE"]).resolve()
r = json.loads(Path(os.environ["VALIDATION_RECEIPT"]).read_text())
assert hashlib.sha256(Path(psi4.core.__file__).read_bytes()).hexdigest() == r["core_sha256"]
print(json.dumps(r, indent=2))
PY
        # Water is an actual GPU-accounting/hook smoke test before larger cases.
        for system in water nanotube protein83; do
            for arm in fixed99 coarse75 fixed75 quarter99 third99; do
                case_root="$out/$variant/$system-$arm"
                mkdir -p "$case_root"
                case_scratch=$(mktemp -d "$TMPDIR/case.XXXXXX")
                echo "START memory $variant $system $arm"
                set +e
                TMPDIR="$case_scratch" SCRATCH="$case_scratch" PSI_SCRATCH="$case_scratch" \
                    "$env/bin/python" "$root/devtools/benchmarks/grid_memory.py" run \
                    --output "$case_root/memory" --interval 0.2 --timeout 900 \
                    --script "$root/devtools/benchmarks/grid_evidence.py" -- \
                    --worker --output "$case_root/case" --system "$system" --arm "$arm" \
                    --repeat 1 --threads "$OMP_NUM_THREADS" --memory "112 GiB" \
                    --variant "$variant" --scientific-commit "$science_pin" \
                    --expected-package "$EXPECTED_PSI4_PACKAGE" \
                    > "$case_root/supervisor.log" 2>&1
                rc=$?
                set -e
                printf '%s\n' "$rc" > "$case_root/returncode.txt"
                echo "END memory $variant $system $arm rc=$rc"
                # Do not spend on larger systems if the diagnostic smoke fails.
                if (( rc != 0 )); then exit "$rc"; fi
                "$env/bin/python" - "$case_root" "$arm" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root/"memory/boundaries.jsonl").read_text().splitlines()]
assert any(r["phase"] == "jk_initialize" for r in rows), "JK boundary hook missing"
assert any(r.get("grid_points", 0) > 0 for r in rows), "Actual GPU grid count unavailable"
if sys.argv[2] != "fixed99" and sys.argv[2] != "fixed75":
    assert any(r["phase"] == "xc_set_grid" and r["event"] == "leave"
               and r.get("initial_grid") is False for r in rows), "Final switch hook missing"
case = json.loads((root/"case/result.json").read_text())
case.update(diagnostic_only=True, eligible_for_speedup_bars=False)
(root/"case/result.json").write_text(json.dumps(case, indent=2) + "\n")
PY
            done
        done
    ) > "$out/$variant.log" 2>&1 &
    # Keep errexit active inside the child; "( ... ) || ..." disables it.
    payload=$!
    set +e
    wait "$payload"; variant_rc=$?
    set -e
    (( variant_rc == 0 )) || overall=1
done
"$env/bin/python" - "$out" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
cases = []
for p in sorted(root.glob("*/*/memory/summary.json")):
    data = json.loads(p.read_text())
    data.update(case=p.parent.parent.name, backend=p.parent.parent.parent.name)
    cases.append(data)
report = dict(diagnostic_only=True, eligible_for_speedup_bars=False,
              expected_cases=30, cases=cases,
              complete=len(cases) == 30 and all(r["ok"] for r in cases),
              scope="Fixed geometry/basis within each system; cross-system NBF is not an independent basis scan.")
(root/"memory-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
if not report["complete"]:
    raise SystemExit(1)
PY
test "$(git -C "$science" rev-parse HEAD)" = "$science_pin"
test "$(git -C "$root" rev-parse HEAD)" = "$harness_pin"
if (( overall == 0 )); then
    printf 'Memory diagnostics complete; sampled peaks are not timing evidence.\n' > "$out/VERIFIED"
fi
exit "$overall"
