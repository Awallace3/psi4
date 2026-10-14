#!/usr/bin/env bash
# Runtime-only evidence campaign. Never changes/rebuilds the scientific source.
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
mkdir -p "$out"
"$env/bin/python" "$root/devtools/benchmarks/grid_workload_gate.py" \
    --validation "$validation" --science "$science" --pin "$science_pin" \
    --output "$out/supported-workload.json"
export PATH="$env/bin:$PATH" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    export TMPDIR="/scratch/$USER/grid-evidence-$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
fi
TMPDIR=$(mktemp -d "$TMPDIR/grid-evidence-$SLURM_JOB_ID.XXXXXX")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "$out/gpu.txt"
grep -q H200 "$out/gpu.txt"
lscpu > "$out/cpu.txt"
scontrol show job -o "$SLURM_JOB_ID" > "$out/allocation.txt"
grep -q 'Account=gts-cs207 ' "$out/allocation.txt"
grep -q 'QOS=inferno ' "$out/allocation.txt"
printf 'scientific_commit=%s\nharness_commit=%s\nvalidation=%s\n' \
    "$science_pin" "$harness_pin" "$validation" > "$out/provenance.txt"
for variant in host cuda; do
    (
        libxc=
        [[ "$variant" == cuda ]] && libxc=/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda/lib64:
        export LD_LIBRARY_PATH="$libxc$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        build="$science/build_grid_$variant"
        # Python stdin adds cwd ahead of --psiapi PYTHONPATH. Never import
        # from the source checkout: it has no staged native core.
        cd "$out"
        eval "$("$build/stage/bin/psi4" --psiapi)"
        export EXPECTED_PSI4_PACKAGE="$build/stage/lib/psi4"
        export VALIDATION_RECEIPT="$validation/$variant/import.json"
        # Keep import, hashing and provenance outside every energy() boundary.
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
        selected_arms=()
        if [[ -n "${GRID_EVIDENCE_ARMS:-}" ]]; then
            read -r -a selected_arms <<< "$GRID_EVIDENCE_ARMS"
            selected_arms=(--arms "${selected_arms[@]}")
        fi
        "$env/bin/python" "$root/devtools/benchmarks/grid_evidence.py" \
            --output "$out/$variant" --variant "$variant" --threads "$OMP_NUM_THREADS" \
            --memory "112 GiB" --repeats 3 --scientific-commit "$science_pin" \
            --expected-package "$EXPECTED_PSI4_PACKAGE" "${selected_arms[@]}" > "$out/$variant.log" 2>&1
    )
done
test "$(git -C "$science" rev-parse HEAD)" = "$science_pin"
test "$(git -C "$root" rev-parse HEAD)" = "$harness_pin"
printf 'Both backend campaigns completed; inspect accuracy_pass separately.\n' > "$out/VERIFIED"
