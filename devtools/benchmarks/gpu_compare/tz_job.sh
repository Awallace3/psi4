#!/bin/bash
# One H200: exporter tests, peptide PBE/WB97X, Protein83; native + stored-DF.
# Headline is full worker lifetime, including imports and DF preconstruction.
# Usage: bash tz_job.sh ABS_REPO COMMIT ABS_BUILD ABS_RUN ABS_DATA QOS BUILD_JOB_ID
# Submit with explicit embers, 8 CPUs, 256G, gpu:h200:1, TERM@120.
set -euo pipefail
repo=${1:?harness checkout}; commit=${2:?harness commit}
build=${3:?exporter build run}; run=${4:?unique run}; data=${5:?staged input tree}
expected_qos=${6:-embers}; build_job=${7:?expected build job ID}
[[ "$expected_qos" == embers || "$expected_qos" == inferno ]]
[[ "$repo" == /* && "$build" == /* && "$run" == /* && "$data" == /* ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
export SLURM_JOB_QOS
SLURM_JOB_QOS=$(squeue -h -j "${SLURM_JOB_ID:?}" -o '%q')
[[ "$SLURM_JOB_QOS" == "$expected_qos" ]]
[[ -f "$build/build-manifest.json" ]]
mkdir "$run"  # Exclusive creation: retries must never alter partial evidence.
root="$repo/devtools/benchmarks/gpu_compare"
psi_env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
gpu_env=/storage/project/r-cs207-0/awallace43/miniconda/envs/gpu4pyscf_compare
cuda13=/storage/project/r-cs207-0/awallace43/software/toolchains/cuda13/lib64
xc=/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda/lib64
psi_package="$build/build/stage/lib/psi4"
source /etc/profile.d/z00_lmod_pace.sh
module load cuda/12.6.1
cuda12="${CUDA_HOME:-${CUDA_PATH:?}}/lib64"
[[ -d "$cuda12" && -d "$psi_package" ]]
export PYTHONNOUSERSITE=1 PYTHONPATH=""
export LD_LIBRARY_PATH="$gpu_env/lib:$cuda12"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# Avoid Phoenix's disappearing private /tmp bind mount. Never delete others' scratch.
export TMPDIR="/storage/scratch1/3/awallace43/gpu-dft-tz-control/${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR/inputs/geometries"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" PYSCF_TMPDIR="$TMPDIR"
for system in peptide protein83; do
    cp "$data/geometries/$system.xyz" "$TMPDIR/inputs/geometries/"
done
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
child=; terminated=0
on_term() {
    terminated=1
    [[ -z "$child" ]] || { kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true; }
}
trap on_term TERM
run_child() {
    (( terminated == 0 )) || exit 143
    setsid "$@" <&0 &
    child=$!
    if (( terminated )); then
        kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
    fi
    set +e
    wait "$child"
    status=$?
    while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
    set -e
    child=
    (( terminated == 0 )) || exit 143
    (( status == 0 )) || exit "$status"
}
# The compiled source stays pinned; newer harness-only edits must not silently
# change the C++ exporter being measured.
run_child "$gpu_env/bin/python" - "$build" "$repo" "$psi_package" "$commit" "$expected_qos" "$build_job" "$run" <<'PY'
import hashlib, json, pathlib, subprocess, sys
build, repo, package = map(pathlib.Path, sys.argv[1:4])
manifest = json.loads((build / "build-manifest.json").read_text())
assert manifest["qos"] == sys.argv[5]
assert manifest["commit"] == sys.argv[4] and manifest["job_id"] == sys.argv[6]
assert pathlib.Path(manifest["package"]).resolve() == package.resolve()
core = next(package.glob("core*.so"))
assert hashlib.sha256(core.read_bytes()).hexdigest() == manifest["core_sha256"]
subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", manifest["commit"], "HEAD",
                "--", "psi4/src/psi4/libfock/v.cc", "psi4/src/psi4/libfock/v.h",
                "psi4/src/export_functional.cc", "tests/pytests/test_cuest_grid_export.py"], check=True)
print("Compiled exporter verified", manifest["commit"], flush=True)
with (pathlib.Path(sys.argv[7]) / "compiled-build-manifest.json").open("x") as stream:
    json.dump(manifest, stream, indent=2)
PY
psi_exe="${psi_package%/lib/psi4}/bin/psi4"
[[ -x "$psi_exe" ]]
eval "$(PATH="$psi_env/bin:$PATH" LD_LIBRARY_PATH="$psi_env/lib:$xc:$cuda13" "$psi_exe" --psiapi)"
run_child /usr/bin/env PYTHONPATH="$psi_package/..:$repo/tests/pytests" \
    LD_LIBRARY_PATH="$psi_env/lib:$xc:$cuda13" \
    "$psi_env/bin/python" -m pytest "$repo/tests/pytests/test_cuest_grid_export.py" \
    -q --basetemp="$TMPDIR/exporter-pytest" --junitxml="$run/exporter-tests.xml"
run_child "$gpu_env/bin/python" - "$run/exporter-tests.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall(".//testcase")
assert len(cases) == 2 and all(not any(case.find(tag) is not None for tag in
    ("failure", "error", "skipped")) for case in cases), "Both allocated-GPU tests must pass, not skip"
print("Exporter GPU validation: 2 passed", flush=True)
PY
args=(--psi-env "$psi_env" --psi-package "$psi_package"
    --psi-cuda-lib "$cuda13" --psi-xc-lib "$xc" --gpu-env "$gpu_env" --cuda12-lib "$cuda12"
    --data "$TMPDIR/inputs" --threads "$OMP_NUM_THREADS")
run_child "$gpu_env/bin/python" "$root/tz_control.py" "${args[@]}" \
    --preflight --systems peptide --repeats 1 --output "$run/peptide-preflight"
# Exit 0 requires matching meshes, overlap audits, memory observations and SCFs
# in all four arms for PBE and WB97X, including the lazy RSH stored-tensor route.
run_child "$gpu_env/bin/python" "$root/tz_control.py" "${args[@]}" \
    --systems protein83 --repeats 3 --output "$run/protein83-panel"
(( terminated == 0 ))
run_child "$gpu_env/bin/python" - "$run" "$commit" <<'PY'
import json, os, pathlib, sys
run = pathlib.Path(sys.argv[1])
for phase in ("peptide-preflight", "protein83-panel"):
    assert json.loads((run / phase / "COMPLETE.json").read_text())["complete"]
with (run / "COMPLETED.json").open("x") as stream:
    json.dump(dict(complete=True, job_id=os.environ["SLURM_JOB_ID"],
                   harness_commit=sys.argv[2], phases=["peptide-preflight", "protein83-panel"]), stream)
PY
