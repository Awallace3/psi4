#!/bin/bash
# Usage: bash compare_job.sh ABS_REPO HARNESS_COMMIT ABS_BUILD ABS_RUN ABS_DATA BUILD_JOB [H100|H200] [preflight|protein83] [embers|inferno] [native|cupy-canonical]
# Default is a peptide-only direct/direct preflight. Paid jobs need fresh approval.
set -euo pipefail
umask 027
repo=${1:?}; commit=${2:?}; build=${3:?}; run=${4:?}; data=${5:?}; build_job=${6:?}
gpu=${7:-H100}; panel=${8:-preflight}; expected_qos=${9:-embers}
eigensolver=${10:-native}
[[ "$gpu" == H100 || "$gpu" == H200 ]]
[[ "$panel" == preflight || "$panel" == protein83 ]]
[[ "$expected_qos" == embers || "$expected_qos" == inferno ]]
[[ "$eigensolver" == native || "$eigensolver" == cupy-canonical ]]
[[ "$repo" == /* && "$build" == /* && "$run" == /* && "$data" == /* ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
export SLURM_JOB_QOS
SLURM_JOB_QOS=$(squeue -h -j "${SLURM_JOB_ID:?}" -o '%q')
[[ "$SLURM_JOB_QOS" == "$expected_qos" ]]
mkdir "$run"
root="$repo/devtools/benchmarks/gpu_compare"
base=/storage/project/r-cs207-0/awallace43
psi_env="$base/miniconda/envs/p4cuest_sapt"
gpu_env="$base/miniconda/envs/gpu4pyscf_compare"
cuda13="$base/software/toolchains/cuda13/lib64"
xc="$base/software/libxc-7.1.2-cuda/lib64"
package="$build/build/stage/lib/psi4"
source /etc/profile.d/z00_lmod_pace.sh
module load cuda/12.6.1
cuda12="${CUDA_HOME:-${CUDA_PATH:?}}/lib64"
export PYTHONNOUSERSITE=1 PYTHONPATH=""
unset LD_PRELOAD
export LD_LIBRARY_PATH="$gpu_env/lib:$cuda12"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# Use Phoenix persistent scratch, not its disappearing private /tmp bind mount.
export TMPDIR="${CUEST_COMPARE_TMPDIR:-/storage/scratch1/3/awallace43/gpu-dft-direct/${SLURM_JOB_ID:?}}"
[[ "$TMPDIR" == /* ]]
mkdir -p "$TMPDIR/inputs/geometries"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" PYSCF_TMPDIR="$TMPDIR"
for system in peptide protein83; do
    cp "$data/geometries/$system.xyz" "$TMPDIR/inputs/geometries/"
done
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
run_child "$gpu_env/bin/python" - "$repo" "$commit" "$build" "$build_job" "$run" "$panel" "$eigensolver" <<'PY'
import hashlib, json, pathlib, sys
repo, build, run = map(pathlib.Path, (sys.argv[1], sys.argv[3], sys.argv[5]))
sys.path.insert(0, str(repo / "devtools/benchmarks/gpu_compare"))
from protocol import verify_compiled_sources
manifest = json.loads((build / "build-manifest.json").read_text())
assert manifest["job_id"] == sys.argv[4] and manifest["cuest_direct_jk_available"]
assert manifest["package"] == str(build / "build/stage/lib/psi4")
core = next(pathlib.Path(manifest["package"]).glob("core*.so"))
assert hashlib.sha256(core.read_bytes()).hexdigest() == manifest["core_sha256"]
assert hashlib.sha256(pathlib.Path(manifest["cuest_library"]).read_bytes()).hexdigest() == manifest["cuest_library_sha256"]
verify_compiled_sources(repo, manifest["commit"], sys.argv[2])
(run / "compiled-build-manifest.json").write_text(json.dumps(manifest, indent=2))
config = json.loads((repo / "devtools/benchmarks/gpu_compare/config-cc-pvtz.json").read_text())
for key in ("auxbasis", "df_fitting_condition", "df_mode"):
    config.pop(key, None)
config.update(jk_mode="conventional-direct",
              pyscf_eigensolver=sys.argv[7],
              require_energy_agreement=True,
              memory_gib=16 if sys.argv[6] == "preflight" else 64)
(run / "config-direct.json").write_text(json.dumps(config, indent=2))
(run / "harness-build-identity.json").write_text(json.dumps(dict(
    harness_commit=sys.argv[2], compiled_commit=manifest["commit"],
    compiled_inputs_unchanged=True), indent=2))
PY
compiled=$("$gpu_env/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$build/build-manifest.json")
run_child /usr/bin/timeout --signal=TERM --kill-after=30s 600 \
    /bin/bash "$repo/devtools/benchmarks/cuest_direct/run_validation.sh" \
    "$repo" "$commit" "$build" "$run/direct-validation" "$expected_qos" "$compiled" "$gpu" rks
run_child /usr/bin/env PYTHONPATH="$package/..:$repo/tests/pytests" \
    LD_LIBRARY_PATH="$psi_env/lib:$xc:$cuda13" \
    "$psi_env/bin/python" -m pytest --noconftest "$repo/tests/pytests/test_cuest_grid_export.py" \
    -q --basetemp="$TMPDIR/exporter-pytest" --junitxml="$run/exporter-tests.xml"
run_child "$gpu_env/bin/python" - "$run/exporter-tests.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall(".//testcase")
assert len(cases) == 2 and all(not any(c.find(t) is not None for t in
    ("failure", "error", "skipped")) for c in cases)
PY
args=(--config "$run/config-direct.json" --jk-mode conventional-direct
      --stop-on-worker-failure
      --psi-env "$psi_env" --psi-package "$package" --psi-cuda-lib "$cuda13"
      --psi-xc-lib "$xc" --gpu-env "$gpu_env" --cuda12-lib "$cuda12"
      --data "$TMPDIR/inputs" --threads "$OMP_NUM_THREADS")
run_child "$gpu_env/bin/python" "$root/tz_control.py" "${args[@]}" \
    --preflight --systems peptide --repeats 1 --output "$run/peptide-preflight"
if [[ "$panel" == protein83 ]]; then
    run_child "$gpu_env/bin/python" "$root/tz_control.py" "${args[@]}" \
        --systems protein83 --repeats 3 --output "$run/protein83-panel"
fi
(( terminated == 0 ))
run_child "$gpu_env/bin/python" - "$run" "$commit" "$panel" <<'PY'
import json, os, pathlib, sys
run = pathlib.Path(sys.argv[1])
phases = ["peptide-preflight"] + (["protein83-panel"] if sys.argv[3] == "protein83" else [])
for phase in phases:
    assert json.loads((run / phase / "COMPLETE.json").read_text())["complete"]
with (run / "COMPLETED.json").open("x") as stream:
    json.dump(dict(complete=True, jk_mode="conventional-direct",
                   job_id=os.environ["SLURM_JOB_ID"], harness_commit=sys.argv[2],
                   phases=phases), stream)
PY
