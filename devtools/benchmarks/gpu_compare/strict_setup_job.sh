#!/bin/bash
# sbatch options, including absolute logs, are supplied explicitly at submission.
# Args: ABS_REPO COMMIT ABS_NEW_RUN
set -euo pipefail
umask 027
repo=${1:?}; commit=${2:?}; run=${3:?}
base=/storage/project/r-cs207-0/awallace43
gpu_env="$base/miniconda/envs/gpu4pyscf_compare"
psi_env="$base/miniconda/envs/p4cuest_sapt"
sdk="$base/software/cuest-0.2.50-a06e8148"
slurm_bin=${GPU_COMPARE_SLURM_BIN:-/opt/slurm/current/bin}
[[ "$slurm_bin" == /* ]]
[[ "$repo" == /* && "$run" == /* ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
[[ ! -e "$run/SETUP_COMPLETE.json" && ! -e "$run/cuda12-candidate.json" ]]
actual_qos=$("$slurm_bin/squeue" -h -j "${SLURM_JOB_ID:?}" -o '%q')
[[ "$actual_qos" == embers ]] || { echo "Expected embers" >&2; exit 2; }
printf 'QOS: %s\nJOB: %s\nCOMMIT: %s\nHOST: %s\n' \
    "$actual_qos" "$SLURM_JOB_ID" "$commit" "$(hostname)"

terminated=0
child=
on_term() {
    terminated=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
trap on_term TERM
on_exit() {
    status=$?
    if (( terminated )); then
        # Preserve interrupted evidence without leaving a success artifact.
        if [[ -e "$run/SETUP_COMPLETE.json" ]]; then
            mv -- "$run/SETUP_COMPLETE.json" "$run/SETUP_INTERRUPTED.json"
        fi
        status=143
    fi
    exit "$status"
}
trap on_exit EXIT
run_step() {
    (( terminated == 0 )) || exit 143
    "$slurm_bin/srun" --kill-on-bad-exit=1 "$@" <&0 &
    child=$!
    if (( terminated )); then
        kill -TERM "$child" 2>/dev/null || true
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

source /etc/profile.d/z00_lmod_pace.sh
module load cuda/12.6.1
cuda12="${CUDA_HOME:-${CUDA_PATH:?}}/lib64"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=""
unset LD_PRELOAD PYTHONHOME PSIDATADIR PYSCF_CONFIG_FILE PYSCF_EXT_PATH CUPY_CUDA_LIB_PATH
export LD_LIBRARY_PATH="$gpu_env/lib:$cuda12"
export CUDA_HOME="${cuda12%/lib64}" CUDA_PATH="${cuda12%/lib64}"
export PATH="$gpu_env/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="/storage/scratch1/3/awallace43/gpu-dft-strict-setup/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
run_step timeout --signal=TERM --kill-after=30s 1200 \
    "$gpu_env/bin/python" -S -B "$repo/devtools/benchmarks/gpu_compare/capture_cuda12.py" \
    --prefix "$gpu_env" --extra-library-dir "$cuda12" --output "$run/cuda12-candidate.json"

# The build child pins CUDA13 independently; do not leak the CUDA12 module's
# include/compiler environment into that child.
module unload cuda/12.6.1
unset CUDA_HOME CUDA_PATH LD_LIBRARY_PATH
run_step /bin/bash "$repo/devtools/benchmarks/gpu_compare/build_grid_export.sh" \
    "$repo" "$commit" "$run/core" embers "$sdk" "$psi_env"
run_step "$psi_env/bin/python" -S -B - "$run" "$commit" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "core/build-manifest.json").read_text())
assert manifest["commit"] == sys.argv[2]
assert manifest["cuest_direct_jk_available"] is True
candidate = root / "cuda12-candidate.json"
assert json.loads(candidate.read_text())["approved"] is False
result = dict(complete=True, job_id=os.environ["SLURM_JOB_ID"], commit=sys.argv[2],
              build_manifest=str(root / "core/build-manifest.json"),
              cuda12_candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
              cuda12_inventory_approved=False, gpu_tests_run=False)
with (root / "SETUP_COMPLETE.json").open("x") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
print("SETUP_ARTIFACTS_READY_FOR_REVIEW")
PY
