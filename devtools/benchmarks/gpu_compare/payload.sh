#!/usr/bin/env bash
set -euo pipefail
mode=$1
shift
source /etc/profile.d/z00_lmod_pace.sh
module load cuda/12.6.1
command -v nvcc
export PYTHONNOUSERSITE=1
# --psiapi emits PYTHONPATH=...:$PYTHONPATH; an empty exported value is
# intentional route isolation and remains safe under this script's set -u.
export PYTHONPATH=""
export LD_LIBRARY_PATH="$COMPARE_GPU_ENV/lib:${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}/lib64"
cuda12_lib="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}/lib64"
[[ -d "$cuda12_lib" ]]
root="$COMPARE_REPO/devtools/benchmarks/gpu_compare"
if [[ "$mode" == setup ]]; then
    # A new environment, never a mutation of the shared Psi4 environment.
    [[ ! -e "$COMPARE_GPU_ENV" ]] || { echo "Dedicated environment already exists; refusing overwrite" >&2; exit 1; }
    /storage/project/r-cs207-0/awallace43/miniconda/bin/conda create -y \
        --prefix "$COMPARE_GPU_ENV" --override-channels -c conda-forge python=3.11 pip
    "$COMPARE_GPU_ENV/bin/python" -m pip install \
        --only-binary=gpu4pyscf-cuda12x,gpu4pyscf-libxc-cuda12x,cupy-cuda12x,cutensor-cu12,pyscf,numpy,scipy \
        -r "$root/requirements-gpu.txt"
    "$COMPARE_GPU_ENV/bin/python" -m pip check
    "$COMPARE_GPU_ENV/bin/python" -m pip freeze --all > "$COMPARE_RUN/metadata/pip-freeze.txt"
    /storage/project/r-cs207-0/awallace43/miniconda/bin/conda list \
        --prefix "$COMPARE_GPU_ENV" --explicit > "$COMPARE_RUN/metadata/conda-explicit.txt"
elif [[ "$mode" != campaign ]]; then
    echo "Unknown mode: $mode" >&2
    exit 2
fi
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then export TMPDIR=/scratch; fi
export TMPDIR="$TMPDIR/gpu-compare-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" PYSCF_TMPDIR="$TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# Bind to the staged development PsiAPI, then the controller deliberately
# supplies independent PYTHONPATH and CUDA library routes to each worker.
psi_exe="${COMPARE_PSI_PACKAGE%/lib/psi4}/bin/psi4"
[[ -x "$psi_exe" ]]
eval "$(PATH="$COMPARE_PSI_ENV/bin:$PATH" \
    LD_LIBRARY_PATH="$COMPARE_PSI_ENV/lib:$COMPARE_PSI_CUDA_LIB" "$psi_exe" --psiapi)"
mkdir -p "$TMPDIR/inputs/geometries"
for system in peptide nanotube protein83 protein157; do
    cp "$COMPARE_RUN/inputs/geometries/$system.xyz" "$TMPDIR/inputs/geometries/"
done
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
nvcc --version
if [[ "$mode" == setup ]]; then
    # Environment installation and small exact-GPU preflight in one allocation.
    systems=(peptide)
    repeats=1
    profiles=()
else
    systems=("$@")
    (( ${#systems[@]} > 0 )) || systems=(peptide nanotube protein83 protein157)
    repeats="${COMPARE_REPEATS:-3}"
    profiles=(--profiles)
fi
exec "$COMPARE_GPU_ENV/bin/python" "$root/compare.py" campaign \
    --systems "${systems[@]}" --repeats "$repeats" "${profiles[@]}" \
    --threads "$OMP_NUM_THREADS" \
    --data "$TMPDIR/inputs" --output "$COMPARE_RUN/campaign" \
    --psi-env "$COMPARE_PSI_ENV" --psi-package "$COMPARE_PSI_PACKAGE" \
    --psi-cuda-lib "$COMPARE_PSI_CUDA_LIB" --gpu-env "$COMPARE_GPU_ENV" \
    --cuda12-lib "$cuda12_lib"
