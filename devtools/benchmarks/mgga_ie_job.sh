#!/usr/bin/env bash
# Args: absolute harness path, pinned commit, unique run directory
#SBATCH --job-name=cuest-mgga-ie
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@120
set -euo pipefail
umask 027
harness=$1
commit=$2
run=$3
extra=("${@:4}")
driver=${MGGA_DRIVER:-mgga_ie.py}
case "$driver" in mgga_ie.py|mgga_probe.py|mgga_grid_probe.py) ;; *) echo "Invalid MGGA_DRIVER" >&2; exit 2;; esac
root=/storage/project/r-cs207-0/awallace43
test "$(git -C "$harness" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$harness" status --porcelain --untracked-files=no)"
set +u
source "$root/miniconda/etc/profile.d/conda.sh"
conda activate p4cuest_sapt
set -u
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
scratch_harness="$root/gits/psi4.saptdft-cuest-protein157-20261004"
test "$(git -C "$scratch_harness" rev-parse HEAD)" = 7d897a7f203297754087ea77536619c378191c8b
TMPDIR=$("$CONDA_PREFIX/bin/python" "$scratch_harness/devtools/benchmarks/node_scratch.py")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
host="$root/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage/lib/psi4"
cuda="$root/gits/psi4.saptdft_cuest/build_cuda_libxc/stage/lib/psi4"
cuda_lib="$root/software/libxc-7.1.2-cuda/lib64"
mkdir -p "$run/metadata"
if [[ $driver == mgga_grid_probe.py ]]; then
    source /etc/profile.d/z00_lmod_pace.sh
    module load gcc/12.3.0 cuda/12.6.1
    command -v nvcc >/dev/null
    cuda_home=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    test ! -e "$run/mgga_capture.so"
    g++ -std=c++17 -O2 -fPIC -shared \
        -I"$CONDA_PREFIX/include" -I"$cuda_home/include" \
        "$harness/devtools/benchmarks/mgga_capture.cc" \
        -L"$CONDA_PREFIX/lib" -L"$cuda_home/lib64" -lcuest -lcudart -ldl \
        -o "$run/mgga_capture.so"
    extra+=(--shim "$run/mgga_capture.so")
    g++ --version > "$run/metadata/compiler.txt"
    sha256sum "$run/mgga_capture.so" > "$run/metadata/capture-shim.sha256"
fi
cd "$run"
{
    date -Iseconds
    scontrol show job "$SLURM_JOB_ID" -o
    nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
    for package in "$host" "$cuda"; do
        printf 'package=%s\n' "$package"
        sha256sum "$package"/core*.so
        if [[ $package == "$cuda" ]]; then
            LD_LIBRARY_PATH="$cuda_lib:$LD_LIBRARY_PATH" ldd "$package"/core*.so
        else
            ldd "$package"/core*.so
        fi
    done
} > "$run/metadata/environment-$SLURM_JOB_ID.txt"
child=
terminated=0
trap 'terminated=1; [[ -z $child ]] || kill -TERM "$child" 2>/dev/null || true' TERM
srun /usr/bin/env TMPDIR="$TMPDIR" SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" \
    "$CONDA_PREFIX/bin/python" "$harness/devtools/benchmarks/$driver" \
    --output "$run/results-$SLURM_JOB_ID" --host-package "$host" --cuda-package "$cuda" \
    --cuda-lib-dir "$cuda_lib" "${extra[@]}" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
(( terminated )) && status=143
printf 'exit=%s terminated=%s\n' "$status" "$terminated" >> "$run/metadata/environment-$SLURM_JOB_ID.txt"
exit "$status"
