#!/bin/bash
#SBATCH --job-name=p157-cuest-grac
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=gpu-a100
#SBATCH --qos=embers
#SBATCH --constraint=A100-80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-timing/20260909-protein157-gpu/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-timing/20260909-protein157-gpu/logs/%x-%j.err
set -eo pipefail
umask 027
RUN=/storage/project/r-cs207-0/awallace43/runs/psi4-cuest-timing/20260909-protein157-gpu
PSI4_BUILD=/storage/project/r-cs207-0/awallace43/runs/psi4-grac/20260908T200831Z/build
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate p4cuest_sapt
unset GCC_ROOT GCCROOT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then export TMPDIR=/scratch; fi
export TMPDIR="$(mktemp -d "$TMPDIR/psi4-p157-${SLURM_JOB_ID}-XXXXXX")"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
eval "$("$PSI4_BUILD/stage/bin/psi4" --psiapi)"
cd "$RUN"
finish() {
  rc=$?
  printf 'exit_code=%s\nfinished=%s\n' "$rc" "$(date --iso-8601=seconds)" >> metadata/runtime.txt
  if [ "$rc" -eq 0 ]; then touch metadata/COMPLETED; else touch metadata/FAILED; fi
}
trap finish EXIT
printf 'job=%s\nnode=%s\nscratch=%s\nstarted=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$TMPDIR" "$(date --iso-8601=seconds)" > metadata/runtime.txt
sha256sum --check expected-binary.sha256
sha256sum --check expected-inputs.sha256
{
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
  lscpu
  taskset -pc $$
  scontrol show job "$SLURM_JOB_ID"
} > metadata/hardware.txt
conda list --json > metadata/conda-packages.json
"$CONDA_PREFIX/bin/python" - <<'PY'
import subprocess
values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True)
assert any(int(v.strip()) >= 80000 for v in values.splitlines()), '80 GB GPU required'
import psi4
print(psi4.__version__, psi4.__file__)
PY
# Allocation-local GPU/GRAC smoke test, before the larger single measurement.
srun --ntasks=1 --cpus-per-task=8 "$CONDA_PREFIX/bin/python" saptdft_cuest_grac.py --case --system water --basis cc-pvdz --mode gpu --output "$RUN/preflight" --threads 8 --memory '112 GiB' > logs/preflight.log 2>&1
srun --ntasks=1 --cpus-per-task=8 "$CONDA_PREFIX/bin/python" saptdft_cuest_grac.py --case --system protein157 --basis '6-31+g**' --mode gpu --output "$RUN/results/protein157-gpu-1" --threads 8 --memory '112 GiB' > logs/protein157.log 2>&1
sha256sum --check expected-binary.sha256
printf 'PROTEIN157_GPU_COMPLETE\n'
