#!/bin/bash
#SBATCH --job-name=ccasp-bz-build-smoke
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/build-smoke-003/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/build-smoke-003/logs/%x-%j.err
# No dependency, retry, GPU, watchdog, requeue request or benchmark driver.
set -euo pipefail
umask 027
ROOT=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-phoenix-benzene-pilot-20261021-build-smoke-v3
RUN=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/build-smoke-003
ENV=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
test "$#" = 2
COMMIT=$1
TREE=$2
test -d "$RUN/logs" && test -w "$RUN/logs"
test ! -e "$RUN/metadata/job-started"
set -o noclobber
printf '%s\n' "$SLURM_JOB_ID" > "$RUN/metadata/job-started"
term_received=0
child=
on_term() {
    term_received=1
    # This is only the live srun child owned by this batch shell.
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
on_exit() {
    rc=$?
    (( term_received )) && rc=143
    # A clean status means unsignalled payload exit 0, not scientific parity.
    printf '{"exit_code":%s,"term_received":%s,"complete":%s}\n' \
        "$rc" "$term_received" "$([[ $rc = 0 ]] && echo true || echo false)" \
        > "$RUN/metadata/batch-terminal.json"
}
trap on_term TERM INT HUP
trap on_exit EXIT
unset GCC_ROOT CC CXX FC F77 F90 CFLAGS CXXFLAGS CPPFLAGS FFLAGS LDFLAGS
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
unset CMAKE_PREFIX_PATH CMAKE_MODULE_PATH CMAKE_TOOLCHAIN_FILE CMAKE_ARGS CMAKE_GENERATOR
unset MATH_ROOT HDF5_ROOT MKLROOT CUDA_HOME CUDA_PATH
unset MKL_THREADING_LAYER MKL_INTERFACE_LAYER MKL_SERVICE_FORCE_INTEL
export PATH="$ENV/bin:/usr/bin:/bin"
export CONDA_PREFIX="$ENV"
export LD_LIBRARY_PATH="$RUN/deps/lib:$ENV/lib"
export PYTHONPATH="$RUN/deps/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export CMAKE_BUILD_PARALLEL_LEVEL=1
export SLURM_EXPORT_ENV=ALL
cd "$RUN"
/opt/slurm/current/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none \
    "$ENV/bin/python3.13" -I -S -B \
    "$ROOT/devtools/ccasp_phoenix_benzene/build_smoke.py" run \
    --commit "$COMMIT" --tree "$TREE" &
child=$!
set +e
wait "$child"
rc=$?
if (( term_received )); then
    wait "$child"
    rc=143
fi
set -e
exit "$rc"
