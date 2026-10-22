#!/bin/bash
#SBATCH --job-name=ccasp-bz-basis
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --nodelist=atl1-1-02-007-23-1
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/basis-probe-001/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/basis-probe-001/logs/%x-%j.err
# BASIS METADATA ONLY. No SCF, density, integrals, ISA, response or benzene properties.
set -euo pipefail
umask 027
ROOT=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-phoenix-benzene-pilot-20261021-science-launch-v3
BUILD=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-phoenix-benzene-pilot-20261021-build-smoke-v3
RUN=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/basis-probe-001
DEPS=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/build-smoke-003/deps
SCRATCH=/storage/scratch1/3/awallace43/ccasp-phoenix-benzene-basis-probe-001
ENV=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
STAGE="$BUILD/build_ccasp_phoenix_benzene/stage"
test "$#" = 2
COMMIT=$1
TREE=$2
test -d "$RUN/logs" && test -w "$RUN/logs"
set -o noclobber
printf '%s\n' "$SLURM_JOB_ID" > "$RUN/metadata/job-started"
term_received=0
child=
on_term() {
    term_received=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
on_exit() {
    rc=$?
    (( term_received )) && rc=143
    printf '{"exit_code":%s,"term_received":%s,"complete":%s}\n' \
        "$rc" "$term_received" "$([[ $rc = 0 ]] && echo true || echo false)" \
        > "$RUN/metadata/batch-terminal.json"
}
trap on_term TERM INT HUP
trap on_exit EXIT
ulimit -c 0
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
unset MKL_THREADING_LAYER MKL_INTERFACE_LAYER MKL_SERVICE_FORCE_INTEL
unset OMP_PROC_BIND OMP_PLACES
export PATH="$ENV/bin:/usr/bin:/bin" CONDA_PREFIX="$ENV"
export LD_LIBRARY_PATH="$DEPS/lib:$ENV/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE SLURM_EXPORT_ENV=ALL
mkdir "$SCRATCH/job-$SLURM_JOB_ID"
export TMPDIR="$SCRATCH/job-$SLURM_JOB_ID"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
export PYTHONPATH=
psiapi="$("$ENV/bin/python3.13" -I -S "$STAGE/bin/psi4" --psiapi)"
eval "$psiapi"
cd "$RUN"
/opt/slurm/current/bin/srun --ntasks=1 --cpus-per-task=2 --cpu-bind=none \
    "$ENV/bin/python3.13" -I -S -B "$ROOT/devtools/ccasp_phoenix_benzene/basis_probe.py" \
    run --commit "$COMMIT" --tree "$TREE" &
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
