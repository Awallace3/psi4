#!/bin/bash
#SBATCH --job-name=ccasp-bz-capture
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --nodelist=atl1-1-02-007-23-1
#SBATCH --signal=TERM@120
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/ccasp-benzene-lw-integrated-20261022/capture-001/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/ccasp-benzene-lw-integrated-20261022/capture-001/logs/%x-%j.err
# Prepared only. Coordinator owns publication, manifest pins, test-only and ONE submission.
set -euo pipefail
umask 077
ROOT=/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-benzene-lw-integrated-20261022
RUN=/storage/project/r-cs207-0/awallace43/runs/ccasp-benzene-lw-integrated-20261022/capture-001
SCRATCH=/storage/scratch1/3/awallace43/ccasp-benzene-lw-integrated-20261022-capture001
DEPS=/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/build-smoke-003/deps
ENV=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
term_received=0
child=
child_ok=0
set -o noclobber
on_term() {
    term_received=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
on_exit() {
    rc=$?
    trap - EXIT
    # Finalize one receipt after child status is known; no payload work below.
    # Freeze the observed signal state so receipt/status cannot diverge mid-write.
    trap '' TERM INT HUP
    (( term_received )) && rc=143
    complete=false
    if [[ $rc = 0 && $child_ok = 1 && $term_received = 0 ]]; then complete=true; fi
    printf '{"exit_code":%s,"term_received":%s,"child_unsignalled_zero":%s,"complete":%s}\n' \
        "$rc" "$term_received" "$child_ok" "$complete" > "$RUN/metadata/batch-terminal.json"
    (( term_received )) && rc=143
    exit "$rc"
}
trap on_term TERM INT HUP
trap on_exit EXIT
test "$#" = 4
COMMIT=$1
TREE=$2
RUNTIME_SHA=$3
DERIVATIVE_SHA=$4
[[ $COMMIT =~ ^[0-9a-f]{40}$ && $TREE =~ ^[0-9a-f]{40}$ ]]
[[ $RUNTIME_SHA =~ ^[0-9a-f]{64}$ && $DERIVATIVE_SHA =~ ^[0-9a-f]{64}$ ]]
test -w "$RUN/logs" && test -w "$RUN/metadata"
printf '%s\n' "$SLURM_JOB_ID" > "$RUN/metadata/job-started"
ulimit -c 0
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD GCC_ROOT
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH
unset MKL_THREADING_LAYER MKL_INTERFACE_LAYER MKL_SERVICE_FORCE_INTEL
unset OMP_PROC_BIND OMP_PLACES
export PATH="$ENV/bin:/usr/bin:/bin" CONDA_PREFIX="$ENV"
export LD_LIBRARY_PATH="$DEPS/lib:$ENV/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE SLURM_EXPORT_ENV=ALL
mkdir "$SCRATCH/bootstrap-$SLURM_JOB_ID"
export TMPDIR="$SCRATCH/bootstrap-$SLURM_JOB_ID"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
cd "$RUN"
if (( term_received )); then exit 143; fi
/opt/slurm/current/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none \
    "$ENV/bin/python3.13" -I -S -B "$ROOT/devtools/ccasp_phoenix_benzene/integrated_job.py" \
    run --commit "$COMMIT" --tree "$TREE" \
    --runtime-sha256 "$RUNTIME_SHA" --derivative-sha256 "$DERIVATIVE_SHA" &
child=$!
if (( term_received )); then kill -TERM "$child" || true; fi
set +e
wait "$child"
rc=$?
if (( term_received )); then
    # Child has already received TERM. Do not let repeated shell signals
    # interrupt its reaping wait and publish terminal status while it is live.
    trap '' TERM INT HUP
    wait "$child"
    rc=143
fi
set -e
if [[ $rc = 0 && $term_received = 0 ]]; then child_ok=1; fi
exit "$rc"
