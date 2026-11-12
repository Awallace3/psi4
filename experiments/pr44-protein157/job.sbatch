#!/usr/bin/env bash
#SBATCH --account=gts-cs207
#SBATCH --qos=inferno
#SBATCH --partition=cpu-gnr
#SBATCH --constraint=graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --mem=1400G
#SBATCH --tmp=2600G
#SBATCH --time=12:00:00
#SBATCH --signal=TERM@120
#SBATCH --no-requeue
# Absolute stdout/stderr paths are supplied by submit.py.
set -euo pipefail
umask 027
harness=$1
campaign=$2
arm=$3
commit=$4
output="$campaign/results/$arm/$SLURM_JOB_ID"
mkdir -p "$output"
cd "$output"
term_received=0
child=
on_term() {
    term_received=1
    [[ -n $child ]] && kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    status=$?
    (( term_received )) && status=143
    printf 'exit_code=%s\nterm_received=%s\n' "$status" "$term_received" > "$output/status"
    if (( status == 0 )); then touch "$output/COMPLETED"; else touch "$output/FAILED"; fi
}
trap on_exit EXIT
trap on_term TERM
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
build_campaign=$("$env/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["build_campaign"])' "$harness/config.json")
python="$build_campaign/runtime-pydantic2/bin/python"
stage="$build_campaign/builds/$arm/stage"
test -x "$python"
test -f "$build_campaign/builds/$arm/validated.json"
export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export OMP_NUM_THREADS=192 MKL_NUM_THREADS=192 OPENBLAS_NUM_THREADS=192
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then export TMPDIR=/scratch; fi
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
eval "$("$stage/bin/psi4" --psiapi)"
export PSI4_DATADIR="$stage/share/psi4"
printf 'host=%s\njob=%s\naccount=%s\nqos=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${SLURM_JOB_ACCOUNT:-unknown}" "${SLURM_JOB_QOS:-unknown}"
srun "$python" "$harness/run.py" "$arm" "$output" "$commit" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
