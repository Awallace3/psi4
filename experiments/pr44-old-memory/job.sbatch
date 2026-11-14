#!/usr/bin/env bash
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --signal=TERM@120
#SBATCH --no-requeue
set -euo pipefail
root=$1
campaign=$2
case_id=$3
memory=$4
commit=$5
mode=$6
output="$campaign/results/$case_id/$SLURM_JOB_ID"
mkdir -p "$output"
cd "$output"
child=
term_received=0
on_term() { term_received=1; [[ -n $child ]] && kill -TERM "$child" 2>/dev/null || true; }
on_exit() {
    code=$?
    (( term_received )) && code=143
    printf 'exit_code=%s\nterm_received=%s\n' "$code" "$term_received" > status
    if (( code == 0 )); then touch COMPLETED; else touch FAILED; fi
}
trap on_term TERM
trap on_exit EXIT
test "$(git -C "$root" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$root" status --porcelain)"
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
build=/storage/scratch1/3/awallace43/psi4-pr44-283ae89a32-embers-v1
python="$build/runtime-pydantic2/bin/python"
stage="$build/builds/base/stage"
export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then export TMPDIR=/scratch; fi
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
eval "$("$stage/bin/psi4" --psiapi)"
export PSI4_DATADIR="$stage/share/psi4"
printf 'job=%s\naccount=%s\nqos=%s\n' "$SLURM_JOB_ID" "${SLURM_JOB_ACCOUNT:-unknown}" "${SLURM_JOB_QOS:-unknown}"
if [[ $mode = smoke ]]; then
    export PR44_EXPECTED_CORE_SHA256=bfd8cbebf6a3c2ae0d69bdb6344f9f446204cfdfcedbe24296b11cc51eaa8fe8
    export PR44_AUDIT_PROTEIN=1
    export PR44_EXPECTED_STAGE="$stage"
    command=("$python" "$root/experiments/pr44-memory/check_namespace.py" "$output/smoke")
elif [[ $mode = full ]]; then
    command=("$python" "$root/experiments/pr44-protein157/run.py" base "$output" "$commit"
             --memory-gb "$memory" --fix-disk-jk-namespace --energy-tolerance 1e-7)
else
    exit 2
fi
srun "${command[@]}" &
child=$!
set +e
wait "$child"; code=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; code=$?; done
exit "$code"
