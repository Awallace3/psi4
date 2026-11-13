#!/usr/bin/env bash
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=cpu-gnr
#SBATCH --constraint=graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --mem=1400G
#SBATCH --tmp=2600G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@120
#SBATCH --no-requeue
set -euo pipefail
root=$1
campaign=$2
case_id=$3
mode=$4
memory=$5
commit=$6
build_campaign=$7
checkpoint=$8
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
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
python=/storage/scratch1/3/awallace43/psi4-pr44-283ae89a32-embers-v1/runtime-pydantic2/bin/python
export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export OMP_NUM_THREADS=192 MKL_NUM_THREADS=192 OPENBLAS_NUM_THREADS=192
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PR44_MEMORY_TRACE=1
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then export TMPDIR=/scratch; fi
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
eval "$("$build_campaign/build/stage/bin/psi4" --psiapi)"
export PSI4_DATADIR="$build_campaign/build/stage/share/psi4"
extra=()
if [[ $mode = replay ]]; then
    extra=(--replay-cache "$checkpoint")
elif [[ $mode != full ]]; then
    exit 2
fi
srun "$python" "$root/experiments/pr44-memory/run.py" \
    "$build_campaign" "$memory" "$output" "$commit" "${extra[@]}" &
child=$!
set +e
wait "$child"; code=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; code=$?; done
exit "$code"
