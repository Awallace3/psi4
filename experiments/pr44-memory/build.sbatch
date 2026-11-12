#!/usr/bin/env bash
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@120
#SBATCH --no-requeue
set -euo pipefail
root=$1
campaign=$2
commit=$3
mkdir -p "$campaign/build-job-$SLURM_JOB_ID"
cd "$campaign/build-job-$SLURM_JOB_ID"
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
srun bash "$root/experiments/pr44-memory/build.sh" "$root" "$campaign" "$commit" &
child=$!
set +e
wait "$child"; code=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; code=$?; done
exit "$code"
