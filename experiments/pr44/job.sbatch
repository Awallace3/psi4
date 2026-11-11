#!/usr/bin/env bash
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@120
# Resource limits and absolute log paths are provided by submit.py.
set -euo pipefail
umask 027
mode=$1
harness=$2
campaign=$3
index=${4:-build}
output="$campaign/results/$mode-$index/$SLURM_JOB_ID"
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
if [[ -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
    export TMPDIR=/scratch
fi
export TMPDIR="$TMPDIR/pr44-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
printf 'host=%s\njob=%s\nqos=%s\nscratch=%s\n' "$(hostname)" "$SLURM_JOB_ID" "${SLURM_JOB_QOS:-unknown}" "$TMPDIR"
if [[ $mode = build ]]; then
    srun bash "$harness/build.sh" "$harness" "$campaign" &
else
    env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
    export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="$env/bin:/opt/slurm/current/bin:/usr/bin:/bin"
    export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24
    export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
    arm=$("$env/bin/python" -c 'import json,sys; c=json.load(open(sys.argv[1]))["cases"][int(sys.argv[2])]; print("base" if c["arm"]=="base" else "head")' "$harness/config.json" "$index")
    stage="$campaign/builds/$arm/stage"
    test -f "$campaign/builds/$arm/validated.json"
    eval "$("$stage/bin/psi4" --psiapi)"
    export PSI4_DATADIR="$stage/share/psi4"
    srun "$env/bin/python" "$harness/case.py" "$campaign" "$index" "$output" &
fi
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
exit "$status"
