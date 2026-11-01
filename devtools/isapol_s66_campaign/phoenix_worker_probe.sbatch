#!/bin/bash
#SBATCH --job-name=ccasp-worker-probe
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --signal=TERM@180

set -euo pipefail
[[ $# -eq 6 ]] || { printf 'SOURCE COMMIT BUILD QUALIFICATION PILOT RUN\n' >&2; exit 2; }
SOURCE=$1
COMMIT=$2
BUILD=$3
QUALIFICATION=$4
PILOT=$5
RUN=$6
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$BUILD" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ ]]
[[ -d "$RUN/logs" && -s "$QUALIFICATION" && -s "$PILOT" && ! -e "$RUN/parent-cache" ]]
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
export PATH="$MANAGER/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
child=''
interrupted=0
on_term() {
    interrupted=1
    if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; fi
}
trap on_term TERM INT
setsid "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/phoenix_worker_probe.py" \
    --build "$BUILD" --qualification "$QUALIFICATION" --pilot "$PILOT" \
    --run "$RUN" --commit "$COMMIT" &
child=$!
set +e
wait "$child"
rc=$?
set -e
if [[ "$interrupted" -ne 0 ]]; then
    printf 'WORKER_PROBE_INTERRUPTED job=%s\n' "${SLURM_JOB_ID:-unknown}" >&2
    exit 143
fi
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/worker-preflight.json" ]]
printf 'WORKER_PROBE_PAYLOAD_COMPLETE job=%s\n' "$SLURM_JOB_ID"
