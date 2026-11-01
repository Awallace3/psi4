#!/bin/bash
#SBATCH --job-name=ccasp-water-pilot
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@180

set -euo pipefail
[[ $# -eq 7 ]] || { printf 'SOURCE COMMIT BUILD QUALIFICATION WORKER_PREFLIGHT PILOT RUN\n' >&2; exit 2; }
SOURCE=$1
COMMIT=$2
BUILD=$3
QUALIFICATION=$4
WORKER=$5
PILOT=$6
RUN=$7
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$BUILD" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ ]]
[[ -d "$RUN/logs" && -s "$QUALIFICATION" && -s "$WORKER" && -s "$PILOT" ]]
[[ ! -e "$RUN/parent-cache" ]]
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
export PATH="$MANAGER/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
# Non-B TERM warnings reach this explicit Slurm payload step. If the batch
# shell is signalled too, the helper forwards once and allows bounded cleanup.
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpus-per-task=4 \
    --cpu-bind=none --kill-on-bad-exit=1 \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/phoenix_pilot.py" \
    --build "$BUILD" --qualification "$QUALIFICATION" --worker-preflight "$WORKER" \
    --pilot "$PILOT" --run "$RUN" --commit "$COMMIT" &
child=$!
if ccasp_wait_child "$child" 20; then rc=0; else rc=$?; fi
if [[ "$rc" -eq 143 ]]; then
    printf 'NATIVE_WATER_PILOT_INTERRUPTED job=%s\n' "${SLURM_JOB_ID:-unknown}" >&2
    exit 143
fi
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/pilot-outcome.json" ]]
printf 'NATIVE_WATER_DELIVERY_COMPLETE_PENDING_QUALIFICATION job=%s\n' "$SLURM_JOB_ID"
