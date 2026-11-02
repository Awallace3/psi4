#!/bin/bash
#SBATCH --job-name=ccasp-cb-worker-probe
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --signal=TERM@180
set -euo pipefail
[[ $# -eq 4 || $# -eq 5 ]] || { printf 'SOURCE OPERATIONAL_COMMIT PILOT RUN [--remaining-case]\n' >&2; exit 2; }
SOURCE=$1 COMMIT=$2 PILOT=$3 RUN=$4
EXTRA=()
if [[ $# -eq 5 ]]; then
    [[ "$5" = --remaining-case ]] || exit 2
    EXTRA=(--remaining-case)
fi
ROOT=/storage/project/r-cs207-0/awallace43/runs
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$RUN" = "$ROOT"/ccasp-native-s66-* && -d "$RUN/logs" && ! -e "$RUN/parent-empty-cache" ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && -s "$PILOT" ]]
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/phoenix_worker_cb63305.py" \
    --build "$ROOT/ccasp-native-s66-cb63305f41-build001" \
    --qualification "$ROOT/ccasp-native-s66-cb63305f41-checks001/payload/installed-validation.json" \
    --inputs "$ROOT/ccasp-native-s66-cb63305f41-preparation001" \
    --descriptors "$ROOT/ccasp-native-s66-cb63305f41-descriptors001/payload" \
    --pilot "$PILOT" --run "$RUN" --commit "$COMMIT" "${EXTRA[@]}" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/worker-preflight.json" ]]
printf 'EXPANDED_WORKER_PREFLIGHT_PAYLOAD_COMPLETE job=%s\n' "$SLURM_JOB_ID"
