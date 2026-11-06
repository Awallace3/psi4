#!/bin/bash
#SBATCH --job-name=ccasp-wave-ops
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
[[ $# -eq 6 ]] || { printf 'SOURCE COMMIT registration|finalizer PLAN PLAN_SHA RUN\n' >&2; exit 2; }
SOURCE=$1 COMMIT=$2 PHASE=$3 PLAN=$4 PLAN_SHA=$5 RUN=$6
ROOT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-waves002
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
[[ "$PHASE" = registration || "$PHASE" = finalizer ]]
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$PLAN" =~ ^${ROOT}/wave-[0-9]{4}-[0-9]{4}/plan\.json$ ]]
[[ "$RUN" = "${PLAN%/plan.json}/$PHASE" && -d "$RUN/logs" && ! -e "$RUN/parent-empty-cache" ]]
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/bank_wave_controller.py" ops \
    --source "$SOURCE" --commit "$COMMIT" --phase "$PHASE" \
    --wave-plan "$PLAN" --wave-plan-sha256 "$PLAN_SHA" --run "$RUN" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
if [[ "$PHASE" = registration ]]; then
    [[ -s "${PLAN%/plan.json}/registration.json" ]]
else
    [[ -s "$RUN/payload/wave-advance.json" && -s "$RUN/payload/bank-checkpoint.json" ]]
fi
printf 'WAVE_OPS_METADATA_COMPLETE job=%s phase=%s\n' "$SLURM_JOB_ID" "$PHASE"
