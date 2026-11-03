#!/bin/bash
#SBATCH --job-name=ccasp-m0008-aux-grid-study
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --signal=TERM@180
set -euo pipefail
[[ $# -eq 3 ]] || { printf 'SOURCE COMMIT RUN\n' >&2; exit 2; }
SOURCE=$1 COMMIT=$2 RUN=$3
ROOT=/storage/project/r-cs207-0/awallace43
MANAGER=$ROOT/miniconda/envs/p4qca
[[ "$SOURCE" = "$ROOT"/gits/* && "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUN" = "$ROOT/runs/ccasp-native-s66-ed561666fc-grid-study001" ]]
[[ -d "$RUN/logs" && ! -e "$RUN/payload" && ! -e "$RUN/parent-empty-cache" ]]
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/grid_charge_diagnostic_ed561666.py" \
    --run "$RUN" --commit "$COMMIT" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/study-receipt.json" ]]
printf 'GRID_STUDY_COMPLETE_NOT_MODEL job=%s\n' "$SLURM_JOB_ID"
