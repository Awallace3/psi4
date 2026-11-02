#!/bin/bash
#SBATCH --job-name=ccasp-benzene-drho-diagnostic
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=TERM@180
set -euo pipefail
[[ $# -eq 4 ]] || { printf 'SOURCE COMMIT GEOMETRY_ONLY_RUNTIME RUN\n' >&2; exit 2; }
SOURCE=$1 COMMIT=$2 RUNTIME=$3 RUN=$4
ROOT=/storage/project/r-cs207-0/awallace43/runs
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$RUN" = "$ROOT"/ccasp-native-s66-* && -d "$RUN/logs" && ! -e "$RUN/parent-empty-cache" ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && -s "$RUNTIME" ]]
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/drho_diagnostic_cb63305.py" \
    --build "$ROOT/ccasp-native-s66-cb63305f41-build001" \
    --runtime "$RUNTIME" --run "$RUN" --commit "$COMMIT" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/diagnostic-receipt.json" ]]
printf 'DRHO_DIAGNOSTIC_CAPTURE_COMPLETE_NOT_MODEL job=%s\n' "$SLURM_JOB_ID"
