#!/bin/bash
#SBATCH --job-name=ccasp-grid400-forensic
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
[[ $# -eq 3 ]] || exit 2
SOURCE=$1 COMMIT=$2 RUN=$3
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-monomer-0008-forensic001 ]]
[[ -d "$RUN/logs" && ! -e "$RUN/parent-empty-cache" && ! -e "$RUN/payload" ]]
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/forensic_grid400.py" \
    --commit "$COMMIT" --run "$RUN" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/forensic-report.json" ]] || exit 1
printf 'FORENSIC_COMPONENT_ANALYSIS_COMPLETE_NOT_MODEL_ACCEPTANCE job=%s\n' "$SLURM_JOB_ID"
