#!/bin/bash
#SBATCH --job-name=ccasp-cb-pilot-audit
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
[[ $# -ge 6 ]] || { printf 'SOURCE OPERATIONAL_COMMIT PILOT PREFLIGHT CASE RUN [remaining-case options]\n' >&2; exit 2; }
SOURCE=$1 COMMIT=$2 PILOT=$3 PREFLIGHT=$4 CASE=$5 RUN=$6
shift 6
EXTRA=("$@")
ROOT=/storage/project/r-cs207-0/awallace43/runs
PRODUCER=cb63305f41
if [[ $# -ge 1 && "$1" = --ed561666-remaining-case ]]; then
    EXTRA=(--producer ed561666 --remaining-case)
    PRODUCER=ed561666fc
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --advance-checkpoint)
                EXTRA+=("$1")
                shift
                ;;
            --owned-checkpoint|--checkpoint-sha256|--checkpoint-proofs|--accepted-audit|--accepted-outcome|--accepted-pilot|--accepted-envelope)
                [[ $# -ge 2 ]] || exit 2
                EXTRA+=("$1" "$2")
                shift 2
                ;;
            *) exit 2 ;;
        esac
    done
    [[ "$PILOT" =~ ^/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-ed561666fc-remaining-preparation001/cases/monomer-[0-9]{4}\.json$ ]]
elif [[ $# -eq 1 && "$1" = --ed561666-fresh ]]; then
    EXTRA=(--producer ed561666)
    PRODUCER=ed561666fc
    [[ "$PILOT" = "$ROOT"/ccasp-native-s66-ed561666fc-benzene-preparation001/pilot.json ]]
else
    for arg in "${EXTRA[@]}"; do
        [[ "$arg" != --producer* && "$arg" != --ed561666-fresh ]] || exit 2
    done
fi
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$RUN" = "$ROOT"/ccasp-native-s66-* && -d "$RUN/logs" && ! -e "$RUN/parent-empty-cache" ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && -s "$PILOT" && -s "$PREFLIGHT" ]]
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/runtime_audit_cb63305.py" \
    --build "$ROOT/ccasp-native-s66-$PRODUCER-build001" \
    --qualification "$ROOT/ccasp-native-s66-$PRODUCER-checks001/payload/installed-validation.json" \
    --inputs "$ROOT/ccasp-native-s66-$PRODUCER-preparation001" \
    --descriptors "$ROOT/ccasp-native-s66-$PRODUCER-descriptors001/payload" \
    --pilot "$PILOT" --worker-preflight "$PREFLIGHT" --case "$CASE" --run "$RUN" --commit "$COMMIT" "${EXTRA[@]}" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/audit-report.json" ]]
printf 'EXPANDED_PILOT_AUDIT_PASS_NOT_ACCURACY job=%s\n' "$SLURM_JOB_ID"
