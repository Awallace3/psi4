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
if [[ $# -ge 1 && "$1" = --grid400-bank-wave ]]; then
    [[ ( $# -eq 8 || $# -eq 10 ) && "$3" =~ ^[0-9a-f]{64}$ ]]
    WAVE_DIR=${2%/plan.json}
    [[ "$2" = "$WAVE_DIR/plan.json" && "$WAVE_DIR" =~ ^/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-waves001/wave-[0-9]{4}-[0-9]{4}$ ]]
    [[ "$4" = "$WAVE_DIR/registration.json" ]]
    [[ "$PILOT" =~ ^"$WAVE_DIR"/cases/(monomer-[0-9]{4})\.json$ ]]
    MID=${BASH_REMATCH[1]}
    [[ "$RUN" = "$WAVE_DIR/members/$MID/audit" && "$CASE" = "$WAVE_DIR/members/$MID/science" && "$PREFLIGHT" = "$WAVE_DIR/members/$MID/worker/payload/worker-preflight.json" ]]
    EXTRA=(--producer grid400 --campaign-scope eligible-wave --remaining-case
        --wave-plan "$2" --wave-plan-sha256 "$3" --wave-registration "$4")
    shift 4
    SEEN_REG=0 SEEN_BARRIER=0 SEEN_BARRIER_SHA=0
    while [[ $# -gt 0 ]]; do
        [[ $# -ge 2 ]]
        case "$1" in
            --wave-registration-sha256)
                [[ "$SEEN_REG" -eq 0 && "$2" =~ ^[0-9a-f]{64}$ ]]
                SEEN_REG=1 ;;
            --wave-science-barrier)
                [[ "$SEEN_BARRIER" -eq 0 && "$2" = "$WAVE_DIR/science-barrier.json" ]]
                SEEN_BARRIER=1 ;;
            --wave-science-barrier-sha256)
                [[ "$SEEN_BARRIER_SHA" -eq 0 && "$2" =~ ^[0-9a-f]{64}$ ]]
                SEEN_BARRIER_SHA=1 ;;
            *) exit 2 ;;
        esac
        EXTRA+=("$1" "$2")
        shift 2
    done
    [[ "$SEEN_BARRIER" -eq 1 && "$SEEN_BARRIER_SHA" -eq 1 ]]
    PRODUCER=grid400-1aaa9e0a30-bank
elif [[ $# -eq 1 && "$1" = --grid400-bank-fresh ]]; then
    EXTRA=(--producer grid400 --campaign-scope eligible-bank --advance-bank-checkpoint)
    PRODUCER=grid400-1aaa9e0a30-bank
    [[ "$PILOT" =~ ^/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-bank001/cases/monomer-[0-9]{4}\.json$ ]]
elif [[ $# -eq 5 && "$1" = --grid400-bank-append ]]; then
    EXTRA=(--producer grid400 --campaign-scope eligible-bank --remaining-case
        --bank-context "$2" --bank-context-sha256 "$3"
        --bank-checkpoint "$4" --bank-checkpoint-sha256 "$5" --advance-bank-checkpoint)
    PRODUCER=grid400-1aaa9e0a30-bank
    [[ "$PILOT" =~ ^/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-bank001/cases/monomer-[0-9]{4}\.json$ ]]
elif [[ $# -ge 1 && "$1" = --ed561666-remaining-case ]]; then
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
elif [[ $# -eq 1 && "$1" = --grid400-fresh ]]; then
    EXTRA=(--producer grid400)
    PRODUCER=grid400-1aaa9e0a30
    [[ "$PILOT" = "$ROOT"/ccasp-native-s66-grid400-1aaa9e0a30-preparation001/pilot.json ]]
else
    for arg in "${EXTRA[@]}"; do
        [[ "$arg" != --producer* && "$arg" != --ed561666-fresh && "$arg" != --grid400* && "$arg" != --campaign-scope* && "$arg" != --bank-* && "$arg" != --advance-bank-checkpoint && "$arg" != --wave-* ]] || exit 2
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
source "$SOURCE/devtools/isapol_s66_campaign/producer_paths.sh"
ccasp_select_paths "$PRODUCER"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/runtime_audit_cb63305.py" \
    --build "$CCASP_BUILD" --qualification "$CCASP_QUALIFICATION" \
    --inputs "$CCASP_INPUTS" --descriptors "$CCASP_DESCRIPTORS" \
    --pilot "$PILOT" --worker-preflight "$PREFLIGHT" --case "$CASE" --run "$RUN" --commit "$COMMIT" "${EXTRA[@]}" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/audit-report.json" ]]
printf 'EXPANDED_PILOT_AUDIT_PASS_NOT_ACCURACY job=%s\n' "$SLURM_JOB_ID"
