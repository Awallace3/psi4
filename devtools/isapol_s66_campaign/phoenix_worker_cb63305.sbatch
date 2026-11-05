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
[[ $# -eq 4 || $# -eq 5 || $# -eq 9 ]] || { printf 'SOURCE COMMIT CASE RUN [MODE [BANK_CONTEXT CONTEXT_SHA BANK_CHECKPOINT CHECKPOINT_SHA]]\n' >&2; exit 2; }
SOURCE=$1 COMMIT=$2 PILOT=$3 RUN=$4
EXTRA=()
PRODUCER=cb63305f41
if [[ $# -ge 5 ]]; then
    if [[ "$5" = --grid400-bank-append ]]; then
        [[ $# -eq 9 ]]
    else
        [[ $# -eq 5 ]]
    fi
    case "$5" in
        --remaining-case) EXTRA=(--remaining-case) ;;
        --ed561666-fresh)
            EXTRA=(--producer ed561666)
            PRODUCER=ed561666fc
            [[ "$PILOT" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-ed561666fc-benzene-preparation001/pilot.json ]]
            ;;
        --ed561666-remaining-case)
            EXTRA=(--producer ed561666 --remaining-case)
            PRODUCER=ed561666fc
            [[ "$PILOT" =~ ^/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-ed561666fc-remaining-preparation001/cases/monomer-[0-9]{4}\.json$ ]]
            ;;
        --grid400-fresh)
            EXTRA=(--producer grid400)
            PRODUCER=grid400-1aaa9e0a30
            [[ "$PILOT" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-preparation001/pilot.json ]]
            ;;
        --grid400-bank-fresh|--grid400-bank-append)
            EXTRA=(--producer grid400 --campaign-scope eligible-bank)
            PRODUCER=grid400-1aaa9e0a30-bank
            [[ "$PILOT" =~ ^/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-bank001/cases/monomer-[0-9]{4}\.json$ ]]
            if [[ "$5" = --grid400-bank-append ]]; then
                [[ "$7" =~ ^[0-9a-f]{64}$ && "$9" =~ ^[0-9a-f]{64}$ ]]
                EXTRA+=(--remaining-case --bank-context "$6" --bank-context-sha256 "$7"
                        --bank-checkpoint "$8" --bank-checkpoint-sha256 "$9")
            fi
            ;;
        *) exit 2 ;;
    esac
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
source "$SOURCE/devtools/isapol_s66_campaign/producer_paths.sh"
ccasp_select_paths "$PRODUCER"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/phoenix_worker_cb63305.py" \
    --build "$CCASP_BUILD" --qualification "$CCASP_QUALIFICATION" \
    --inputs "$CCASP_INPUTS" --descriptors "$CCASP_DESCRIPTORS" \
    --pilot "$PILOT" --run "$RUN" --commit "$COMMIT" "${EXTRA[@]}" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/worker-preflight.json" ]]
printf 'EXPANDED_WORKER_PREFLIGHT_PAYLOAD_COMPLETE job=%s\n' "$SLURM_JOB_ID"
