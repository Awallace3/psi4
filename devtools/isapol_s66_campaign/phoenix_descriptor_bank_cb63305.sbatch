#!/bin/bash
#SBATCH --job-name=ccasp-expanded-descriptors
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --signal=TERM@180

# Absolute output/error paths and pre-existing log parents are submission args.
set -euo pipefail
[[ $# -eq 3 ]] || { printf 'SOURCE OPERATIONAL_COMMIT RUN\n' >&2; exit 2; }
SOURCE=$1
COMMIT=$2
RUN=$3
BASE=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
BUILD=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-cb63305f41-build001
QUAL=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-cb63305f41-checks001/payload/installed-validation.json
INPUT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-cb63305f41-preparation001
BANK_SHA=3492e9c0529d5790d69359c5917a6317d7a5c33bf87bdadf0da70b44522b00a1
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && -d "$RUN/logs" && ! -e "$RUN/parent-empty-cache" ]]
export PATH="$BASE/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$BASE/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$BASE/bin/python3.13" -I -S -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/phoenix_descriptor_bank_cb63305.py" \
    --build "$BUILD" --qualification "$QUAL" --bank "$INPUT/descriptor-bank.json" \
    --resources "$INPUT/resource-declarations.json" --inventory "$INPUT/inventory.json" \
    --entries "$INPUT/entries.json" --declarations "$INPUT/declarations.json" \
    --input-validation "$INPUT/input-validation.json" --run "$RUN" --commit "$COMMIT" \
    --expected-bank-sha256 "$BANK_SHA" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
if [[ "$rc" -eq 143 ]]; then
    printf 'DESCRIPTOR_ALLOCATION_INTERRUPTED job=%s\n' "${SLURM_JOB_ID:-unknown}" >&2
    exit 143
fi
if [[ "$rc" -ne 0 ]]; then
    printf 'DESCRIPTOR_ALLOCATION_FAILED exit=%s\n' "$rc" >&2
    exit "$rc"
fi
[[ -s "$RUN/payload/descriptor-allocation.json" ]]
printf 'DESCRIPTOR_ALLOCATION_PAYLOAD_COMPLETE job=%s\n' "$SLURM_JOB_ID"
