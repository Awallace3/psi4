#!/bin/bash
#SBATCH --job-name=ccasp-native-checks
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@180

# Supply absolute output/error paths whose parent already exists.
set -euo pipefail
[[ $# -eq 7 || $# -eq 8 ]] || { printf 'HARNESS BUILD_COMMIT TREE HARNESS_COMMIT BUILD RUN DEPENDENCIES [legacy|grid400]\n' >&2; exit 2; }
HARNESS=$1
COMMIT=$2
TREE=$3
HARNESS_COMMIT=$4
BUILD=$5
RUN=$6
BASE=$7
SUITE=${8:-legacy}
[[ "$SUITE" = legacy || "$SUITE" = grid400 ]]
[[ "$HARNESS" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$BUILD" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-* ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$BASE" = /storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && "$TREE" =~ ^[[:xdigit:]]{40}$ && "$HARNESS_COMMIT" =~ ^[[:xdigit:]]{40}$ ]]
[[ -d "$RUN/logs" && -f "$HARNESS/devtools/isapol_s66_campaign/phoenix_installed_checks.py" ]]
[[ ! -e "$RUN/parent-empty-cache" ]]
export PATH="$BASE/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$BASE/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
child=''
interrupted=0
on_term() {
    interrupted=1
    if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; fi
}
trap on_term TERM INT
setsid "$BASE/bin/python3.13" -I -S -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$HARNESS/devtools/isapol_s66_campaign/phoenix_installed_checks.py" \
    --build "$BUILD" --run "$RUN" --dependencies "$BASE" \
    --commit "$COMMIT" --tree "$TREE" --harness-commit "$HARNESS_COMMIT" --suite "$SUITE" &
child=$!
set +e
wait "$child"
rc=$?
set -e
if [[ "$interrupted" -ne 0 ]]; then
    printf 'INSTALLED_TESTS_INTERRUPTED job=%s\n' "${SLURM_JOB_ID:-unknown}" >&2
    exit 143
fi
if [[ "$rc" -ne 0 ]]; then
    printf 'INSTALLED_TESTS_FAILED exit=%s\n' "$rc" >&2
    exit "$rc"
fi
[[ -s "$RUN/payload/installed-validation.json" ]]
printf 'INSTALLED_TESTS_PAYLOAD_COMPLETE job=%s build=%s\n' "$SLURM_JOB_ID" "$COMMIT"
