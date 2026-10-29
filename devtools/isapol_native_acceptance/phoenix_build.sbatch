#!/bin/bash
#SBATCH --job-name=ccasp-native-build
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@180

# Supply absolute --output/--error paths with existing parents at submission.
set -euo pipefail
if [[ $# -ne 5 ]]; then
    printf 'Usage: SOURCE COMMIT TREE NEW_PRIVATE_ROOT DEPENDENCY_ENV\n' >&2
    exit 2
fi
SOURCE=$1
COMMIT=$2
TREE=$3
ROOT=$4
BASE=$5
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$ROOT" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-* ]]
[[ "$BASE" = /storage/project/r-cs207-0/awallace43/miniconda/envs/* ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && "$TREE" =~ ^[[:xdigit:]]{40}$ ]]
[[ -d "$ROOT/logs" && -f "$SOURCE/devtools/isapol_native_acceptance/phoenix_build.py" ]]
export PATH="$BASE/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$BASE/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
child=''
interrupted=0
on_term() {
    interrupted=1
    if [[ -n "$child" ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
    fi
}
trap on_term TERM INT
setsid "$BASE/bin/python3.13" -I -B \
    "$SOURCE/devtools/isapol_native_acceptance/phoenix_build.py" \
    --root "$ROOT" --dependencies "$BASE" --commit "$COMMIT" --tree "$TREE" &
child=$!
set +e
wait "$child"
rc=$?
set -e
if [[ "$interrupted" -ne 0 ]]; then
    printf 'BUILD_INTERRUPTED %s\n' "${SLURM_JOB_ID:-unknown}" >&2
    exit 143
fi
if [[ "$rc" -ne 0 ]]; then
    printf 'BUILD_FAILED exit=%s\n' "$rc" >&2
    exit "$rc"
fi
[[ -s "$ROOT/build-identity.json" ]]
printf 'BUILD_PAYLOAD_COMPLETE job=%s commit=%s\n' "$SLURM_JOB_ID" "$COMMIT"
