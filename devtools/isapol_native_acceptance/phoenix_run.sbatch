#!/bin/bash
#SBATCH --job-name=ccasp-native-science
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@180
set -euo pipefail
if [[ $# -lt 7 || $# -gt 9 ]]; then
    printf 'Usage: SOURCE COMMIT TREE BUILD NEW_RUN BASE CASE [GEOMETRY SHA256]\n' >&2
    exit 2
fi
SOURCE=$1 COMMIT=$2 TREE=$3 BUILD=$4 RUN=$5 BASE=$6 CASE=$7
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$BUILD" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-*/build-* ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-*/* ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && "$TREE" =~ ^[[:xdigit:]]{40}$ ]]
[[ "$CASE" = water-smoke || "$CASE" = water-rank2 || "$CASE" = benzene ]]
args=(--build "$BUILD" --run "$RUN" --dependencies "$BASE"
      --commit "$COMMIT" --tree "$TREE" --case "$CASE")
if [[ "$CASE" = benzene ]]; then
    [[ $# -eq 9 && "$9" =~ ^[[:xdigit:]]{64}$ ]]
    args+=(--geometry "$8" --geometry-sha256 "$9")
else
    [[ $# -eq 7 ]]
fi
export PATH="$BASE/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$BASE/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
child='' interrupted=0
on_term() {
    interrupted=1
    if [[ -n "$child" ]]; then kill -TERM -- "-$child" 2>/dev/null || true; fi
}
trap on_term TERM INT
setsid "$BASE/bin/python3.13" -I -B \
    "$SOURCE/devtools/isapol_native_acceptance/phoenix_run.py" "${args[@]}" &
child=$!
set +e
wait "$child"
rc=$?
set -e
if [[ "$interrupted" -ne 0 ]]; then
    printf 'SCIENCE_INTERRUPTED job=%s\n' "${SLURM_JOB_ID:-unknown}" >&2
    exit 143
fi
if [[ "$rc" -ne 0 ]]; then
    printf 'SCIENCE_FAILED exit=%s\n' "$rc" >&2
    exit "$rc"
fi
[[ -s "$RUN/acceptance.json" ]]
printf 'SCIENCE_PAYLOAD_COMPLETE job=%s commit=%s case=%s\n' "$SLURM_JOB_ID" "$COMMIT" "$CASE"
