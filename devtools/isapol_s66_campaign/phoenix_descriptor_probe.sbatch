#!/bin/bash
#SBATCH --job-name=ccasp-descriptors
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --signal=TERM@180

set -euo pipefail
[[ $# -ge 6 ]] || { printf 'SOURCE COMMIT BUILD QUALIFICATION RUN PILOT [PILOT...]\n' >&2; exit 2; }
SOURCE=$1
COMMIT=$2
BUILD=$3
QUALIFICATION=$4
RUN=$5
shift 5
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$BUILD" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ && -d "$RUN/logs" && ! -e "$RUN/parent-cache" ]]
for pilot in "$@"; do [[ -s "$pilot" ]]; done
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpus-per-task=4 --cpu-bind=none --kill-on-bad-exit=1 \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/phoenix_descriptor_probe.py" \
    --build "$BUILD" --qualification "$QUALIFICATION" --run "$RUN" --commit "$COMMIT" \
    --memory-bytes 21474836480 --pilots "$@" &
child=$!
if ccasp_wait_child "$child" 20; then rc=0; else rc=$?; fi
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/descriptor-allocation.json" ]]
printf 'DESCRIPTOR_PROBE_PAYLOAD_COMPLETE job=%s\n' "$SLURM_JOB_ID"
