#!/bin/bash
#SBATCH --job-name=ccasp-worker-probe
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
[[ $# -ge 6 ]] || { printf 'SOURCE COMMIT BUILD QUALIFICATION PILOT RUN [checkpoint flags]\n' >&2; exit 2; }
SOURCE=$1
COMMIT=$2
BUILD=$3
QUALIFICATION=$4
PILOT=$5
RUN=$6
shift 6
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$BUILD" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-* ]]
[[ "$COMMIT" =~ ^[[:xdigit:]]{40}$ ]]
[[ -d "$RUN/logs" && -s "$QUALIFICATION" && -s "$PILOT" && ! -e "$RUN/parent-cache" ]]
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
export PATH="$MANAGER/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpus-per-task=4 --cpu-bind=none --kill-on-bad-exit=1 \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/phoenix_worker_probe.py" \
    --build "$BUILD" --qualification "$QUALIFICATION" --pilot "$PILOT" \
    --run "$RUN" --commit "$COMMIT" "$@" &
child=$!
if ccasp_wait_child "$child" 20; then rc=0; else rc=$?; fi
[[ "$rc" -eq 0 ]] || exit "$rc"
[[ -s "$RUN/payload/worker-preflight.json" ]]
printf 'WORKER_PROBE_PAYLOAD_COMPLETE job=%s\n' "$SLURM_JOB_ID"
