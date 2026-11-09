#!/bin/bash
#SBATCH --job-name=ccasp-wave-ops
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --signal=TERM@180
#SBATCH --no-requeue
set -euo pipefail
[[ $# -eq 6 || $# -eq 8 ]] || { printf 'SOURCE COMMIT registration|finalizer PLAN PLAN_SHA RUN [qomp ADOPTION_SHA]\n' >&2; exit 2; }
SOURCE=$1 COMMIT=$2 PHASE=$3 PLAN=$4 PLAN_SHA=$5 RUN=$6
ROOT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-waves002
MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca
if [[ $# -eq 8 ]]; then
    [[ "$7" = qomp && "$8" =~ ^[0-9a-f]{64}$ ]]
    [[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-qomp-production148-v4 ]]
    ROOT=/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-qomp148-waves004
    [[ "$PLAN" =~ ^${ROOT}/wave-[0-9]{4}-[0-9]{4}/plan\.json$ ]]
    [[ "$RUN" = "${PLAN%/plan.json}/$PHASE" && -d "$RUN/logs" && ! -e "$RUN/control" ]]
    [[ "$PHASE" = registration || "$PHASE" = finalizer ]]
    [[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
    export PATH=/usr/bin:/bin
    unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME BASH_ENV ENV
    umask 077
    /usr/bin/mkdir -m 700 "$RUN/control"
    /usr/bin/mkdir -m 700 "$RUN/control/home" "$RUN/control/cache" "$RUN/control/cwd"
    source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
    bootstrap='import signal,sys
def stop(n,f): raise KeyboardInterrupt("QOMP wave ops signal "+str(n))
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
p=sys.argv.pop(1)
exec(compile(open(p,"rb").read(),p,"exec"),dict(__name__="__main__",__file__=p,__package__=None))'
    cd "$RUN/control/cwd"
    /usr/bin/setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
      /usr/bin/env -i USER=awallace43 LOGNAME=awallace43 HOME="$RUN/control/home" \
      PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
      SLURM_CLUSTER_NAME="$SLURM_CLUSTER_NAME" SLURMD_NODENAME="$SLURMD_NODENAME" \
      "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/control/cache" \
      -c "$bootstrap" "$SOURCE/devtools/isapol_s66_campaign/bank_wave_controller.py" ops \
      --producer qomp --adoption-sha256 "$8" --source "$SOURCE" --commit "$COMMIT" --phase "$PHASE" \
      --wave-plan "$PLAN" --wave-plan-sha256 "$PLAN_SHA" --run "$RUN" &
    child=$!
    set +e
    ccasp_wait_child "$child" 60
    rc=$?
    exit "$rc"
fi
[[ "$PHASE" = registration || "$PHASE" = finalizer ]]
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/* ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$PLAN" =~ ^${ROOT}/wave-[0-9]{4}-[0-9]{4}/plan\.json$ ]]
[[ "$RUN" = "${PLAN%/plan.json}/$PHASE" && -d "$RUN/logs" && ! -e "$RUN/parent-empty-cache" ]]
export PATH="$MANAGER/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PSIPATH PSIDATADIR LD_PRELOAD
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
    "$MANAGER/bin/python" -I -B -X "pycache_prefix=$RUN/parent-empty-cache" \
    "$SOURCE/devtools/isapol_s66_campaign/bank_wave_controller.py" ops \
    --source "$SOURCE" --commit "$COMMIT" --phase "$PHASE" \
    --wave-plan "$PLAN" --wave-plan-sha256 "$PLAN_SHA" --run "$RUN" &
child=$!
set +e
ccasp_wait_child "$child" 20
rc=$?
set -e
[[ "$rc" -eq 0 ]] || exit "$rc"
if [[ "$PHASE" = registration ]]; then
    [[ -s "${PLAN%/plan.json}/registration.json" ]]
else
    [[ -s "$RUN/payload/wave-advance.json" && -s "$RUN/payload/bank-checkpoint.json" ]]
fi
printf 'WAVE_OPS_METADATA_COMPLETE job=%s phase=%s\n' "$SLURM_JOB_ID" "$PHASE"
