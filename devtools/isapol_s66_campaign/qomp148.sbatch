#!/bin/bash
# Newv4 registered-wave member ONLY; v3 serial checkout stays immutable.
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --signal=TERM@180
#SBATCH --no-requeue
set -euo pipefail
umask 077
[[ $# -eq 14 ]] || exit 2
SOURCE=$1 COMMIT=$2 PHASE=$3 RUN=$4 PILOT=$5 WP=$6 WH=$7 SCI=$8 PLAN=$9
PLAN_SHA=${10} REG=${11} REG_SHA=${12} BARRIER=${13} BARRIER_SHA=${14}
P=/storage/project/r-cs207-0/awallace43
[[ "$SOURCE" = "$P/gits/psi4.ccasp-native-qomp-production148-v4" ]]
ROOT="$P/runs/ccasp-native-s66-qomp148-waves004"
[[ "$PLAN" =~ ^${ROOT}/wave-[0-9]{4}-[0-9]{4}/plan\.json$ ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
MID=${PILOT##*/}; MID=${MID%.json}; D=${PLAN%/plan.json}
[[ "$MID" =~ ^monomer-[0-9]{4}$ && "$PILOT" = "$D/cases/$MID.json" ]]
[[ "$RUN" = "$D/members/$MID/$PHASE" && -d "$RUN/logs" && ! -e "$RUN/controller" ]]
args=(--build "$P/runs/ccasp-native-s66-qomp-fb11-build002"
 --qualification "$P/runs/ccasp-native-s66-qomp-runtime146-check001/inputs/receipts/qualification/installed-validation.json"
 --inputs "$P/runs/ccasp-native-s66-qomp-profile146-preparation001"
 --descriptors "$P/runs/ccasp-native-s66-qomp-profile146-descriptors001/payload"
 --pilot "$PILOT" --run "$RUN" --commit "$COMMIT" --producer qomp
 --campaign-scope qomp-wave --remaining-case --wave-plan "$PLAN" --wave-plan-sha256 "$PLAN_SHA")
case "$PHASE" in
 worker)
  [[ "$WP" = - && "$WH" = - && "$SCI" = - && "$REG" = - && "$REG_SHA" = - && "$BARRIER" = - && "$BARRIER_SHA" = - ]]
  ENTRY=phoenix_worker_cb63305.py ;;
 science|audit)
  [[ "$WP" = "$D/members/$MID/worker/payload/worker-preflight.json" && "$WH" =~ ^[0-9a-f]{64}$ ]]
  [[ "$REG" = "$D/registration.json" ]]
  args+=(--worker-preflight "$WP" --worker-preflight-sha256 "$WH" --wave-registration "$REG")
  if [[ "$PHASE" = science ]]; then
   [[ "$SCI" = - && "$REG_SHA" = - && "$BARRIER" = - && "$BARRIER_SHA" = - ]]
   ENTRY=phoenix_pilot_cb63305.py
  else
   [[ "$SCI" = "$D/members/$MID/science" && "$REG_SHA" =~ ^[0-9a-f]{64}$ ]]
   [[ "$BARRIER" = "$D/science-barrier.json" && "$BARRIER_SHA" =~ ^[0-9a-f]{64}$ ]]
   args+=(--case "$SCI" --wave-registration-sha256 "$REG_SHA"
     --wave-science-barrier "$BARRIER" --wave-science-barrier-sha256 "$BARRIER_SHA")
   ENTRY=runtime_audit_cb63305.py
  fi ;;
 *) exit 2 ;;
esac
export PATH=/usr/bin:/bin
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME BASH_ENV ENV
/usr/bin/mkdir -m 700 "$RUN/controller"
/usr/bin/mkdir -m 700 "$RUN/controller/home" "$RUN/controller/cache" "$RUN/controller/cwd"
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
bootstrap='import signal,sys
def stop(n,f): raise KeyboardInterrupt("QOMP wave signal "+str(n))
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
p=sys.argv.pop(1)
exec(compile(open(p,"rb").read(),p,"exec"),dict(__name__="__main__",__file__=p,__package__=None))'
cd "$RUN/controller/cwd"
/usr/bin/setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
 /usr/bin/env -i USER=awallace43 LOGNAME=awallace43 HOME="$RUN/controller/home" \
 PATH="$P/miniconda/envs/p4qca/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$P/miniconda/envs/p4qca/lib" \
 LANG=C.UTF-8 LC_ALL=C.UTF-8 SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
 SLURM_CLUSTER_NAME="$SLURM_CLUSTER_NAME" SLURMD_NODENAME="$SLURMD_NODENAME" \
 "$P/miniconda/envs/p4qca/bin/python" -I -B -X "pycache_prefix=$RUN/controller/cache" \
 -c "$bootstrap" "$SOURCE/devtools/isapol_s66_campaign/$ENTRY" "${args[@]}" &
child=$!
set +e
ccasp_wait_child "$child" 60
rc=$?
exit "$rc"
