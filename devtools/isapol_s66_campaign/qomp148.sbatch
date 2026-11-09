#!/bin/bash
# Resources are the exact phase_command(case) arguments, checked again in Python.
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
[[ $# -eq 5 || $# -eq 7 || $# -eq 8 ]] || exit 2
SOURCE=$1 COMMIT=$2 PHASE=$3 RUN=$4 PILOT=$5
[[ "$SOURCE" = /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-qomp-production148-v2 ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$RUN" = /storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-qomp148-"${PHASE}002" ]]
[[ -d "$RUN/logs" && ! -e "$RUN/controller" && ! -e "$RUN/payload" ]]
case "$PHASE" in
  worker) [[ $# -eq 5 ]]; ENTRY=phoenix_worker_cb63305.py ;;
  science) [[ $# -eq 7 ]]; ENTRY=phoenix_pilot_cb63305.py ;;
  audit) [[ $# -eq 8 ]]; ENTRY=runtime_audit_cb63305.py ;;
  *) exit 2 ;;
esac
P=/storage/project/r-cs207-0/awallace43
export PATH=/usr/bin:/bin
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME BASH_ENV ENV
/usr/bin/mkdir -m 700 "$RUN/controller"
/usr/bin/mkdir -m 700 "$RUN/controller/home" "$RUN/controller/cache" "$RUN/controller/cwd"
export HOME="$RUN/controller/home"
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
args=(--build "$P/runs/ccasp-native-s66-qomp-fb11-build002"
  --qualification "$P/runs/ccasp-native-s66-qomp-runtime146-check001/inputs/receipts/qualification/installed-validation.json"
  --inputs "$P/runs/ccasp-native-s66-qomp-profile146-preparation001"
  --descriptors "$P/runs/ccasp-native-s66-qomp-profile146-descriptors001/payload"
  --pilot "$PILOT" --run "$RUN" --commit "$COMMIT" --producer qomp --campaign-scope qomp-pilot)
if [[ "$PHASE" != worker ]]; then
  [[ "$7" =~ ^[0-9a-f]{64}$ ]]
  args+=(--worker-preflight "$6" --worker-preflight-sha256 "$7")
fi
if [[ "$PHASE" = audit ]]; then args+=(--case "$8"); fi
bootstrap='import signal,sys
def stop(n,f): raise KeyboardInterrupt("QOMP phase signal "+str(n))
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
