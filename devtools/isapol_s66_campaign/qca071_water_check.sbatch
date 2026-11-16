#!/bin/bash
# Two original water-dimer fragments, direct and real QCA 0.71 delivery.
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --signal=TERM@180
#SBATCH --no-requeue
set -euo pipefail
umask 077
[[ $# -eq 1 && "$1" =~ ^[0-9a-f]{40}$ ]]
P=/storage/project/r-cs207-0/awallace43
SOURCE="$P/gits/psi4.ccasp-native-qomp-production148-v4"
RUN="$P/runs/ccasp-native-s66-qca071-water-check001"
MANAGER="$P/miniconda/envs/p4qca"
export PATH="$MANAGER/bin:/opt/slurm/current/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$MANAGER/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD BASH_ENV ENV
[[ -d "$RUN/logs" && ! -e "$RUN/payload" ]]
source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"
setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none \
  "$MANAGER/bin/python" -I -B \
  "$SOURCE/devtools/isapol_s66_campaign/qca071_water_check.py" \
  --commit "$1" --run "$RUN" &
child=$!
set +e
ccasp_wait_child "$child" 60
rc=$?
exit "$rc"
