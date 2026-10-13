#!/usr/bin/env bash
#SBATCH --job-name=cuest-fdds-diagnosis-v1
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261013-diagnosis-01/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261013-diagnosis-01/logs/%x-%j.err
set -euo pipefail
science=/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-validation-20261012-v3
driver=/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-diagnosis-20261013-v1
out=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261013-diagnosis-01/results-$SLURM_JOB_ID
mkdir -p "$out"
child=
trap 'trap - TERM INT; if [[ -n "$child" ]]; then kill -TERM "$child" || true; wait "$child" || true; fi; exit 143' TERM INT
srun bash "$driver/devtools/benchmarks/fdds_diagnose.sh" "$science" "$driver" "$out" &
child=$!
set +e
wait "$child"
status=$?
set -e
child=
if [[ "$status" == 0 && -s "$out/DIAGNOSED.json" ]]; then
    printf '%s\n' "diagnostic execution completed, not a correctness pass" > "$out/COMPLETED"
fi
exit "$status"
