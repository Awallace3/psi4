#!/usr/bin/env bash
#SBATCH --job-name=cuest-fdds-validation
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-02/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-02/logs/%x-%j.err
set -euo pipefail
: "${CUEST_FDDS_PIN:?set immutable FDDS validation commit}"
root=/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-validation-20261012-v2
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
out=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261012-validation-02/results-$SLURM_JOB_ID
mkdir -p "$out"
child=
trap 'trap - TERM INT; if [[ -n "$child" ]]; then kill -TERM "$child" || true; wait "$child" || true; fi; exit 143' TERM INT
srun bash "$root/devtools/benchmarks/fdds_validate.sh" "$root" "$env" "$CUEST_FDDS_PIN" "$out" &
child=$!
set +e
wait "$child"
status=$?
set -e
child=
if [[ "$status" == 0 && -s "$out/VERIFIED.json" ]]; then
    printf '%s\n' "$CUEST_FDDS_PIN" > "$out/COMPLETED"
fi
exit "$status"
