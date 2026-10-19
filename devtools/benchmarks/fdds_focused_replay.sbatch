#!/usr/bin/env bash
#SBATCH --job-name=cuest-fdds-focused-replay
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --signal=TERM@60
#SBATCH --output=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261019-integrated-focused-02/logs/%x-%j.out
#SBATCH --error=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261019-integrated-focused-02/logs/%x-%j.err
set -euo pipefail
: "${CUEST_FDDS_PAYLOAD_ROOT:?set separately pinned validation harness}"
root=/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-integrated-20261019-v2
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
out=/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261019-integrated-focused-02/results-$SLURM_JOB_ID
export CUEST_FDDS_PIN=3b21adb662f47fc4b89f1cbb89a44592f7a1d10f
export CUEST_FDDS_READONLY_BUILD_ROOT=/storage/project/r-cs207-0/awallace43/gits/psi4.fdds-integrated-20261017-v1
export CUEST_FDDS_BINARY_PIN=9d4b1cd4aab718dec1f7b831a53dd54d68f2a609
export CUEST_FDDS_EXPECTED_CORE_SHA256=68e34452afa80002d4bb7fb1b56297e19389617fcb3ee3b926abc0b5fcead0b7
export CUEST_FDDS_FOCUSED=1
mkdir -p "$out"
git -C "$CUEST_FDDS_PAYLOAD_ROOT" rev-parse HEAD > "$out/harness-commit.txt"
test -z "$(git -C "$CUEST_FDDS_PAYLOAD_ROOT" status --porcelain --untracked-files=no)"
child=
trap 'trap - TERM INT; if [[ -n "$child" ]]; then kill -TERM "$child" || true; wait "$child" || true; fi; exit 143' TERM INT
srun bash "$CUEST_FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_validate.sh" "$root" "$env" "$CUEST_FDDS_PIN" "$out" &
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
