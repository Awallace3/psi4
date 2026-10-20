#!/usr/bin/env bash
#SBATCH --job-name=fdds-opt-gpu-xy-v1
#SBATCH --account=gts-cs207
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@60

# Own immutable source/build + distinct immutable harness, shared deps read-only.
set -euo pipefail
: "${FDDS_SOURCE_ROOT:?}" "${FDDS_SOURCE_PIN:?}"
: "${FDDS_PAYLOAD_ROOT:?}" "${FDDS_PAYLOAD_PIN:?}"
: "${FDDS_RUN_ROOT:?}" "${FDDS_CACHE_ROOT:?}"
for pair in "$FDDS_SOURCE_ROOT:$FDDS_SOURCE_PIN" "$FDDS_PAYLOAD_ROOT:$FDDS_PAYLOAD_PIN"; do
    tree=${pair%:*}
    pin=${pair##*:}
    test "$(git -C "$tree" rev-parse HEAD)" = "$pin"
    test -z "$(git -C "$tree" status --porcelain --untracked-files=no)"
done
for system in peptide nanotube protein83; do
    test -s "$FDDS_CACHE_ROOT/$system/metadata.json"
    test -s "$FDDS_CACHE_ROOT/$system/orbitals.npz"
done
out="$FDDS_RUN_ROOT/validation-$SLURM_JOB_ID"
mkdir "$out"
printf '%s\n' "$FDDS_PAYLOAD_PIN" > "$out/harness-commit.txt"
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
child=
terminated=0
on_term() {
    terminated=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
trap on_term TERM INT
export CUEST_FDDS_FOCUSED=1 CUEST_FDDS_FULL_AFTER_FOCUSED=1
bash "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_validate.sh" \
    "$FDDS_SOURCE_ROOT" "$env" "$FDDS_SOURCE_PIN" "$out" &
child=$!
rc=0
wait "$child" || rc=$?
child=
if ((terminated)); then exit 143; fi
if ((rc)); then exit "$rc"; fi
test -s "$out/VERIFIED.json"
printf '%s\n' "$FDDS_SOURCE_PIN" > "$out/COMPLETED"

# The full scientific gate is complete before any speed claim or large replay.
# Run matched fresh CPU/GPU controls using this very same newly built stage.
export FDDS_STAGE_ROOT="$FDDS_SOURCE_ROOT" FDDS_BINARY_PIN="$FDDS_SOURCE_PIN"
FDDS_CORE_SHA256=$(sha256sum "$FDDS_SOURCE_ROOT"/build_fdds_validation/stage/lib/psi4/core*.so)
export FDDS_CORE_SHA256="${FDDS_CORE_SHA256%% *}" FDDS_GPU_EXCHANGE_REQUIRED=1
unset FDDS_CPU_JOB_ID
bash "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_gpu_profile.sbatch" &
child=$!
rc=0
wait "$child" || rc=$?
child=
if ((terminated)); then exit 143; fi
if ((rc)); then exit "$rc"; fi
test -s "$FDDS_RUN_ROOT/matched-$SLURM_JOB_ID/COMPLETED"
printf '%s\n' "$FDDS_SOURCE_PIN" > "$FDDS_RUN_ROOT/optimization-$SLURM_JOB_ID.COMPLETED"
