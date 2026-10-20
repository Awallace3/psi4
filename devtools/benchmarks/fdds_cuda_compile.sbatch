#!/usr/bin/env bash
#SBATCH --job-name=fdds-cuda-compile-only
#SBATCH --account=gts-cs207
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=TERM@60

# Distinct source/build from the queued GPU validation; no GPU evidence claim.
set -euo pipefail
: "${FDDS_SOURCE_ROOT:?}" "${FDDS_SOURCE_PIN:?}"
: "${FDDS_PAYLOAD_ROOT:?}" "${FDDS_PAYLOAD_PIN:?}" "${FDDS_RUN_ROOT:?}"
for pair in "$FDDS_SOURCE_ROOT:$FDDS_SOURCE_PIN" "$FDDS_PAYLOAD_ROOT:$FDDS_PAYLOAD_PIN"; do
    tree=${pair%:*}
    pin=${pair##*:}
    test "$(git -C "$tree" rev-parse HEAD)" = "$pin"
    test -z "$(git -C "$tree" status --porcelain --untracked-files=no)"
done
out="$FDDS_RUN_ROOT/compile-$SLURM_JOB_ID"
mkdir "$out"
printf '%s\n' "$FDDS_PAYLOAD_PIN" > "$out/harness-commit.txt"
export CUEST_FDDS_COMPILE_ONLY=1 CUEST_FDDS_BINARY_PIN="$FDDS_SOURCE_PIN"
export TMPDIR="/storage/scratch1/3/$USER/fdds-cuda-compile/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
env=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt
child=
terminated=0
on_term() {
    terminated=1
    if [[ -n "$child" ]]; then kill -TERM "$child" || true; fi
}
trap on_term TERM INT
bash "$FDDS_PAYLOAD_ROOT/devtools/benchmarks/fdds_validate.sh" \
    "$FDDS_SOURCE_ROOT" "$env" "$FDDS_SOURCE_PIN" "$out" &
child=$!
rc=0
wait "$child" || rc=$?
child=
if ((terminated)); then exit 143; fi
if ((rc)); then exit "$rc"; fi
test -s "$out/BUILD_ONLY.txt"
test ! -e "$out/VERIFIED.json"
printf '%s\n' "$FDDS_SOURCE_PIN" > "$out/BUILD_ONLY_COMPLETED"
