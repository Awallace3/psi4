#!/usr/bin/env bash
# One attributable step on an already authorized inferno H200 reservation.
# Args: JOB_ID REPO COMMIT CAMPAIGN_ROOT PHASE [system ...]
set -euo pipefail
umask 027
job=$1 repo=$2 pin=$3 base=$4 phase=$5
shift 5
[[ "$job" =~ ^[0-9]+$ && "$repo" = /* && "$base" = /* ]]
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$phase" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$pin" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]
info=$(scontrol show job "$job" -o)
[[ "$info" == *"JobState=RUNNING"* && "$info" == *"QOS=inferno"* ]]
[[ "$info" == *"Account=gts-cs207-chemx"* && "$info" == *"gpu-h200"* ]]
run="$base/$phase"
[[ ! -e "$run" ]] || { echo "Refusing an existing phase: $run" >&2; exit 2; }
mkdir -p "$run"/{inputs/geometries,logs,metadata}
printf '%s\n' "$info" > "$run/metadata/allocation.txt"
printf '%s\n' "$pin" > "$run/metadata/harness-commit.txt"
for system in peptide nanotube protein83 protein157; do
    cp "$base/inputs/geometries/$system.xyz" "$run/inputs/geometries/"
done
export COMPARE_REPO="$repo" COMPARE_RUN="$run"
export COMPARE_CONFIG="${COMPARE_CONFIG:-$repo/devtools/benchmarks/gpu_compare/config-aug-cc-pvtz.json}"
export COMPARE_PSI_ENV="${COMPARE_PSI_ENV:-/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt}"
export COMPARE_GPU_ENV="${COMPARE_GPU_ENV:-/storage/project/r-cs207-0/awallace43/miniconda/envs/gpu4pyscf_compare}"
export COMPARE_PSI_PACKAGE="${COMPARE_PSI_PACKAGE:-/storage/project/r-cs207-0/awallace43/gits/psi4.saptdft_cuest/build_cuda_libxc/stage/lib/psi4}"
export COMPARE_PSI_CUDA_LIB="${COMPARE_PSI_CUDA_LIB:-/storage/project/r-cs207-0/awallace43/software/libxc-7.1.2-cuda/lib64}"
# Derived from the validated reservation above, not an assumed submit-time route.
export SLURM_JOB_QOS=inferno
child=
term_received=0
on_term() {
    term_received=1
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
}
trap on_term TERM INT
echo "Starting $phase on allocation $job; logs: $run/logs/step.log"
srun --jobid="$job" --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --kill-on-bad-exit=1 --export=ALL /usr/bin/env /bin/bash \
    "$repo/devtools/benchmarks/gpu_compare/payload.sh" campaign "$@" \
    > "$run/logs/step.log" 2>&1 &
child=$!
set +e
wait "$child"
status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
(( term_received == 0 )) || status=143
printf 'exit_code=%s\n' "$status" > "$run/metadata/step-exit.txt"
echo "Step $phase finished: rc=$status"
exit "$status"
