#!/usr/bin/env bash
# Args: absolute HARNESS_ROOT HARNESS_COMMIT RUN_ROOT
# Scientific builds/receipts remain pinned to the completed lifecycle smoke.
#SBATCH --job-name=cuest-protein157
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@120
set -euo pipefail
umask 027
harness=$1
harness_commit=$2
run=$3
root=/storage/project/r-cs207-0/awallace43
new="$root/gits/psi4.saptdft-cuest-lifecycle-20261003"
old="$root/gits/psi4.saptdft_cuest_head"
receipts="$root/runs/psi4-cuest-lifecycle/20261003T1015/metadata"
new_commit=532654b842a4a76abe76b2f588bc4cd10c1138e1
old_commit=9317f406b2b66f33b65866cbc44a4189f913339e
test "$(git -C "$harness" rev-parse HEAD)" = "$harness_commit"
test -z "$(git -C "$harness" diff HEAD -- devtools)"
test "$(git -C "$new" rev-parse HEAD)" = "$new_commit"
smoke_commit=$(head -n 1 "$receipts/SMOKE_COMMIT")
git -C "$new" diff --quiet "$smoke_commit" "$new_commit" -- psi4 tests cmake CMakeLists.txt external
set +u
source "$root/miniconda/etc/profile.d/conda.sh"
conda activate p4cuest_sapt
set -u
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
scratch_base="${TMPDIR:-/scratch}"
[[ -d "$scratch_base" ]] || scratch_base=/scratch
export TMPDIR="$scratch_base/psi4-protein157-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
mkdir -p "$run/metadata"
cd "$run"
eval "$("$new/build_lifecycle/stage/bin/psi4" --psiapi)"
export PYTHONPATH="$new/build_lifecycle/stage/lib"
metadata="$run/metadata/$SLURM_JOB_ID"
{
    printf 'harness_commit=%s\nscientific_commit=%s\njob=%s\nstarted=%s\n' \
        "$harness_commit" "$new_commit" "$SLURM_JOB_ID" "$(date -Iseconds)"
    scontrol show job "$SLURM_JOB_ID" -o
} > "$metadata.env"
term_received=0
child=
on_term() {
    term_received=1
    [[ -n $child ]] && kill -TERM "$child" 2>/dev/null || true
}
on_exit() {
    status=$?
    (( term_received )) && status=143
    printf 'exit_code=%s\nterm_received=%s\nfinished=%s\n' \
        "$status" "$term_received" "$(date -Iseconds)" >> "$metadata.env"
    if (( status == 0 )); then touch "$metadata.COMPLETED"; else touch "$metadata.FAILED"; fi
}
trap on_term TERM
trap on_exit EXIT
"$CONDA_PREFIX/bin/python" "$harness/devtools/benchmarks/lifecycle_campaign.py" \
    --output "$run/results-$SLURM_JOB_ID" --protein157 --repeats 1 \
    --threads 8 --memory "256 GiB" --case-timeout 10800 --require-in-core \
    --old-source "$old" --old-commit "$old_commit" \
    --old-package "$old/build_saptdft_cuest_head/stage/lib/psi4" \
    --new-source "$new" --new-commit "$new_commit" \
    --new-package "$new/build_lifecycle/stage/lib/psi4" \
    --old-receipt "$receipts/old-build.json" --new-receipt "$receipts/new-build.json" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
(( term_received )) && exit 143
exit "$status"
