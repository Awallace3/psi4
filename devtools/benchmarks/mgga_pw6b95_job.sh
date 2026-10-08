#!/usr/bin/env bash
# Args: absolute harness path, pinned commit, unique run directory.
#SBATCH --job-name=mgga-pw6b95-cpu
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --signal=TERM@120
set -euo pipefail
umask 027
harness=$1
commit=$2
run=$3
root=/storage/project/r-cs207-0/awallace43
test "$(git -C "$harness" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$harness" status --porcelain --untracked-files=no)"
env="$root/miniconda/envs/p4cuest_sapt"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
scratch_harness="$root/gits/psi4.saptdft-cuest-protein157-20261004"
test "$(git -C "$scratch_harness" rev-parse HEAD)" = 7d897a7f203297754087ea77536619c378191c8b
TMPDIR=$("$env/bin/python" "$scratch_harness/devtools/benchmarks/node_scratch.py")
export TMPDIR SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR"
package="$root/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage/lib/psi4"
mkdir -p "$run/metadata"
cd "$run"
{
    date -Iseconds
    scontrol show job "$SLURM_JOB_ID" -o
    printf 'package=%s\n' "$package"
    sha256sum "$package"/core*.so
    ldd "$package"/core*.so
} > "$run/metadata/environment-$SLURM_JOB_ID.txt"
child=
terminated=0
trap 'terminated=1; [[ -z $child ]] || kill -TERM "$child" 2>/dev/null || true' TERM
srun /usr/bin/env TMPDIR="$TMPDIR" SCRATCH="$TMPDIR" PSI_SCRATCH="$TMPDIR" \
    "$env/bin/python" "$harness/devtools/benchmarks/mgga_pw6b95.py" \
    --output "$run/results-$SLURM_JOB_ID" --package "$package" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
(( terminated )) && status=143
printf 'exit=%s terminated=%s\n' "$status" "$terminated" >> "$run/metadata/environment-$SLURM_JOB_ID.txt"
exit "$status"
