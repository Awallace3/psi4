#!/bin/bash
#SBATCH --job-name=cuest0301-install
#SBATCH --account=gts-cs207-chemx
#SBATCH --partition=cpu-small
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --signal=TERM@60
# Submit with absolute --output/--error paths in an existing directory.
# Arguments: frozen source worktree, exact commit, fresh SDK prefix, output root.
set -euo pipefail
source_root=$1
pin=$2
prefix=$3
output=$4
[[ $(git -C "$source_root" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$source_root" status --porcelain --untracked-files=no) ]]
mkdir -p "$output"
output="$output/job-${SLURM_JOB_ID}"
mkdir "$output"
term_received=0
child=
on_term() {
    term_received=1
    [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
}
trap on_term TERM INT
export PYTHONNOUSERSITE=1
conda_root=/storage/project/r-cs207-0/awallace43/miniconda
srun "$conda_root/bin/python" "$source_root/devtools/disp0/install_cuest0301.py" \
    --conda "$conda_root/bin/conda" --prefix "$prefix" --output "$output/package" &
child=$!
set +e
wait "$child"; status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
set -e
(( term_received )) && status=143
printf 'exit_code=%s\nterm_received=%s\nsource=%s\npin=%s\n' \
    "$status" "$term_received" "$source_root" "$pin" > "$output/status.txt"
exit "$status"
