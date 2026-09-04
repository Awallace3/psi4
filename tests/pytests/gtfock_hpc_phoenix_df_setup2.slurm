#!/bin/bash
# Second pass at the density-fitting setup split: close the two gaps job
# 12638902 left open.
#
#   sbatch --job-name=gtfdf-setup2 gtfock_hpc_phoenix_df_setup2.slurm
#
# Job 12638902 measured ``DFH: initialize()`` at all four documented sizes and
# came back with two rows that cannot be published as they stand.
#
# The peptide point ran 27.50 s against 3.3 s and 3.8 s in two earlier jobs,
# with its ``JK: JK`` timer at 0.62 s matching both to the digit -- the whole
# excess is outside the Fock build. It was the first process this job started on
# a freshly mounted scratch, and the nanotube point behind it came in at 17.95 s
# against 32.7 and 44.5 s earlier, faster rather than slower. That is the shape
# of one cold cache paid once by whichever point goes first, so here a throwaway
# peptide point goes first and its result is discarded. Both recorded points
# also drop to the 120 GB the earlier sweeps gave their reference arms, so the
# SCF totals are comparable with the tables they will sit beside rather than
# differing in two ways at once.
#
# At 1863 basis functions Psi4 chose ``DiskDFJK``, not ``MemDFJK``: the 160 GB
# budget that keeps the 1555-basis-function tensor in core (94.7 GB, confirmed)
# does not hold the 109.5 GiB one at 1863. ``DiskDFJK`` is a separate builder
# that does not go through ``DFHelper``, so it emits no ``DFH: initialize()``
# record at all and its setup is unattributed inside the SCF remainder. This
# retries that point at 185 GB of the node's 187 to see whether the in-core
# builder can be reached at all on this hardware. If it cannot, the answer is
# still worth having: it means the density-fitting figure this document quotes
# at 1863 basis functions is a disk algorithm, which is a fact about the
# comparison and not a defect in the run.
#
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --constraint=core24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

set -eo pipefail

FM_ROOT=${FM_ROOT:-/storage/scratch1/3/awallace43/fm-gtfock}
FM_ENV=${FM_ENV:-$FM_ROOT/envs/p4gtf}
FM_PSI4=${FM_PSI4:-$FM_ROOT/psi4}
FM_METHOD=${FM_METHOD:-scf}
CORES_PER_RANK=${SLURM_CPUS_PER_TASK:-24}

OUT=$FM_ROOT/results/dfsetup2_${SLURM_JOB_ID}
mkdir -p "$OUT"

module purge 2>/dev/null || true
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate "$FM_ENV"
eval $("$FM_PSI4/objdir_gtfock/stage/bin/psi4" --psiapi)

export OMP_PLACES=cores
export OMP_PROC_BIND=close

{
  echo "=== provenance ==="
  echo "job         : ${SLURM_JOB_ID} ${SLURM_JOB_NAME}"
  echo "batch host  : $(hostname)"
  echo "nodes       : ${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
  echo "partition   : ${SLURM_JOB_PARTITION}  account: ${SLURM_JOB_ACCOUNT}  qos: ${SLURM_JOB_QOS}"
  echo "psi4        : $(git -C "$FM_PSI4" log --oneline -1)"
  echo "tree        : $(git -C "$FM_PSI4" status --porcelain | wc -l) file(s) modified vs that commit"
  git -C "$FM_PSI4" diff --stat
  lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)):"
  free -g | head -2
} | tee "$OUT/provenance.txt"

FAILED_POINTS=()

# run_df <system> <basis> <memory-gb> <label>
run_df() {
    local system=$1 basis=$2 mem_gb=$3 label=$4
    local dest="$OUT/$label"
    mkdir -p "$dest"
    echo "=== ${label}: ${system} / ${basis} / df, 1 rank x ${CORES_PER_RANK} thread(s), ${mem_gb} GB ==="
    export OMP_NUM_THREADS=$CORES_PER_RANK MKL_NUM_THREADS=$CORES_PER_RANK
    local rc=0
    python "$FM_PSI4/tests/pytests/gtfock_hpc_benchmark.py" \
        --system "$system" --arm df --basis "$basis" \
        --method "$FM_METHOD" --threads "$CORES_PER_RANK" \
        --memory "${mem_gb} GB" --scratch "$dest/$label" \
        --json-out "$dest/$label" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "POINT FAILED: ${label} exited ${rc}; continuing the sweep." >&2
        FAILED_POINTS+=("${label}(rc=${rc})")
    fi
    return 0
}

# The warm-up. Its JSON lands under warmup/ and is never reduced; it exists so
# that the first-process cost is paid by a point nothing is claimed about.
run_df peptide '6-31+G**' 120 warmup

run_df peptide    '6-31+G**' 120 peptide_260
run_df nanotube   '6-31+G**' 120 nanotube_574
run_df protein157 '6-31+G**' 185 protein157_1863_mem185

echo "=== sacct ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,NNodes,NCPUS -P \
    | tee "$OUT/sacct.txt"
echo "results in $OUT"
echo "DF_SETUP2_DONE"
if [ ${#FAILED_POINTS[@]} -ne 0 ]; then
    echo "FAILED POINTS (${#FAILED_POINTS[@]}): ${FAILED_POINTS[*]}" >&2
    exit 1
fi
