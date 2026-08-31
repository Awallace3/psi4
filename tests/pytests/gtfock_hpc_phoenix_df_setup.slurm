#!/bin/bash
# Measure what the ``JK: JK`` timer does not see in a density-fitting SCF.
#
#   sbatch --job-name=gtfdf-setup gtfock_hpc_phoenix_df_setup.slurm
#
# Every table in gtfock.rst puts a df ``jk_wall`` beside a direct or gtfock
# ``jk_wall`` and reads the quotient as a speed ratio. That quotient is not one.
# ``JK::initialize()`` (libfock/jk.cc:593) calls the subclass ``preiterations()``
# outside ``timer_on("JK: JK")`` (jk.cc:650); for ``MemDFJK`` that call is
# ``dfh_->initialize()`` with method STORE (libfock/MemDFJK.cc:71), which builds
# and keeps the entire three-index tensor. ``DirectJK::preiterations()`` does
# nothing comparable. So on the exact-ERI arms ``JK: JK`` covers all the integral
# work, and on the df arm it covers only the GEMM half -- and the df column is
# flattered by however long the tensor took to build.
#
# The size of that omission is already recoverable: ``DFH: initialize()`` has
# been a timer since lib3index/dfhelper.cc:153. Nothing in C++ changes and
# nothing is rebuilt; the benchmark script now reads that record and reports the
# SCF remainder alongside it, and this job re-runs the df arm at every size the
# documentation quotes so the three-way split can be stated from measurement.
#
# Four points, one process each, no MPI: this arm never had any. They are the
# four df rows in the docs -- peptide 260 bf, nanotube 574 bf, protein157 in
# 6-31G** at 1555 bf, and protein157 in 6-31+G** at 1863 bf. Previously measured
# SCF wall clocks were 3.8, 44.5, 393.5 and 829 s, so the sweep is roughly 21
# minutes of compute; the hour requested is for the node's own variance and the
# 109.5 GiB tensor's allocation, not for the arithmetic.
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

# One budget, the reference one. At 1863 basis functions and 8462 auxiliary
# functions the packed (Q|mn) tensor is 109.5 GiB; under a smaller budget
# MemDFJK silently becomes DiskDFJK, which would measure scratch I/O and label
# it density fitting. The node has 186 GiB. The benchmark records the JK class
# it actually received, so a fit that fails anyway says DiskDFJK in the JSON
# rather than quietly mislabelling the point.
REFERENCE_MEMORY_GB=${FM_REFERENCE_MEMORY_GB:-160}

OUT=$FM_ROOT/results/dfsetup_${SLURM_JOB_ID}
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
  echo "reference   : ${CORES_PER_RANK} cores, ${REFERENCE_MEMORY_GB} GB, single process"
  echo "psi4        : $(git -C "$FM_PSI4" log --oneline -1)"
  # A commit SHA alone is a lie about a checkout patched in place, and this one
  # is staged by copy rather than by pull, so say so and show what moved.
  echo "tree        : $(git -C "$FM_PSI4" status --porcelain | wc -l) file(s) modified vs that commit"
  git -C "$FM_PSI4" diff --stat
  lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)):"
  free -g | head -2
} | tee "$OUT/provenance.txt"

FAILED_POINTS=()

# run_df <system> <basis> <label>
#
# Each point gets its own directory. gtfock_hpc_collect.py keys a point on
# (system, arm, ranks) and basis is not part of that key, so the two protein157
# points -- same system, same arm, one rank each, different basis -- would
# collide into a single group with two rank-0 records and be rejected as a mixed
# run. Separate directories keep them separate, and each is reduced on its own.
run_df() {
    local system=$1 basis=$2 label=$3
    local dest="$OUT/$label"
    mkdir -p "$dest"
    echo "=== ${label}: ${system} / ${basis} / df, 1 rank x ${CORES_PER_RANK} thread(s), ${REFERENCE_MEMORY_GB} GB ==="
    export OMP_NUM_THREADS=$CORES_PER_RANK MKL_NUM_THREADS=$CORES_PER_RANK
    local rc=0
    python "$FM_PSI4/tests/pytests/gtfock_hpc_benchmark.py" \
        --system "$system" --arm df --basis "$basis" \
        --method "$FM_METHOD" --threads "$CORES_PER_RANK" \
        --memory "${REFERENCE_MEMORY_GB} GB" --scratch "$dest/$label" \
        --json-out "$dest/$label" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "POINT FAILED: ${label} exited ${rc}; continuing the sweep." >&2
        FAILED_POINTS+=("${label}(rc=${rc})")
    fi
    return 0
}

# Cheapest first. Under a preemption on embers the points already written stay
# written, and the order means an interrupted job still leaves the small sizes
# behind rather than nothing.
run_df peptide    '6-31+G**' peptide_260
run_df nanotube   '6-31+G**' nanotube_574
run_df protein157 '6-31G**'  protein157_1555
run_df protein157 '6-31+G**' protein157_1863

echo "=== sacct ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,NNodes,NCPUS -P \
    | tee "$OUT/sacct.txt"
echo "results in $OUT"
echo "DF_SETUP_DONE"
if [ ${#FAILED_POINTS[@]} -ne 0 ]; then
    echo "FAILED POINTS (${#FAILED_POINTS[@]}): ${FAILED_POINTS[*]}" >&2
    exit 1
fi
