#!/bin/bash
# Multi-node strong scaling for the GTFock J/K engine, as actually submitted on
# Georgia Tech's Phoenix (PACE) cluster.
#
#   sbatch --job-name=gtfmn-peptide  --export=ALL,FM_SYSTEM=peptide  gtfock_hpc_phoenix_multinode.slurm
#   sbatch --job-name=gtfmn-nanotube --export=ALL,FM_SYSTEM=nanotube gtfock_hpc_phoenix_multinode.slurm
#
# This is the companion to gtfock_hpc_phoenix.slurm and asks the opposite
# question. That job held the total core count fixed at one 24-core node and
# only changed how the ranks carved it up, which is the right control for "does
# adding ranks cost anything" but cannot show a distributed speedup: no rank
# ever had a core that a lower-rank point did not already have. Here the cores
# per rank are what is held fixed -- one rank owns one whole 24-core node -- so
# the rank count and the hardware grow together, 24 -> 48 -> 72 -> 96 cores.
#
# All four GTFock points come out of a single 4-node allocation, running on 1,
# 2, 3 and 4 of its nodes. One allocation means the 4-node point cannot be
# faster merely because it landed on newer silicon or a quieter switch than the
# 1-node point, and it makes SLURM_JOB_NODELIST identical across the points, so
# gtfock_hpc_collect.py sees one run identity and reduces them together.
#
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --constraint=core24
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

set -eo pipefail

# Site paths. Everything lives in scratch; nothing is written into the repo.
FM_ROOT=${FM_ROOT:-/storage/scratch1/3/awallace43/fm-gtfock}
FM_ENV=${FM_ENV:-$FM_ROOT/envs/p4gtf}
FM_PSI4=${FM_PSI4:-$FM_ROOT/psi4}
FM_SYSTEM=${FM_SYSTEM:?set FM_SYSTEM to peptide or nanotube}
FM_BASIS=${FM_BASIS:-6-31+G**}
FM_METHOD=${FM_METHOD:-scf}
# Per-rank cores and per-rank memory are what is held constant here, because
# each rank owns a whole node. The rank count therefore sets the total.
CORES_PER_RANK=${SLURM_CPUS_PER_TASK:-24}
MEMORY_PER_RANK_GB=${FM_MEMORY_PER_RANK_GB:-120}
MAX_RANKS=${SLURM_JOB_NUM_NODES:-4}

OUT=$FM_ROOT/results/mn_${FM_SYSTEM}_${SLURM_JOB_ID}
mkdir -p "$OUT"

module purge 2>/dev/null || true
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate "$FM_ENV"
eval $("$FM_PSI4/objdir_gtfock/stage/bin/psi4" --psiapi)

# GTFock's GTMatrix layer *does* use MPI one-sided RMA -- MPI_Win_create,
# MPI_Get, MPI_Accumulate and win lock/unlock, in GTMatrix_Get.c and
# GTMatrix_Update.c -- so the choice of one-sided component is not irrelevant
# here, and the fixed-core job's comment claiming GTFock has no RMA is simply
# wrong. What is true is that OpenMPI's UCX osc component loses its own priority
# query on this cluster and falls back, so excluding it should only silence a
# per-rank log line. Intra-node that was safe to assume; across a network it is
# an assumption about the component carrying the traffic, so the control point at
# the end of this job re-measures the largest rank count with the exclusion
# lifted instead of leaving the claim asserted.
export OMPI_MCA_osc=^ucx
export OMP_PLACES=cores
export OMP_PROC_BIND=close

{
  echo "=== provenance ==="
  echo "job         : ${SLURM_JOB_ID} ${SLURM_JOB_NAME}"
  echo "batch host  : $(hostname)"
  echo "nodes       : ${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
  echo "partition   : ${SLURM_JOB_PARTITION}  account: ${SLURM_JOB_ACCOUNT}  qos: ${SLURM_JOB_QOS}"
  echo "per rank    : ${CORES_PER_RANK} cores, ${MEMORY_PER_RANK_GB} GB; one rank per node"
  echo "system      : ${FM_SYSTEM} / ${FM_BASIS} / ${FM_METHOD}"
  echo "psi4        : $(git -C "$FM_PSI4" log --oneline -1)"
  echo "mpirun      : $(which mpirun) -- $(mpirun --version | head -1)"
  lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)):"
  free -g | head -2
  for host in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
      scontrol show node "$host" | tr ' ' '\n' \
          | grep -E "^(NodeName|CPUTot|RealMemory|AvailableFeatures)=" | paste -sd' '
  done
} | tee "$OUT/provenance.txt"

# Cross-node launch is the one thing this job needs that the single-node job did
# not, so prove it in seconds rather than discovering it inside a paid SCF.
echo "=== mpi placement check ==="
mpirun -n "$MAX_RANKS" --map-by ppr:1:node --bind-to none \
    bash -c 'echo "rank ${OMPI_COMM_WORLD_RANK}/${OMPI_COMM_WORLD_SIZE} on $(hostname)"' \
    | sort | tee "$OUT/placement.txt"
distinct_hosts=$(awk '{print $NF}' "$OUT/placement.txt" | sort -u | wc -l)
if [ "$distinct_hosts" -ne "$MAX_RANKS" ]; then
    echo "FATAL: ${MAX_RANKS} ranks landed on ${distinct_hosts} distinct hosts;" \
         "this job would measure oversubscription, not distribution." >&2
    exit 1
fi

# run_point <arm> <ranks> [destination-directory]
#
# The destination defaults to $OUT, whose *.rank*.json files are the sweep the
# reducer sees. A point written somewhere else is deliberately invisible to it:
# gtfock_hpc_collect.py globs one directory level and keys a point on
# (system, arm, ranks), so a control point that repeats an existing rank count
# has to live in a subdirectory or it would collide with the point it controls.
run_point() {
    local arm=$1 ranks=$2 dest=${3:-$OUT}
    local tag="${FM_SYSTEM}_${arm}_n${ranks}"
    mkdir -p "$dest"
    echo "=== ${tag}: ${ranks} rank(s) x ${CORES_PER_RANK} thread(s), ${MEMORY_PER_RANK_GB} GB each ==="
    local -a cmd=(python "$FM_PSI4/tests/pytests/gtfock_hpc_benchmark.py"
                  --system "$FM_SYSTEM" --arm "$arm" --basis "$FM_BASIS"
                  --method "$FM_METHOD" --threads "$CORES_PER_RANK"
                  --memory "${MEMORY_PER_RANK_GB} GB" --scratch "$dest/$tag"
                  --json-out "$dest/$tag")
    export OMP_NUM_THREADS=$CORES_PER_RANK MKL_NUM_THREADS=$CORES_PER_RANK
    if [ "$arm" = "gtfock" ]; then
        # Every rank gets a whole node, so there is nothing to partition and
        # nothing to bind: a core set covering all 24 cores spans both sockets,
        # which OpenMPI refuses to bind ("would cause the process to be bound to
        # CPUs in more than one package"). Every point in this sweep therefore
        # runs unbound with OMP_PLACES=cores spreading its threads, including
        # the 1-rank point -- unlike the fixed-core job, where the rank counts
        # were placed differently from each other out of necessity.
        mpirun -n "$ranks" --map-by ppr:1:node --bind-to none --report-bindings "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

# The GTFock points first, cheapest to most expensive in rank count. They are
# what this job exists to measure; the reference arms are single-process by
# construction, cannot use a second node, and are already measured in the
# fixed-core job, so they go last where a wall-clock overrun would cost least.
for ranks in $(seq 1 "$MAX_RANKS"); do
    run_point gtfock "$ranks"
done
run_point direct 1    # exact ERIs on one node: the algorithmic apples-to-apples baseline
run_point df     1    # density fitting: a DIFFERENT algorithm, recorded for context

# Control: the widest GTFock point again with OMPI_MCA_osc unset, so the one-sided
# component OpenMPI picks on its own carries the cross-node RMA. If this lands on
# the primary point's wall clock, the exclusion above did not shape the result.
echo "=== control: n${MAX_RANKS} with OMPI_MCA_osc unset ==="
( unset OMPI_MCA_osc; run_point gtfock "$MAX_RANKS" "$OUT/control_osc_default" )

echo "=== sacct ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,NNodes,NCPUS -P \
    | tee "$OUT/sacct.txt"
echo "results in $OUT"
echo "SCALING_DONE"
