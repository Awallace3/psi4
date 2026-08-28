#!/bin/bash
# Multi-node strong scaling for the GTFock J/K engine at production size, as
# actually submitted on Georgia Tech's Phoenix (PACE) cluster.
#
#   sbatch --job-name=gtfp-protein157 gtfock_hpc_phoenix_protein.slurm
#
# Same experiment as gtfock_hpc_phoenix_multinode.slurm -- one rank per whole
# 24-core node, 1 to 4 nodes -- on a system 3.2x larger: a 157-atom fragment
# pair from PDB 3acx, 1863 basis functions in 6-31+G**. The smaller systems
# answered "does the distribution work"; this one asks whether it still pays at
# a size where the J/K build is unambiguously the whole cost. At 1863 basis
# functions the four-rank point owns roughly 466 rows of the AO dimension each,
# so the partitioning has real work to hide the gather-and-broadcast behind.
#
# It is a separate script rather than another FM_SYSTEM value for the multinode
# job because that script is the committed record of jobs 12433617 and 12433686
# and is not edited after the fact. Two things genuinely differ here anyway: the
# walltime, and the memory given to the single-process reference arms.
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
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -eo pipefail

# Site paths. Everything lives in scratch; nothing is written into the repo.
FM_ROOT=${FM_ROOT:-/storage/scratch1/3/awallace43/fm-gtfock}
FM_ENV=${FM_ENV:-$FM_ROOT/envs/p4gtf}
FM_PSI4=${FM_PSI4:-$FM_ROOT/psi4}
FM_SYSTEM=${FM_SYSTEM:-protein157}
FM_BASIS=${FM_BASIS:-6-31+G**}
FM_METHOD=${FM_METHOD:-scf}
CORES_PER_RANK=${SLURM_CPUS_PER_TASK:-24}
MAX_RANKS=${SLURM_JOB_NUM_NODES:-4}

# Two memory budgets, and the asymmetry is deliberate rather than sloppy.
#
# A GTFock rank at this size measured under 1 GB resident on the smaller
# systems, so 120 GB is already orders of magnitude more headroom than it can
# use. The density-fitting reference is the opposite: at 1863 basis functions
# and 8462 auxiliary functions the packed (Q|mn) tensor is 109.5 GiB, which does
# not fit a 120 GB budget, and MemDFJK would silently become DiskDFJK -- a third
# algorithm, bottlenecked on scratch I/O, wearing the label of the second. The
# node has 186 GiB, so the reference arms get 160 GB: enough to keep density
# fitting in core and give Psi4's default its best showing on this hardware. The
# benchmark records the JK class it actually got, so if the fit fails anyway the
# results say DiskDFJK rather than quietly mislabelling the point.
MEMORY_PER_RANK_GB=${FM_MEMORY_PER_RANK_GB:-120}
REFERENCE_MEMORY_GB=${FM_REFERENCE_MEMORY_GB:-160}

OUT=$FM_ROOT/results/prot_${FM_SYSTEM}_${SLURM_JOB_ID}
mkdir -p "$OUT"

module purge 2>/dev/null || true
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate "$FM_ENV"
eval $("$FM_PSI4/objdir_gtfock/stage/bin/psi4" --psiapi)

# GTFock's GTMatrix layer does use MPI one-sided RMA (MPI_Win_create, MPI_Get,
# MPI_Accumulate, win lock/unlock), and OpenMPI's UCX osc component loses its
# own priority query on this cluster, logging a line per rank. Excluding it is
# expected to be cosmetic, and was measured to be so at 574 basis functions --
# but the RMA volume per Fock build grows with the square of the basis, so that
# measurement does not transfer for free. The control point at the end of this
# job re-measures the widest rank count with the exclusion lifted.
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
  echo "reference   : ${CORES_PER_RANK} cores, ${REFERENCE_MEMORY_GB} GB, single process"
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

# Prove cross-node launch in seconds rather than discovering it inside an SCF
# that costs hours at this size.
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
# reducer sees. gtfock_hpc_collect.py globs one directory level and keys a point
# on (system, arm, ranks), so a control point repeating an existing rank count
# has to live in a subdirectory or it would collide with the point it controls.
run_point() {
    local arm=$1 ranks=$2 dest=${3:-$OUT}
    local tag="${FM_SYSTEM}_${arm}_n${ranks}"
    local mem_gb=$MEMORY_PER_RANK_GB
    [ "$arm" = "gtfock" ] || mem_gb=$REFERENCE_MEMORY_GB
    mkdir -p "$dest"
    echo "=== ${tag}: ${ranks} rank(s) x ${CORES_PER_RANK} thread(s), ${mem_gb} GB each ==="
    local -a cmd=(python "$FM_PSI4/tests/pytests/gtfock_hpc_benchmark.py"
                  --system "$FM_SYSTEM" --arm "$arm" --basis "$FM_BASIS"
                  --method "$FM_METHOD" --threads "$CORES_PER_RANK"
                  --memory "${mem_gb} GB" --scratch "$dest/$tag"
                  --json-out "$dest/$tag")
    export OMP_NUM_THREADS=$CORES_PER_RANK MKL_NUM_THREADS=$CORES_PER_RANK
    if [ "$arm" = "gtfock" ]; then
        # A core set covering all 24 cores spans both sockets, which OpenMPI
        # refuses to bind, so every point runs unbound with OMP_PLACES=cores
        # spreading its threads.
        mpirun -n "$ranks" --map-by ppr:1:node --bind-to none --report-bindings "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

# Point order is chosen so that a walltime overrun costs the least interesting
# points. The GTFock sweep comes first, then density fitting -- Psi4's own
# default SCF_TYPE, and the comparison this run was asked for -- and the exact
# ERI baseline last, because at 1863 basis functions it is the single most
# expensive point in the job by a wide margin and the one whose loss would still
# leave a complete answer behind.
for ranks in $(seq 1 "$MAX_RANKS"); do
    run_point gtfock "$ranks"
done
run_point df     1    # density fitting: Psi4's default, a DIFFERENT algorithm
run_point direct 1    # exact ERIs on one node: the algorithmic apples-to-apples baseline

# Control: the widest GTFock point again with OMPI_MCA_osc unset, so the
# one-sided component OpenMPI picks on its own carries the cross-node RMA. If
# this lands on the primary point's wall clock, the exclusion above did not
# shape the result at this size either.
echo "=== control: n${MAX_RANKS} with OMPI_MCA_osc unset ==="
( unset OMPI_MCA_osc; run_point gtfock "$MAX_RANKS" "$OUT/control_osc_default" )

echo "=== sacct ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,NNodes,NCPUS -P \
    | tee "$OUT/sacct.txt"
echo "results in $OUT"
echo "SCALING_DONE"
