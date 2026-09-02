#!/bin/bash
# Multi-node scaling for GTFock's *distributed density fitting*, as submitted on
# Georgia Tech's Phoenix (PACE) cluster.
#
#   sbatch --job-name=gtfdfmn-peptide  --export=ALL,FM_SYSTEM=peptide  gtfock_hpc_phoenix_df_multinode.slurm
#   sbatch --job-name=gtfdfmn-nanotube --export=ALL,FM_SYSTEM=nanotube gtfock_hpc_phoenix_df_multinode.slurm
#
# The exact path's multi-node job (gtfock_hpc_phoenix_multinode.slurm) asks how
# the J/K build scales with nodes. The DF path has a second question the exact
# path does not: its setup ends in an all-to-all that moves the entire fitted
# tensor onto the auxiliary axis, so every rank sends most of its slice to every
# other rank exactly once. On one node that is a memcpy through shared memory
# and the local sweep measured it at a fifth of a second. Across a network it is
# real traffic, and nothing measured so far says how much.
#
# Two sweeps, therefore:
#
#   1. Strong scaling, one rank per node, 24 cores each. Rank count and hardware
#      grow together, 24 -> 96 cores, and the redist phase clock grows with the
#      number of peers each rank must exchange with.
#
#   2. A locality control at fixed rank count and fixed cores per rank: four
#      ranks of six cores packed onto one node, then the same four ranks of six
#      cores spread one per node. Same partition, same tensor, same arithmetic;
#      the only difference is whether the all-to-all crosses the fabric. The
#      difference between their redist clocks is the network cost of the setup,
#      with nothing else varying to explain it away.
#
# All of it comes out of a single 4-node allocation so no point can be faster
# merely for having landed on quieter hardware than another.
#
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --constraint=core24
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

set -eo pipefail

FM_ROOT=${FM_ROOT:-/storage/scratch1/3/awallace43/fm-gtfock}
FM_ENV=${FM_ENV:-$FM_ROOT/envs/p4gtf}
FM_PSI4=${FM_PSI4:-$FM_ROOT/psi4}
FM_SYSTEM=${FM_SYSTEM:?set FM_SYSTEM to peptide or nanotube}
FM_BASIS=${FM_BASIS:-6-31+G**}
FM_METHOD=${FM_METHOD:-scf}
# Cores per node, not per task: the allocation is deliberately shaped as 24
# single-CPU tasks so that OpenMPI has 24 slots per node to place ranks in, so
# SLURM_CPUS_PER_TASK is 1 here and says nothing about the hardware.
CORES_PER_RANK=${SLURM_CPUS_ON_NODE:-24}
MEMORY_PER_RANK_GB=${FM_MEMORY_PER_RANK_GB:-120}
MAX_RANKS=${SLURM_JOB_NUM_NODES:-4}

OUT=$FM_ROOT/results/dfmn_${FM_SYSTEM}_${SLURM_JOB_ID}
mkdir -p "$OUT"

module purge 2>/dev/null || true
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate "$FM_ENV"
eval $("$FM_PSI4/objdir_gtfock/stage/bin/psi4" --psiapi)

# The DF engine's collectives are all two-sided -- Sendrecv for the
# redistribution, Allreduce for the metric and for J and K -- so unlike the
# exact path's GTMatrix layer it uses no one-sided RMA and the choice of osc
# component cannot shape its numbers. The exclusion stays only to silence
# OpenMPI's per-rank UCX priority-query line, and this job runs no osc control
# because there is nothing here for one to control.
export OMPI_MCA_osc=^ucx
export OMP_PLACES=cores
export OMP_PROC_BIND=close

{
  echo "=== provenance ==="
  echo "job         : ${SLURM_JOB_ID} ${SLURM_JOB_NAME}"
  echo "batch host  : $(hostname)"
  echo "nodes       : ${SLURM_JOB_NUM_NODES} (${SLURM_JOB_NODELIST})"
  echo "partition   : ${SLURM_JOB_PARTITION}  account: ${SLURM_JOB_ACCOUNT}  qos: ${SLURM_JOB_QOS}"
  echo "per rank    : ${CORES_PER_RANK} cores, ${MEMORY_PER_RANK_GB} GB in the node-per-rank sweep"
  echo "slots       : ${SLURM_NTASKS_PER_NODE} per node (mapping budget, not the thread count)"
  echo "system      : ${FM_SYSTEM} / ${FM_BASIS} / ${FM_METHOD}"
  echo "psi4        : $(git -C "$FM_PSI4" log --oneline -1)"
  echo "gtfock      : $(git -C "$FM_ROOT/gtfock_psi4" log --oneline -1)"
  echo "mpirun      : $(which mpirun) -- $(mpirun --version | head -1)"
  lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)):"
  free -g | head -2
  for host in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
      scontrol show node "$host" | tr ' ' '\n' \
          | grep -E "^(NodeName|CPUTot|RealMemory|AvailableFeatures)=" | paste -sd' '
  done
} | tee "$OUT/provenance.txt"

echo "=== psi4 build check ==="
python -c "
import psi4, sys
if not psi4.core.gtfock_df_enabled():
    sys.exit('FATAL: this build has no GTFock DF engine; run stageD_df_build.slurm first')
print('gtfock', psi4.core.gtfock_enabled(), 'gtfock_df', psi4.core.gtfock_df_enabled())
"

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

# run_point <arm> <ranks> <threads> <map-by> <bind-to> [destination-directory]
#
# Placement is expressed entirely through --map-by, which works because the
# allocation shape gives OpenMPI 24 slots per node to place into.
#
# The destination defaults to $OUT, whose *.rank*.json files are the sweep the
# reducer sees. gtfock_hpc_collect.py globs one directory level and keys a point
# on (system, arm, ranks), so the locality control -- which repeats a rank count
# already in the sweep -- has to live in a subdirectory or it would collide with
# the point it is controlling.
run_point() {
    local arm=$1 ranks=$2 threads=$3 mapby=$4 bind=$5 dest=${6:-$OUT}
    local tag="${FM_SYSTEM}_${arm}_n${ranks}"
    # Per-rank budget, overridable because the packed control puts four ranks on
    # one node and four times the node-per-rank budget is more memory than the
    # node has. Psi4 sizes its own allocations from this, so handing every rank
    # the whole node would let four of them each plan to use all of it.
    local mem_gb=${FM_MEM_OVERRIDE_GB:-$MEMORY_PER_RANK_GB}
    mkdir -p "$dest"
    echo "=== ${tag}: ${ranks} rank(s) x ${threads} thread(s), ${mem_gb} GB each," \
         "map-by ${mapby}, bind-to ${bind} ==="
    local -a cmd=(python "$FM_PSI4/tests/pytests/gtfock_hpc_benchmark.py"
                  --system "$FM_SYSTEM" --arm "$arm" --basis "$FM_BASIS"
                  --method "$FM_METHOD" --threads "$threads"
                  --memory "${mem_gb} GB" --scratch "$dest/$tag"
                  --json-out "$dest/$tag")
    export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
    case "$arm" in
        gtfock|gtfock_df)
            mpirun -n "$ranks" --map-by "$mapby" --bind-to "$bind" \
                   --report-bindings "${cmd[@]}"
            ;;
        *)
            "${cmd[@]}"
            ;;
    esac
}

# assert_hosts <destination> <tag> <expected distinct hosts>
#
# The sweep gets its placement checked before it runs, but the locality control
# is the point where placement *is* the measurement, and an unchecked one is
# worse than none: an earlier run of this job reported a packed/spread pair that
# had in fact both run spread, because --map-by ppr:4:node cannot pack when the
# allocation grants one slot per node. The two rows differed by ordering noise
# and read like a network effect. Check after the fact, from the records
# themselves, rather than trusting the mapping request.
assert_hosts() {
    local dest=$1 tag=$2 want=$3
    local got
    got=$(python - "$dest"/"$tag".rank*.json <<'PY'
import json, sys
hosts = {json.load(open(f))["host"] for f in sys.argv[1:]}
print(len(hosts))
PY
)
    if [ "$got" -ne "$want" ]; then
        echo "FATAL: ${tag} in ${dest} ran on ${got} distinct host(s), wanted ${want};" \
             "this point does not measure what it is named for." >&2
        exit 1
    fi
    echo "placement ok: ${tag} on ${got} distinct host(s)"
}

# A discarded warm-up before anything is measured. The first psi4 process in a
# fresh allocation pays for importing a 300 MB extension and reading the basis
# sets off the parallel filesystem, and on a cold node that tax lands entirely
# on whichever point happens to run first. An earlier run of this job measured
# it: the one-rank point reported 17.8 s outside J/K where every later point
# reported under 2.3 s, and the replicated metric factorization -- byte-for-byte
# the same arithmetic on every point -- read 0.369 s there against 0.036 s at
# two ranks. Since the one-rank point is the denominator of every speedup in the
# table, a contaminated baseline does not just add noise, it inflates the whole
# column. This runs at full width so every node's page cache is warm, and writes
# into a directory the reducer is never pointed at.
echo "=== warm-up (discarded): touching psi4 and the basis sets on all nodes ==="
run_point gtfock_df "$MAX_RANKS" "$CORES_PER_RANK" ppr:1:node none "$OUT/warmup_discarded"

# Sweep 1: one rank per whole node. Binding is off for the same reason as in the
# exact path's job -- a core set covering all 24 cores spans both sockets and
# OpenMPI refuses to bind it -- so OMP_PLACES=cores does the spreading.
for ranks in $(seq 1 "$MAX_RANKS"); do
    run_point gtfock_df "$ranks" "$CORES_PER_RANK" ppr:1:node none
done

# The references, single-process by construction and therefore one node's worth
# of hardware however many nodes this job holds. They go last, where a wall-clock
# overrun costs least.
run_point df     1 "$CORES_PER_RANK" - -
run_point direct 1 "$CORES_PER_RANK" - -

# Sweep 2: the locality control. Four ranks of six cores, packed and spread.
# Both points hold 24 cores in total, so a difference between them is not a
# difference in arithmetic. These two bind, unlike the sweep above: a six-core
# set fits inside one socket, so PE=6 places each rank on a disjoint set of
# physical cores instead of letting the runtime spread threads as it likes.
CONTROL_RANKS=4
CONTROL_THREADS=$((CORES_PER_RANK / CONTROL_RANKS))
# Both halves get the packed half's memory budget. The packed half cannot have
# more -- four ranks share one node's RAM -- and giving the spread half the full
# per-rank budget makes the memory setting a second variable, which is fatal to a
# control. Measured on the nanotube at 120 GB spread against 30 GB packed: the
# replicated dense Cholesky read 1.32x apart and the non-J/K part of SCF 3.2x
# apart, both on identical cores of the same node, so placement was not what the
# two halves differed in and the redistribution numbers meant nothing.
CONTROL_MEM_GB=$((MEMORY_PER_RANK_GB / CONTROL_RANKS))
echo "=== locality control: ${CONTROL_RANKS} x ${CONTROL_THREADS} cores, packed then spread ==="
# The spread half runs first. It is the half that placed correctly on the very
# first attempt, and it costs nothing to order the fragile half last: if the
# packed point is refused or lands wrong, the job dies here having already
# written the sweep, the references, and a control that measures something.
FM_MEM_OVERRIDE_GB=$CONTROL_MEM_GB \
run_point gtfock_df "$CONTROL_RANKS" "$CONTROL_THREADS" \
    "ppr:1:node:PE=${CONTROL_THREADS}" core "$OUT/control_spread_four_nodes"
assert_hosts "$OUT/control_spread_four_nodes" "${FM_SYSTEM}_gtfock_df_n${CONTROL_RANKS}" "$CONTROL_RANKS"

# -n 4 against ppr:4:node fills the first node and stops, so no --host is
# needed. An earlier version passed one; under a SLURM allocation OpenMPI treats
# --host as a filter over the allocated nodes rather than a slot declaration, so
# it changed nothing and the run died on "Procs mapped: 1".
FM_MEM_OVERRIDE_GB=$CONTROL_MEM_GB \
run_point gtfock_df "$CONTROL_RANKS" "$CONTROL_THREADS" \
    "ppr:${CONTROL_RANKS}:node:PE=${CONTROL_THREADS}" core \
    "$OUT/control_packed_one_node"
assert_hosts "$OUT/control_packed_one_node" "${FM_SYSTEM}_gtfock_df_n${CONTROL_RANKS}" 1

echo "=== sacct ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,NNodes,NCPUS -P \
    | tee "$OUT/sacct.txt"
echo "results in $OUT"
echo "SCALING_DONE"
