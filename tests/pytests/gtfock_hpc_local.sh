#!/bin/bash
# Fixed-core rank scaling for both GTFock J/K engines against both of Psi4's own,
# on a single workstation. This is the sweep that answers the question the
# Phoenix tables leave open: how a *distributed* density fitting compares against
# distributed exact GTFock, measured with the same build, on the same silicon, in
# one sitting.
#
#   FM_SYSTEM=peptide  bash tests/pytests/gtfock_hpc_local.sh
#   FM_SYSTEM=nanotube bash tests/pytests/gtfock_hpc_local.sh
#
# Every point uses the whole machine: the two single-process reference arms as
# one process with every core, the two GTFock arms as 1xN, 2x(N/2) and 4x(N/4)
# ranks-by-threads. Holding the core count fixed is what makes a wall-clock ratio
# between two points a property of the engine rather than of the hardware handed
# to it.
#
# This is a workstation, not a cluster node, and the difference matters for what
# may be compared with what: it is one socket and one NUMA domain, so no point
# here pays the cross-socket penalty that shapes the Phoenix tables, and no
# number from this sweep belongs in the same table as one from those.

set -eo pipefail

FM_PSI4=${FM_PSI4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FM_SYSTEM=${FM_SYSTEM:?set FM_SYSTEM to peptide or nanotube}
FM_BASIS=${FM_BASIS:-6-31+G**}
FM_METHOD=${FM_METHOD:-scf}
# Physical cores, not hardware threads: the SMT sibling of a busy core adds
# contention rather than throughput to an integral engine, and counting the
# siblings would make "all the cores" mean a 2x oversubscription.
TOTAL_CORES=${FM_TOTAL_CORES:-$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)}
TOTAL_MEMORY_GB=${FM_TOTAL_MEMORY_GB:-120}

OUT=${FM_OUT:-$FM_PSI4/../fm-gtfock-local/${FM_SYSTEM}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"

eval "$("$FM_PSI4/objdir_gtfock/stage/bin/psi4" --psiapi)"

# GTFock does not use MPI RMA, and OpenMPI's UCX one-sided component logs a
# priority-query failure on every rank before falling back to one that works.
export OMPI_MCA_osc=^ucx
export OMP_PLACES=cores
export OMP_PROC_BIND=close

{
  echo "=== provenance ==="
  echo "host        : $(hostname)"
  echo "date        : $(date -Is)"
  echo "cores       : ${TOTAL_CORES} physical, memory ${TOTAL_MEMORY_GB} GB total to Psi4"
  echo "system      : ${FM_SYSTEM} / ${FM_BASIS} / ${FM_METHOD}"
  echo "psi4        : $(git -C "$FM_PSI4" log --oneline -1)"
  echo "psi4 dirty  : $(git -C "$FM_PSI4" status --porcelain | wc -l) modified file(s)"
  echo "mpirun      : $(command -v mpirun) -- $(mpirun --version | head -1)"
  lscpu | grep -E "^(Model name|Socket|Core|Thread|NUMA node\(s\)|CPU\(s\)):"
  free -g | head -2
} | tee "$OUT/provenance.txt"

# run_point <arm> <ranks> <threads-per-rank>
run_point() {
    local arm=$1 ranks=$2 threads=$3
    local tag="${FM_SYSTEM}_${arm}_n${ranks}"
    local mem_gb=$((TOTAL_MEMORY_GB / ranks))
    echo "=== ${tag}: ${ranks} rank(s) x ${threads} thread(s), ${mem_gb} GB each ==="
    local -a cmd=(python "$FM_PSI4/tests/pytests/gtfock_hpc_benchmark.py"
                  --system "$FM_SYSTEM" --arm "$arm" --basis "$FM_BASIS"
                  --method "$FM_METHOD" --threads "$threads"
                  --memory "${mem_gb} GB" --scratch "$OUT/$tag"
                  --json-out "$OUT/$tag")
    export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
    case "$arm" in
        gtfock|gtfock_df)
            # PE=<threads> hands each rank a disjoint set of physical cores. On
            # this single-socket part every rank count binds, including one rank
            # over the whole package, so unlike the two-socket cluster sweep no
            # point here runs unbound and the placement is uniform across the
            # column.
            mpirun -n "$ranks" --map-by "ppr:${ranks}:node:PE=${threads}" \
                   --bind-to core --report-bindings "${cmd[@]}"
            ;;
        *)
            "${cmd[@]}"
            ;;
    esac
}

# The reference arms first: if the machine or the environment is broken, it costs
# one serial SCF to find out rather than the whole sweep.
run_point direct    1 "$TOTAL_CORES"          # exact ERIs: the reference for gtfock
run_point df        1 "$TOTAL_CORES"          # Psi4's density fitting: the reference for gtfock_df
run_point gtfock    1 "$TOTAL_CORES"
run_point gtfock    2 $((TOTAL_CORES / 2))
run_point gtfock    4 $((TOTAL_CORES / 4))
run_point gtfock_df 1 "$TOTAL_CORES"
run_point gtfock_df 2 $((TOTAL_CORES / 2))
run_point gtfock_df 4 $((TOTAL_CORES / 4))

echo "results in $OUT"
echo "SCALING_DONE"
