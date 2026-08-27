#!/bin/bash
# Rank-scaling measurement for the GTFock J/K engine, as actually submitted on
# Georgia Tech's Phoenix (PACE) cluster. One job measures one molecule, and every
# point in it runs on the same physical node in the same allocation, so a
# wall-clock ratio between two points cannot be an artifact of landing on
# different hardware.
#
#   sbatch --job-name=gtf-peptide  --export=ALL,FM_SYSTEM=peptide  gtfock_hpc_phoenix.slurm
#   sbatch --job-name=gtf-nanotube --export=ALL,FM_SYSTEM=nanotube gtfock_hpc_phoenix.slurm
#
# The allocation is a whole 24-core node, and every point uses all 24 cores:
# the reference arms as one process with 24 threads, GTFock as 1x24, 2x12 and
# 4x6 ranks-by-threads. Comparing a 4-rank run against a 1-rank run that had
# only a quarter of the machine would measure the allocation, not the engine.
#
#SBATCH --account=gts-cs207-chemx
#SBATCH --qos=inferno
#SBATCH --partition=cpu-small
#SBATCH --constraint=core24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -eo pipefail

# Site paths. Everything lives in scratch; nothing is written into the repo.
FM_ROOT=${FM_ROOT:-/storage/scratch1/3/awallace43/fm-gtfock}
FM_ENV=${FM_ENV:-$FM_ROOT/envs/p4gtf}
FM_PSI4=${FM_PSI4:-$FM_ROOT/psi4}
FM_SYSTEM=${FM_SYSTEM:?set FM_SYSTEM to peptide or nanotube}
FM_BASIS=${FM_BASIS:-6-31+G**}
FM_METHOD=${FM_METHOD:-scf}
# Total cores and total Psi4 memory are the two things held constant across
# every point; per-process values are these divided by the rank count.
TOTAL_CORES=${SLURM_NTASKS:-24}
TOTAL_MEMORY_GB=${FM_TOTAL_MEMORY_GB:-120}

OUT=$FM_ROOT/results/${FM_SYSTEM}_${SLURM_JOB_ID}
mkdir -p "$OUT"

module purge 2>/dev/null || true
source /storage/project/r-cs207-0/awallace43/miniconda/etc/profile.d/conda.sh
conda activate "$FM_ENV"
eval $("$FM_PSI4/objdir_gtfock/stage/bin/psi4" --psiapi)

# OpenMPI's UCX one-sided component logs a priority-query failure on every rank
# on this cluster and then falls back; GTFock does not use MPI RMA, so skip it
# rather than read the noise.
export OMPI_MCA_osc=^ucx
export OMP_PLACES=cores
export OMP_PROC_BIND=close

{
  echo "=== provenance ==="
  echo "job         : ${SLURM_JOB_ID} ${SLURM_JOB_NAME}"
  echo "node        : $(hostname) (${SLURM_JOB_NODELIST})"
  echo "partition   : ${SLURM_JOB_PARTITION}  account: ${SLURM_JOB_ACCOUNT}  qos: ${SLURM_JOB_QOS}"
  echo "cores       : ${TOTAL_CORES} total, memory ${TOTAL_MEMORY_GB} GB total"
  echo "system      : ${FM_SYSTEM} / ${FM_BASIS} / ${FM_METHOD}"
  echo "psi4        : $(git -C "$FM_PSI4" log --oneline -1)"
  echo "mpirun      : $(which mpirun) -- $(mpirun --version | head -1)"
  lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)):"
  free -g | head -2
  scontrol show node "${SLURMD_NODENAME}" | tr ' ' '\n' | grep -E "^(CPUTot|RealMemory|AvailableFeatures)="
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
    if [ "$arm" = "gtfock" ]; then
        # PE=<threads> hands each rank its own set of physical cores, so the
        # ranks do not contend for the same ones as the rank count rises.
        mpirun -n "$ranks" --map-by "ppr:${ranks}:node:PE=${threads}" --bind-to core \
            "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

# The reference arms first: if the machine or the environment is broken, it costs
# one serial SCF to find out rather than the whole sweep.
run_point direct 1 "$TOTAL_CORES"    # exact ERIs, the algorithmic apples-to-apples baseline
run_point df     1 "$TOTAL_CORES"    # density fitting: a DIFFERENT algorithm, recorded for context
run_point gtfock 1 "$TOTAL_CORES"
run_point gtfock 2 $((TOTAL_CORES / 2))
run_point gtfock 4 $((TOTAL_CORES / 4))

echo "=== sacct ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,NNodes,NCPUS -P \
    | tee "$OUT/sacct.txt"
echo "results in $OUT"
echo "SCALING_DONE"
