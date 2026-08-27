"""Run one SCF energy through Psi4's GTFock MPI J/K engine and report per rank.

Launched by ``tests/pytests/test_gtfock.py`` and by
``tests/pytests/gtfock_benchmark.py`` as::

    mpirun -n <N> python gtfock_mpi_driver.py <scratch-dir> [options]

Every rank prints one ``PSI4-GTFOCK-JSON <json>`` line describing what its own
process saw: the MPI world according to mpi4py *and* according to Psi4's linked
GTFock shim, the AO block GTFock handed this rank and how GTFock blocked it, the
JK class libfock actually built, how many GTFock Fock builds ran, the SCF energy,
and the wall-clock the SCF took. The callers assert on those, so this script only
reports; it does not judge.

The default case is the STO-3G water RHF the milestone-2 test used. ``--molecule``
and ``--basis`` reach the larger systems the rank-count and timing evidence need,
and ``--method`` reaches hybrid DFT; ``scf`` and any hybrid functional go through
the same GTFock J/K, so one script covers both.
"""

import argparse
import json
import os
import sys
import time

# mpi4py's import is what calls MPI_Init_thread. Do it through Psi4's own entry
# point so the test exercises the shipped module rather than a private path.
import psi4
from psi4.driver import gtfock

# Geometries are C1 with no reorientation and no centre-of-mass shift, so every
# rank builds a bit-identical molecule; GTFock's calls are collective and a
# geometry that differed between ranks would deadlock rather than disagree.
_MOLECULES = {
    # The milestone-2 smoke case: one water, 7 basis functions in STO-3G.
    "water": """
0 1
O   0.000000000000   0.000000000000  -0.068516219320
H   0.000000000000  -0.790689573744   0.543701060715
H   0.000000000000   0.790689573744   0.543701060715
units angstrom
symmetry c1
no_reorient
no_com
""",
    # A water hexamer cage: 18 atoms, 60 shells and 114 basis functions in
    # Cartesian 6-31G*, 390 in Cartesian cc-pVTZ. Big enough that GTFock's 2x2
    # grid at four ranks gives each rank a 55x55 AO panel which GTFock then
    # splits into 4x4 task blocks -- see local_task_shape in the report. A
    # single water would give each rank one block and prove nothing about the
    # decomposition.
    "water6": """
0 1
O   -0.7021  -1.4498   1.2298
H   -1.0688  -0.6396   1.6136
H   -0.0921  -1.1329   0.5566
O    1.2456  -1.3175  -0.3899
H    1.9020  -0.7157   0.0018
H    0.4368  -0.9294  -0.0208
O    0.6021   1.4079  -1.2043
H    0.9873   0.5451  -1.0090
H   -0.2069   1.2039  -0.7108
O   -1.2864   1.2517   0.3745
H   -1.9403   0.5615  -0.0060
H   -0.4655   0.8484   0.0463
O    2.0281   1.0157   1.5088
H    2.6083   1.7827   1.5334
H    1.6122   1.0416   0.6383
O   -2.1962  -1.0532  -1.5177
H   -2.8130  -1.7811  -1.4179
H   -1.7008  -1.1670  -0.6919
units angstrom
symmetry c1
no_reorient
no_com
""",
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scratch", nargs="?", default=".",
                        help="directory for per-rank scratch and output files")
    parser.add_argument("--molecule", choices=sorted(_MOLECULES), default="water")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--method", default="scf",
                        help="anything psi4.energy accepts that needs only J and K, "
                             "i.e. scf or a hybrid functional such as b3lyp")
    parser.add_argument("--memory", default="1 GB")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    info = gtfock.initialize()
    rank = info["rank"]

    # Ranks share a filesystem, so keep scratch and output per rank.
    rank_scratch = os.path.join(args.scratch, f"rank{rank}")
    os.makedirs(rank_scratch, exist_ok=True)
    psi4.core.IOManager.shared_object().set_default_path(rank_scratch)
    psi4.core.set_output_file(os.path.join(args.scratch, f"gtfock_rank{rank}.out"), False)

    # One thread per rank keeps the run reproducible across ranks, which matters
    # because every rank must reach each collective GTFock call in lockstep.
    psi4.set_num_threads(1)
    psi4.set_memory(args.memory)

    psi4.geometry(_MOLECULES[args.molecule])
    # GTFock's Simint path is Cartesian; see MinimalInterface::check_supported.
    psi4.set_options({
        "basis": args.basis,
        "puream": False,
        "scf_type": "gtfock",
        "df_scf_guess": False,
        "guess": "core",
        "e_convergence": 1e-10,
        "d_convergence": 1e-9,
        # Keep the JK object alive past HF::finalize() so its class name can be
        # reported; without this psi4 releases it and wfn.jk() is None.
        "save_jk": True,
    })

    builds_before = gtfock.fock_builds()
    # Wall-clock of the whole SCF, timed identically on every rank. The ranks
    # leave the SCF together (the last GTFock build is collective), so the
    # spread across ranks is load imbalance plus startup skew, not signal.
    start = time.perf_counter()
    energy, wfn = psi4.energy(args.method, return_wfn=True)
    elapsed = time.perf_counter() - start

    report = {
        "mpi4py_rank": info["rank"],
        "mpi4py_size": info["size"],
        "core_mpi": gtfock.mpi_info(),
        "process_grid": psi4.core.gtfock_process_grid(),
        "local_block": psi4.core.gtfock_local_block(),
        "local_task_shape": psi4.core.gtfock_local_task_shape(),
        "jk_name": wfn.jk().name(),
        "fock_builds": gtfock.fock_builds() - builds_before,
        "scf_energy": energy,
        "scf_wall_seconds": elapsed,
        "iterations": wfn.iteration_,
        "molecule": args.molecule,
        "basis": args.basis,
        "method": args.method,
        "nbf": wfn.basisset().nbf(),
        "nshell": wfn.basisset().nshell(),
        "processor": info["processor"],
    }
    print("PSI4-GTFOCK-JSON " + json.dumps(report), flush=True)

    # Releasing the GTFock engine is collective, so drop the wavefunction here,
    # at the same point on every rank, rather than at interpreter teardown.
    del wfn
    return 0


if __name__ == "__main__":
    sys.exit(main())
