"""Run one RHF energy through Psi4's GTFock MPI J/K engine and report per rank.

Launched by ``tests/pytests/test_gtfock.py`` as::

    mpirun -n <N> python gtfock_mpi_driver.py <scratch-dir>

Every rank prints one ``PSI4-GTFOCK-JSON <json>`` line describing what its own
process saw: the MPI world according to mpi4py *and* according to Psi4's linked
GTFock shim, the AO block GTFock handed this rank, the JK class libfock actually
built, how many GTFock Fock builds ran, and the RHF energy. The test asserts on
those, so this script only reports; it does not judge.
"""

import json
import os
import sys

# mpi4py's import is what calls MPI_Init_thread. Do it through Psi4's own entry
# point so the test exercises the shipped module rather than a private path.
import psi4
from psi4.driver import gtfock

# STO-3G water, C1, no reorientation so the geometry is identical on every rank.
GEOMETRY = """
0 1
O   0.000000000000   0.000000000000  -0.068516219320
H   0.000000000000  -0.790689573744   0.543701060715
H   0.000000000000   0.790689573744   0.543701060715
units angstrom
symmetry c1
no_reorient
no_com
"""


def main() -> int:
    scratch = sys.argv[1] if len(sys.argv) > 1 else "."

    info = gtfock.initialize()
    rank = info["rank"]

    # Ranks share a filesystem, so keep scratch and output per rank.
    rank_scratch = os.path.join(scratch, f"rank{rank}")
    os.makedirs(rank_scratch, exist_ok=True)
    psi4.core.IOManager.shared_object().set_default_path(rank_scratch)
    psi4.core.set_output_file(os.path.join(scratch, f"gtfock_rank{rank}.out"), False)

    # One thread per rank keeps the run reproducible across ranks, which matters
    # because every rank must reach each collective GTFock call in lockstep.
    psi4.set_num_threads(1)
    psi4.set_memory("1 GB")

    psi4.geometry(GEOMETRY)
    # GTFock's Simint path is Cartesian; see MinimalInterface::check_supported.
    psi4.set_options({
        "basis": "sto-3g",
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
    energy, wfn = psi4.energy("scf", return_wfn=True)

    report = {
        "mpi4py_rank": info["rank"],
        "mpi4py_size": info["size"],
        "core_mpi": gtfock.mpi_info(),
        "process_grid": psi4.core.gtfock_process_grid(),
        "local_block": psi4.core.gtfock_local_block(),
        "jk_name": wfn.jk().name(),
        "fock_builds": gtfock.fock_builds() - builds_before,
        "scf_energy": energy,
        "nbf": wfn.basisset().nbf(),
        "processor": info["processor"],
    }
    print("PSI4-GTFOCK-JSON " + json.dumps(report), flush=True)

    # Releasing the GTFock engine is collective, so drop the wavefunction here,
    # at the same point on every rank, rather than at interpreter teardown.
    del wfn
    return 0


if __name__ == "__main__":
    sys.exit(main())
