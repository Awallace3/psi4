"""Run one point of the GTFock HPC rank-scaling benchmark and report per rank.

One invocation is one SCF: one system, one basis, one J/K algorithm, one rank
count. The orchestration -- which points to run and in which order -- lives in
the SLURM script (:source:`tests/pytests/gtfock_hpc_phoenix.slurm`), so this
script stays a single measurement that reports and does not judge::

    # GTFock, four ranks, six OpenMP threads each
    mpirun -n 4 python gtfock_hpc_benchmark.py --system peptide --arm gtfock \
        --threads 6 --memory "40 GB" --scratch runs/pep_gtfock_n4

    # the reference arm, one process with all the cores
    python gtfock_hpc_benchmark.py --system peptide --arm direct \
        --threads 24 --memory "160 GB" --scratch runs/pep_direct

Every process prints one ``PSI4-GTFOCK-JSON <json>`` line and, with
``--json-out``, writes the same record to a per-rank file for
:source:`tests/pytests/gtfock_hpc_collect.py` to gather. The record carries the
whole SCF wall clock, the J/K-build wall clock on its own, the iteration count,
the final energy and this process's peak RSS, so a differing iteration count or a
memory blow-up cannot hide inside a wall-clock ratio.

The two systems are dimers from the SAPT(DFT) + DFT-D4 timing set: a
twelve-atom-per-monomer peptide backbone pair and an ethene molecule on a carbon
nanotube fragment. They are run as single closed-shell molecules, not as SAPT --
this measures a Fock build, not an interaction energy.

The basis is 6-31+G**, which |PSIfour| ships as a Cartesian basis
(``psi4/share/psi4/basis/6-31pgss.gbs`` declares ``cartesian``), so it needs no
``puream`` override to run through GTFock's Cartesian-only Simint path. The
driver asserts that, since a silently spherical basis would be refused by the
engine at best and be a different computation at worst.
"""

import argparse
import json
import os
import platform
import socket
import sys
import time

import psi4

# Geometries transcribed from the SAPT(DFT) timing set (ids 144 and 154), in
# angstrom, C1 with no reorientation and no centre-of-mass shift: GTFock's calls
# are collective, so a geometry that differed between ranks would deadlock rather
# than disagree.
_SYSTEMS = {
    # Two 12-atom peptide-backbone fragments (6 C, 2 N, 2 O, 14 H). 260 basis
    # functions in 6-31+G**.
    "peptide": """
0 1
    C      3.281507600000     1.445273000000     1.034710730000
    C      1.604507600000     0.361273004000    -2.219289270000
    C      2.275507600000     0.774273004000     0.103710727000
    O      1.326507600000     0.135273004000     0.560710727000
    N      2.500507600000     0.907273004000    -1.203289270000
    H      3.279507600000     0.902273004000     1.980710730000
    H      3.307507600000     1.437273000000    -1.515289270000
    H      0.579507596000     0.630273004000    -1.981289270000
    H      4.297647600000     1.415103000000     0.614526727000
    H      2.980587600000     2.483703000000     1.237488730000
    H      1.855712800000     0.766963004000    -3.210406270000
    H      1.647067600000    -0.734436996000    -2.306570270000
    C     -1.669492400000    -2.208727000000    -0.834289273000
    C     -1.959492400000     0.854273004000     1.385710730000
    C     -2.391492400000    -0.402726996000     0.645710727000
    O     -3.564492400000    -0.771726996000     0.627710727000
    N     -1.417492400000    -1.039727000000    -0.000289273110
    H     -0.470492404000    -0.681726996000     0.081710726900
    H     -0.735492404000    -2.756727000000    -0.957289273000
    H     -2.384492400000    -2.865727000000    -0.336289273000
    H     -0.871492404000     0.898273004000     1.327710730000
    H     -2.060812400000    -1.976247000000    -1.835698270000
    H     -2.342852400000     1.773423000000     0.918589727000
    H     -2.197492400000     0.842013004000     2.459585730000
units angstrom
symmetry c1
no_reorient
no_com
""",
    # Ethene on a carbon-nanotube fragment (26 C, 16 H). 574 basis functions in
    # 6-31+G**, which is ~140 per rank at four ranks -- comfortably more than one
    # AO block each, which is what makes the decomposition worth measuring.
    "nanotube": """
0 1
    H     -0.923387110000     2.620325580000     1.228608870000
    H      0.923387110000     2.620325580000     1.228608870000
    H      0.923387110000     2.620325580000    -1.228608870000
    H     -0.923387110000     2.620325580000    -1.228608870000
    C      0.000000000000     2.620325580000    -0.666616630000
    C      0.000000000000     2.620325580000     0.666616630000
    C      3.330973000000    -1.379737000000    -0.710500000000
    C      3.330973000000    -1.379737000000     0.710500000000
    C     -3.330973000000    -1.379737000000    -0.710500000000
    C     -3.330973000000    -1.379737000000     0.710500000000
    C      2.355354000000    -0.631118000000    -2.843523000000
    C      2.355354000000    -0.631118000000    -1.422523000000
    C      2.355354000000    -0.631118000000     1.422523000000
    C      2.355354000000    -0.631118000000     2.843523000000
    C     -2.355354000000    -0.631118000000    -2.843523000000
    C     -2.355354000000    -0.631118000000    -1.422523000000
    C     -2.355354000000    -0.631118000000     1.422523000000
    C     -2.355354000000    -0.631118000000     2.843523000000
    C      1.219221000000    -0.160516000000    -3.555546500000
    C      1.219221000000    -0.160516000000    -0.710500000000
    C      1.219221000000    -0.160516000000     0.710500000000
    C      1.219221000000    -0.160516000000     3.555546500000
    C     -1.219221000000    -0.160516000000    -3.555546500000
    C     -1.219221000000    -0.160516000000    -0.710500000000
    C     -1.219221000000    -0.160516000000     0.710500000000
    C     -1.219221000000    -0.160516000000     3.555546500000
    C      0.000000000000    -0.000002999906    -2.843523000000
    C      0.000000000000    -0.000002999906    -1.422523000000
    C      0.000000000000    -0.000002999906     1.422523000000
    C      0.000000000000    -0.000002999906     2.843523000000
    H      3.902336500000    -2.124352200000    -1.253932500000
    H      3.902336500000    -2.124352200000     1.253932500000
    H     -3.902336500000    -2.124352200000    -1.253932500000
    H     -3.902336500000    -2.124352200000     1.253932500000
    H      3.099969000000    -1.202481400000    -3.386955800000
    H      3.099969000000    -1.202481400000     3.386955800000
    H     -3.099969000000    -1.202481400000    -3.386955800000
    H     -3.099969000000    -1.202481400000     3.386955800000
    H      1.219221000000    -0.160516000000    -4.640086500000
    H      1.219221000000    -0.160516000000     4.640086500000
    H     -1.219221000000    -0.160516000000    -4.640086500000
    H     -1.219221000000    -0.160516000000     4.640086500000
units angstrom
symmetry c1
no_reorient
no_com
""",
}

# Expected basis-function counts, asserted so a geometry or basis mix-up shows up
# in the first second of a queued job rather than in the results table.
_EXPECTED_NBF = {"peptide": 260, "nanotube": 574}

# J/K algorithms this script can put under the same SCF. "gtfock" is the engine
# under test; "direct" is Psi4's own exact-ERI builder and is the algorithmic
# apples-to-apples reference; "df" is density fitting, a *different* algorithm
# that is what most users actually run, recorded as a secondary line.
_ARMS = {"gtfock": "gtfock", "direct": "direct", "df": "df", "pk": "pk"}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=sorted(_SYSTEMS), required=True)
    parser.add_argument("--arm", choices=sorted(_ARMS), required=True,
                        help="which J/K algorithm to put under the SCF")
    parser.add_argument("--basis", default="6-31+G**")
    parser.add_argument("--method", default="scf",
                        help="anything psi4.energy accepts that needs only J and K")
    parser.add_argument("--threads", type=int, default=1,
                        help="OpenMP threads for this process; ranks x threads is "
                             "held constant across the comparison")
    parser.add_argument("--memory", default="4 GB")
    parser.add_argument("--scratch", default=".",
                        help="directory for per-rank scratch and Psi4 output files")
    parser.add_argument("--json-out", default=None,
                        help="write this rank's record to <json-out>.rank<N>.json")
    return parser.parse_args(argv)


def peak_rss_mb() -> float:
    """This process's high-water resident set size, in MiB.

    VmHWM is the kernel's own peak, so it survives the allocator having already
    given the memory back by the time the SCF returns.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def jk_build_seconds():
    """Wall clock inside ``JK::compute()``, and how many times it ran.

    ``JK: JK`` is the timer libfock wraps around ``compute_JK()`` for every JK
    subclass, so the same number means the same thing in every arm: integrals
    plus whatever communication that arm needs, and nothing of the
    diagonalization, DIIS or guess around it.
    """
    for record in psi4.core.get_timer_records(False).values():
        if record.get("timer_name") == "JK: JK":
            return record["wall_time"], record["n_calls"]
    return None, 0


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.arm == "gtfock":
        # mpi4py's import is what calls MPI_Init_thread; go through Psi4's own
        # entry point so the benchmark exercises the shipped module.
        from psi4.driver import gtfock
        info = gtfock.initialize()
        rank, size = info["rank"], info["size"]
        processor = info["processor"]
    else:
        # The reference arms are one process with every core, so there is no MPI
        # to bring up and nothing to initialize.
        gtfock = None
        rank, size, processor = 0, 1, platform.processor()

    rank_scratch = os.path.join(args.scratch, f"rank{rank}")
    os.makedirs(rank_scratch, exist_ok=True)
    psi4.core.IOManager.shared_object().set_default_path(rank_scratch)
    psi4.core.set_output_file(
        os.path.join(args.scratch, f"psi4_rank{rank}.out"), False)

    psi4.set_num_threads(args.threads)
    psi4.set_memory(args.memory)

    psi4.geometry(_SYSTEMS[args.system])
    # Every arm gets the same convergence, the same guess and the same integral
    # threshold, so the only thing that differs between two points is the J/K
    # algorithm and the rank count. df_scf_guess is off everywhere: it would
    # otherwise put a DF-SCF in front of the algorithm being measured.
    psi4.set_options({
        "basis": args.basis,
        "scf_type": _ARMS[args.arm],
        "df_scf_guess": False,
        # SAD, and SAD's own atomic J/K pinned to DF, on every point. SAD builds
        # its own JK for the atomic subproblems (libscf_solver/sad.cc) rather
        # than routing through SCF_TYPE, so pinning SAD_SCF_TYPE makes the guess
        # bit-identical work in all three arms -- and replicated on every rank,
        # which is part of what the total-vs-J/K wall clock split below exposes.
        "guess": "sad",
        "sad_scf_type": "df",
        "e_convergence": 1e-8,
        "d_convergence": 1e-7,
        "ints_tolerance": 1e-12,
        "maxiter": 100,
        # Keeps the JK object alive past HF::finalize() so its class can be
        # reported; without it psi4 releases the object and wfn.jk() is None.
        "save_jk": True,
    })

    builds_before = gtfock.fock_builds() if gtfock is not None else 0
    start = time.perf_counter()
    energy, wfn = psi4.energy(args.method, return_wfn=True)
    elapsed = time.perf_counter() - start
    jk_wall, jk_calls = jk_build_seconds()

    basis = wfn.basisset()
    if basis.has_puream():
        raise RuntimeError(
            f"{args.basis} came up spherical: GTFock's Simint path is Cartesian "
            "only, and a spherical basis would make the two arms different "
            "computations. Check the basis definition rather than forcing "
            "puream.")
    expected = _EXPECTED_NBF[args.system]
    if basis.nbf() != expected:
        raise RuntimeError(
            f"{args.system}/{args.basis} gave {basis.nbf()} basis functions, "
            f"expected {expected}: the geometry or the basis is not what this "
            "benchmark was sized for.")

    report = {
        "system": args.system,
        "arm": args.arm,
        "basis": args.basis,
        "method": args.method,
        "nbf": basis.nbf(),
        "nshell": basis.nshell(),
        "puream": basis.has_puream(),
        "ranks": size,
        "rank": rank,
        "threads_per_rank": args.threads,
        "total_cores": size * args.threads,
        "scf_energy": energy,
        "iterations": wfn.iteration_,
        "scf_wall_seconds": elapsed,
        "jk_wall_seconds": jk_wall,
        "jk_calls": jk_calls,
        "peak_rss_mb": peak_rss_mb(),
        "jk_name": wfn.jk().name(),
        "df_basis_scf": psi4.core.get_global_option("DF_BASIS_SCF") or "(auto)",
        "host": socket.gethostname(),
        "processor": processor,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }
    if gtfock is not None:
        report.update({
            "fock_builds": gtfock.fock_builds() - builds_before,
            "core_mpi": gtfock.mpi_info(),
            "process_grid": psi4.core.gtfock_process_grid(),
            "local_block": psi4.core.gtfock_local_block(),
            "local_task_shape": psi4.core.gtfock_local_task_shape(),
        })

    print("PSI4-GTFOCK-JSON " + json.dumps(report), flush=True)
    if args.json_out:
        with open(f"{args.json_out}.rank{rank}.json", "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    # Releasing the GTFock engine is collective, so drop the wavefunction here,
    # at the same point on every rank, rather than at interpreter teardown.
    del wfn
    return 0


if __name__ == "__main__":
    sys.exit(main())
