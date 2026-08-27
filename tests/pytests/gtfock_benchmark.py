"""Measure how a GTFock-driven SCF scales with MPI rank count.

This is the end-to-end distributed run the GTFock documentation quotes numbers
from. It is a script, not a pytest test: the default case takes minutes, which
is too long for the suite, and its output is a measurement rather than an
assertion.

Usage::

    python gtfock_benchmark.py --ranks 1,2,4 --molecule water6 --basis cc-pVTZ

For each rank count it launches ``mpirun -n <N> python gtfock_mpi_driver.py``,
collects the per-rank JSON reports, and prints one row per rank count: the
process grid GTFock chose, the AO blocking, the slowest rank's wall-clock, the
speedup over the smallest rank count measured, and the spread in SCF energy
across ranks. The energy spread is the rank-count-invariance evidence; the
wall-clock is the scaling evidence. Both are reported as measured -- a speedup
below one is a real result about this machine, not a failure of the script.

``--reference direct`` additionally runs a single-process non-GTFock SCF with
Psi4's own DirectJK and reports each rank count's deviation from it. That costs
roughly as much as the one-rank GTFock run, so it is off by default; the pytest
suite pins GTFock against Psi4's PK integrals on a smaller basis.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gtfock_mpi_driver.py")
JSON_PREFIX = "PSI4-GTFOCK-JSON "


def oversubscribe_flag(mpirun):
    """``--oversubscribe`` is an Open MPI spelling; MPICH's mpiexec rejects it.

    Both implementations install a program called ``mpirun``, so ask the
    launcher on PATH whether it accepts the flag rather than guessing from its
    name. Without it Open MPI refuses to place more ranks than the box has
    slots, which is the usual case on a shared machine.
    """
    try:
        probe = subprocess.run([mpirun, "--oversubscribe", "-n", "1", sys.executable, "-c", ""],
                               capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return []
    return ["--oversubscribe"] if probe.returncode == 0 else []


def run_ranks(launch, nranks, scratch, args):
    """Run one rank count and return its per-rank reports."""
    rank_scratch = os.path.join(scratch, f"n{nranks}")
    os.makedirs(rank_scratch, exist_ok=True)

    env = dict(os.environ)
    # One OpenMP thread per rank. Threads and ranks competing for the same cores
    # would make the rank-count comparison measure oversubscription instead of
    # distribution.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = launch + ["-n", str(nranks), sys.executable, DRIVER, rank_scratch,
                    "--molecule", args.molecule, "--basis", args.basis,
                    "--method", args.method, "--memory", args.memory]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.timeout)
    if proc.returncode != 0:
        raise SystemExit(f"mpirun -n {nranks} failed:\n{proc.stdout}\n{proc.stderr}")

    reports = [json.loads(line[len(JSON_PREFIX):])
               for line in proc.stdout.splitlines() if line.startswith(JSON_PREFIX)]
    if len(reports) != nranks:
        raise SystemExit(f"expected {nranks} rank reports, got {len(reports)}:\n{proc.stdout}")
    return reports


def reference_energy(args):
    """Single-process, non-GTFock SCF with Psi4's own exact-integral JK."""
    import psi4

    sys.path.insert(0, os.path.dirname(DRIVER))
    from gtfock_mpi_driver import _MOLECULES

    psi4.core.set_output_file(os.path.join(args.scratch or ".", "gtfock_benchmark_reference.out"), False)
    psi4.set_num_threads(args.reference_threads)
    psi4.set_memory(args.memory)
    psi4.geometry(_MOLECULES[args.molecule])
    psi4.set_options({"basis": args.basis, "puream": False, "scf_type": args.reference,
                      "df_scf_guess": False, "guess": "core",
                      "e_convergence": 1e-10, "d_convergence": 1e-9})
    return psi4.energy(args.method)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ranks", default="1,2,4",
                        help="comma-separated MPI rank counts to measure (default 1,2,4)")
    parser.add_argument("--molecule", default="water6")
    parser.add_argument("--basis", default="cc-pVTZ")
    parser.add_argument("--method", default="scf",
                        help="scf, or a hybrid functional such as b3lyp")
    parser.add_argument("--memory", default="4 GB")
    parser.add_argument("--reference", default="none",
                        choices=["none", "direct", "pk", "df"],
                        help="also run a single-process non-GTFock SCF to compare against")
    parser.add_argument("--reference-threads", type=int, default=1,
                        help="threads for the reference run; 1 makes it comparable to one rank")
    parser.add_argument("--scratch", default=None,
                        help="scratch directory (a temporary one is used and kept if omitted)")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--json", default=None, help="also write the raw reports here")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        raise SystemExit("no mpirun/mpiexec on PATH; the GTFock path needs an MPI launcher")
    launch = [mpirun] + oversubscribe_flag(mpirun)

    args.scratch = args.scratch or tempfile.mkdtemp(prefix="gtfock_benchmark_")
    os.makedirs(args.scratch, exist_ok=True)

    e_ref = reference_energy(args) if args.reference != "none" else None

    rank_counts = [int(n) for n in args.ranks.split(",") if n.strip()]
    results = {}
    for nranks in rank_counts:
        results[nranks] = run_ranks(launch, nranks, args.scratch, args)
        slowest = max(r["scf_wall_seconds"] for r in results[nranks])
        print(f"  n={nranks}: {slowest:.1f} s", file=sys.stderr, flush=True)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({str(k): v for k, v in results.items()}, handle, indent=2)

    first = results[rank_counts[0]][0]
    baseline = max(r["scf_wall_seconds"] for r in results[rank_counts[0]])
    print()
    print(f"{args.method} / {args.molecule} / {args.basis}   "
          f"nbf={first['nbf']} nshell={first['nshell']} on {first['processor']}")
    header = f"{'ranks':>5}  {'grid':>7}  {'blocks/rank':>11}  {'iters':>5}  {'wall (s)':>9}  {'speedup':>7}  {'dE across ranks':>15}"
    if e_ref is not None:
        header += f"  {'dE vs ' + args.reference:>14}"
    print(header)
    print("-" * len(header))
    for nranks in rank_counts:
        reports = results[nranks]
        wall = max(r["scf_wall_seconds"] for r in reports)
        energies = [r["scf_energy"] for r in reports]
        spread = max(energies) - min(energies)
        grid = "x".join(str(v) for v in reports[0]["process_grid"])
        blocks = "x".join(str(v) for v in reports[0]["local_task_shape"][:2])
        row = (f"{nranks:>5}  {grid:>7}  {blocks:>11}  {reports[0]['iterations']:>5}  "
               f"{wall:>9.1f}  {baseline / wall:>7.2f}  {spread:>15.2e}")
        if e_ref is not None:
            row += f"  {max(abs(e - e_ref) for e in energies):>14.2e}"
        print(row)
    print()
    print(f"energies: " + ", ".join(f"n={n}: {results[n][0]['scf_energy']:.10f}" for n in rank_counts))
    if e_ref is not None:
        print(f"reference ({args.reference}, single process): {e_ref:.10f}")
    print(f"scratch and per-rank outputs: {args.scratch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
