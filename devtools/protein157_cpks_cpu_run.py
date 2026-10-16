#!/usr/bin/env python
"""All-CPU SAPT(DFT)-D4(I) with coupled CPKS induction, under an XC grid-switching policy.

    python protein157_cpks_cpu_run.py --suite GEOMETRIES_DIR --system protein157 --output DIR \
        --iterator scf_iterator.py [--basis aug-cc-pvdz] [--threads 24] [--memory "620 GiB"]

The CPU baseline for the GPU timing-suite SAPT runs (cuest_xc_response_probe.py sapt): same
SAPT settings (PBE0, GRAC shifts 0.136/0.136, delta-HF, CPKS induction, D4(I)), on native CPU
DF J/K, native XC and CPU SAD. The XC grid policy (initial -> final, DFT_GRID_SWITCH_FACTOR,
minimum final-grid iterations) is applied through the scf_iterate overlay of
protein157_grid_policy_run.py, so the pinned build is untouched. The CPKS solves are timed
separately from the whole energy() call. No memory sampler; ru_maxrss is read once at the end.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

COMPONENTS = [f"SAPT {term} ENERGY" for term in ("ELST", "EXCH", "IND", "DISP", "TOTAL")]


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--basis", default="aug-cc-pvdz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterator", type=Path, required=True)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--memory", default="620 GiB")
    p.add_argument("--initial", default="35x194")
    p.add_argument("--final", default="75x302")
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--minimum", type=int, default=2)
    p.add_argument("--grac-shift", type=float, default=0.136)
    p.add_argument("--expected-nbf", type=int)
    p.add_argument("--source-commit")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    sys.argv.append("--einsums:no-attach-debugger")  # exit on a crash, never park in a debugger wait
    import psi4
    sys.argv.pop()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protein157_grid_policy_run import install_iterator
    sys.path.insert(0, str(args.suite))
    from systems import SYSTEMS
    assert args.system in SYSTEMS and args.system != "pla15_3pe2"
    geometry = SYSTEMS[args.system]

    os.chdir(args.output)
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)
    initial = [int(x) for x in args.initial.split("x")]
    final = [int(x) for x in args.final.split("x")]
    options = {
        "basis": args.basis, "scf_type": "df", "SCF_SUBTYPE": "AUTO",
        "SAPT_DFT_FUNCTIONAL": "pbe0", "SAPT_DFT_GRAC_SHIFT_A": args.grac_shift,
        "SAPT_DFT_GRAC_SHIFT_B": args.grac_shift, "SAPT_DFT_INDUCTION_TYPE": "CPKS",
        "SAPT_DFT_DO_DHF": True, "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
        "USE_CUEST": False, "CUEST_XC": False, "CUEST_SAD": False, "CUEST_MIXED_PRECISION": False,
        "DFT_RADIAL_POINTS": final[0], "DFT_SPHERICAL_POINTS": final[1],
        "DFT_RADIAL_POINTS_INITIAL": initial[0], "DFT_SPHERICAL_POINTS_INITIAL": initial[1],
        "DFT_GRID_SWITCH_FACTOR": args.factor,
    }
    record = {
        "system": args.system, "basis": args.basis, "geometry_sha256": sha256(geometry),
        "suite_systems_sha256": sha256((args.suite / "systems.py").read_bytes()),
        "options": options, "method": "sapt(dft)-d4(i)",
        "grid_policy": {"initial": args.initial, "final": args.final, "switch_factor": args.factor,
                        "minimum_final_iterations": args.minimum},
        "threads": args.threads, "memory": args.memory, "psi4_module": psi4.__file__,
        "psi4_version": psi4.__version__, "source_commit": args.source_commit,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_step_id": os.getenv("SLURM_STEP_ID"), "ok": False,
    }
    proofs = []
    record["iterator_overlay"] = {"path": str(args.iterator),
                                  "sha256": install_iterator(psi4, args.iterator, args.minimum, proofs)}

    # Time every CPKS solve on its own, inside the timed energy() call.
    modules = [importlib.import_module("psi4.driver.procrouting.sapt." + name)
               for name in ("sapt_jk_terms_ein", "sapt_jk_terms")]
    originals = [m._sapt_cpscf_solve for m in modules]
    cpks = []

    def wrap(original):
        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                cpks.append(time.perf_counter() - t0)
        return timed

    for m, original in zip(modules, originals):
        m._sapt_cpscf_solve = wrap(original)
    start = None
    try:
        molecule = psi4.geometry(geometry)
        psi4.set_options(options)
        build = psi4.core.BasisSet.build
        record["nbf"] = build(molecule, "BASIS", args.basis).nbf()
        record["nbf_monomer_a"] = build(molecule.extract_subsets(1), "BASIS", args.basis).nbf()
        record["nbf_monomer_b"] = build(molecule.extract_subsets(2), "BASIS", args.basis).nbf()
        record["naux_jkfit"] = build(molecule, "DF_BASIS_SCF", "", "JKFIT", args.basis).nbf()
        if args.expected_nbf is not None:
            assert record["nbf"] == args.expected_nbf, (record["nbf"], args.expected_nbf)
        psi4.core.clean_timers()
        start = time.perf_counter()
        energy = psi4.energy("sapt(dft)-d4(i)", molecule=molecule)
        record["wall_s"] = time.perf_counter() - start
        record["cpks_solve_s"] = sum(cpks)
        record["cpks_solve_calls_s"] = cpks
        record["timer_records"] = psi4.core.get_timer_records()
        record["returned_energy_hartree"] = energy
        record["components_hartree"] = {k: float(psi4.variable(k)) for k in COMPONENTS}
        record["grac_shifts_hartree"] = {s: float(psi4.variable(f"SAPT DFT GRAC SHIFT {s}"))
                                         for s in "AB" if psi4.core.has_variable(f"SAPT DFT GRAC SHIFT {s}")}
        psi4.core.close_outfile()
        text = (args.output / "psi4.out").read_text()
        record["memdfjk_seen"] = "MemDFJK: Density-Fitted" in text
        record["diskdfjk_seen"] = "DiskDFJK: Density-Fitted" in text
        record["cuest_seen"] = "cuESTJK" in text
        record["proofs"] = proofs
        switched = [q for q in proofs if q["switched"]]
        problems = []
        if record["cuest_seen"] or not (record["memdfjk_seen"] or record["diskdfjk_seen"]):
            problems.append("J/K is not CPU DF")
        if not switched:
            problems.append("no SCF switched grids")
        if any(not q["converged"] or q["final"] < args.minimum or not q["final_grid_after"] or q["initial"] < 1
               for q in switched):
            problems.append("a switched SCF violated the grid policy")
        if not cpks:
            problems.append("no CPKS solve was timed")
        record["policy_checks"] = {"switched_scfs": len(switched), "problems": problems}
        record["ok"] = not problems
        if problems:
            record["error"] = "; ".join(problems)
    except Exception as exc:
        if start is not None:
            record.setdefault("wall_s", time.perf_counter() - start)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["proofs"] = proofs
        record["cpks_solve_calls_s"] = cpks
    finally:
        for m, original in zip(modules, originals):
            m._sapt_cpscf_solve = original
        record["ru_maxrss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        psi4.core.close_outfile()
        (args.output / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: record.get(k) for k in ("system", "ok", "error", "wall_s", "cpks_solve_s", "nbf",
                                                  "naux_jkfit", "components_hartree", "policy_checks",
                                                  "ru_maxrss_gib")}), flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
