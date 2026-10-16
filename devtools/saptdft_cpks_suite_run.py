#!/usr/bin/env python
"""One fresh-process SAPT(DFT)-D4(I) coupled-CPKS energy for the CPU/GPU size-scaling suite.

    python saptdft_cpks_suite_run.py --arm cpu|gpu --suite GEOMETRIES_DIR --system peptide \
        --output DIR [--threads 24] [--memory "200 GiB"] [--smoke]

Both arms run identical scientific options, the Protein157 6-31+g** specification of CPU
job 13868775 with coupled CPKS induction and fixed GRAC shifts 0.136/0.136 (as in the earlier
coupled-CPKS suite) instead of induction NONE and iterative GRAC. Only the backend options
differ: cpu = native DF J/K + native XC + CPU SAD; gpu = cuEST DF J/K + cuEST XC + GPU SAD.
The XC grid policy is 35x194 -> 75x302, DFT_GRID_SWITCH_FACTOR 1, at least 2 final-grid
iterations per switched SCF (scf_iterate(..., xc_final_min_iterations=2), wrapped to record a
proof per SCF). --smoke runs the same options on a water dimer.

Records: energy() wall (fresh process), psi4 timer records (the whole hierarchical tree, as
read by the report's 17-operation partition), each CPKS solve's wall time and its printed
residual history, components, GRAC shifts, actual nbf/naux (spherical, PUREAM true), backends
seen in the output, per-SCF grid proofs, ru_maxrss, and sha256 of the runner, staged iterator,
geometry, systems.py and the loaded core/libcuest/libxc libraries.
"""
import argparse
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

COMPONENTS = [f"SAPT {term} ENERGY" for term in ("ELST", "EXCH", "IND", "DISP", "TOTAL")]
WATER_DIMER = """0 1
O -0.702196054 -0.056060256 0.009942262
H -1.022193224 0.846775782 -0.011488714
H 0.257521062 0.042121496 0.005218999
--
0 1
O 2.268880784 0.026340101 0.000508029
H 2.645502399 -0.412039965 0.766632411
H 2.641145101 -0.449872874 -0.744894473
units angstrom
symmetry c1
no_reorient
no_com
"""
# Protein157 6-31+g** (13868775) scientific options, CPKS induction and fixed GRAC.
SCIENCE = {
    "basis": "6-31+g**", "scf_type": "df", "SCF_SUBTYPE": "AUTO", "reference": "rhf",
    "DF_BASIS_SCF": "def2-universal-jkfit", "DF_BASIS_MP2": "aug-cc-pvdz-ri", "PUREAM": True,
    "D_CONVERGENCE": 8, "E_CONVERGENCE": 9, "MAXITER": 150, "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
    "SAPT_DFT_FUNCTIONAL": "pbe0", "SAPT_DFT_DO_DHF": True, "SAPT_DFT_USE_EINSUMS": True,
    "SAPT_DFT_GRAC_SHIFT_A": 0.136, "SAPT_DFT_GRAC_SHIFT_B": 0.136,
    "SAPT_DFT_INDUCTION_TYPE": "CPKS", "CUEST_MIXED_PRECISION": False,
}
BACKEND = {"cpu": {"USE_CUEST": False, "CUEST_XC": False, "CUEST_SAD": False},
           "gpu": {"USE_CUEST": True, "CUEST_XC": True, "CUEST_SAD": True}}
SOLVER_ROW = re.compile(r"^\s+(Guess|\d+)\s+([0-9.eE+-]+)\*?\s+([0-9.eE+-]+)\*?\s+\d+\s*$")


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def loaded_libraries():
    """sha256 of the psi4 core, libcuest and libxc actually mapped into this process."""
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        name = os.path.basename(path)
        if name.startswith(("core.", "libcuest", "libxc")) and os.path.isfile(path):
            paths.add(path)
    return {p: sha256(Path(p).read_bytes()) for p in sorted(paths)}


def solver_histories(text):
    """Residual rows (relative norms, A<-B and B->A) of every "SAPT Coupled Induction Solver"."""
    histories = []
    for block in text.split("SAPT Coupled Induction Solver")[1:]:
        conv = float(re.search(r"Convergence\s+=\s+([0-9.eE+-]+)", block).group(1))
        rows = []
        for line in block.splitlines():
            m = SOLVER_ROW.match(line)
            if m:
                rows.append([float(m.group(2)), float(m.group(3))])
            elif rows and line.strip().startswith("---"):
                break
        histories.append({"convergence": conv, "iterations": len(rows) - 1, "residuals": rows,
                          "converged": bool(rows) and max(rows[-1]) < conv})
    return histories


def install_iterator_proofs(psi4, minimum, proofs):
    """HF.iterations = the staged scf_iterate with the final-grid minimum, recording a proof per SCF."""
    module = importlib.import_module("psi4.driver.procrouting.scf_proc.scf_iterator")
    original = module.scf_iterate

    @functools.wraps(original)
    def iterate(wfn, *args, **kwargs):
        potential = wfn.V_potential()
        switched = bool(potential and potential.initial_grid())
        ok = False
        try:
            result = original(wfn, *args, xc_final_min_iterations=minimum, **kwargs)
            ok = True
            return result
        finally:
            proof = {"switched": switched, "converged": ok, "functional": wfn.functional().name(),
                     "nbf": wfn.basisset().nbf(), "nalpha": wfn.nalpha(), "nbeta": wfn.nbeta(),
                     "iterations": wfn.iteration_}
            for field, variable in (("initial", "SCF DFT INITIAL GRID ITERATIONS"),
                                    ("final", "SCF DFT FINAL GRID ITERATIONS")):
                proof[field] = int(wfn.variable(variable)) if wfn.has_variable(variable) else None
            if switched:
                proof["final_grid_after"] = not potential.initial_grid()
            proofs.append(proof)

    psi4.core.HF.iterations = iterate
    return {"path": module.__file__, "sha256": sha256(Path(module.__file__).read_bytes())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("cpu", "gpu"), required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int)
    p.add_argument("--memory", default="200 GiB")
    p.add_argument("--initial", default="35x194")
    p.add_argument("--final", default="75x302")
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--minimum", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="water dimer with the same options")
    p.add_argument("--source-commit")
    args = p.parse_args()
    threads = args.threads or {"cpu": 24, "gpu": 8}[args.arm]
    args.output.mkdir(parents=True, exist_ok=False)

    sys.argv.append("--einsums:no-attach-debugger")  # exit on a crash, never park in a debugger wait
    import psi4
    sys.argv.pop()
    sys.path.insert(0, str(args.suite))
    from systems import SYSTEMS
    assert args.system in SYSTEMS and args.system != "pla15_3pe2"
    geometry = WATER_DIMER if args.smoke else SYSTEMS[args.system]

    os.chdir(args.output)
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_memory(args.memory)
    psi4.set_num_threads(threads)
    initial = [int(x) for x in args.initial.split("x")]
    final = [int(x) for x in args.final.split("x")]
    options = {**SCIENCE, **BACKEND[args.arm],
               "DFT_RADIAL_POINTS": final[0], "DFT_SPHERICAL_POINTS": final[1],
               "DFT_RADIAL_POINTS_INITIAL": initial[0], "DFT_SPHERICAL_POINTS_INITIAL": initial[1],
               "DFT_GRID_SWITCH_FACTOR": args.factor}
    record = {
        "arm": args.arm, "system": "water_dimer_smoke" if args.smoke else args.system,
        "suite_system": args.system, "basis": options["basis"], "method": "sapt(dft)-d4(i)",
        "geometry_sha256": sha256(geometry), "suite_systems_sha256": sha256((args.suite / "systems.py").read_bytes()),
        "runner_sha256": sha256(Path(__file__).read_bytes()), "options": options,
        "science_options_sha256": sha256(json.dumps(SCIENCE, sort_keys=True)),
        "grid_policy": {"initial": args.initial, "final": args.final, "switch_factor": args.factor,
                        "minimum_final_iterations": args.minimum},
        "threads": threads, "memory": args.memory, "psi4_module": psi4.__file__,
        "psi4_version": psi4.__version__, "source_commit": args.source_commit,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_step_id": os.getenv("SLURM_STEP_ID"),
        "hostname": os.uname().nodename, "ok": False,
    }
    proofs = []
    record["iterator"] = install_iterator_proofs(psi4, args.minimum, proofs)

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
        record["nbf"] = build(molecule, "BASIS", options["basis"]).nbf()
        record["nbf_monomer_a"] = build(molecule.extract_subsets(1), "BASIS", options["basis"]).nbf()
        record["nbf_monomer_b"] = build(molecule.extract_subsets(2), "BASIS", options["basis"]).nbf()
        record["naux_jkfit"] = build(molecule, "DF_BASIS_SCF", options["DF_BASIS_SCF"], "JKFIT",
                                     options["basis"]).nbf()
        record["naux_ri"] = build(molecule, "DF_BASIS_MP2", options["DF_BASIS_MP2"], "RIFIT",
                                  options["basis"]).nbf()
        record["natom"] = molecule.natom()
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
        record["libraries_sha256"] = loaded_libraries()
        psi4.core.close_outfile()
        text = (args.output / "psi4.out").read_text()
        record["cpks_solvers"] = solver_histories(text)
        record["gpu_builder_seen"] = "cuESTJK: GPU-Accelerated" in text
        record["cpu_builder_seen"] = any(s in text for s in ("MemDFJK: Density-Fitted", "DiskDFJK: Density-Fitted"))
        record["disk_df_builder_seen"] = "DiskDFJK: Density-Fitted" in text
        record["direct_seen"] = "DirectJK" in text
        record["sad_gpu_builder_seen"] = "SAD J/K backend: cuESTJK" in text
        record["proofs"] = proofs
        switched = [q for q in proofs if q["switched"]]
        problems = []
        if args.arm == "gpu":
            if not record["gpu_builder_seen"] or record["cpu_builder_seen"] or record["direct_seen"]:
                problems.append("J/K backend is not cuEST DF-JK only")
            if not record["sad_gpu_builder_seen"]:
                problems.append("GPU SAD not observed")
            if not any("libcuest" in k for k in record["libraries_sha256"]):
                problems.append("libcuest not loaded")
        elif record["gpu_builder_seen"] or record["sad_gpu_builder_seen"] or not record["cpu_builder_seen"]:
            problems.append("J/K backend is not CPU DF")
        if not switched:
            problems.append("no SCF switched grids")
        if any(not q["converged"] or q["final"] < args.minimum or not q["final_grid_after"] or q["initial"] < 1
               for q in switched):
            problems.append("a switched SCF violated the grid policy")
        if not cpks or len(record["cpks_solvers"]) != len(cpks):
            problems.append("CPKS solves not all timed and parsed")
        if not all(s["converged"] for s in record["cpks_solvers"]):
            problems.append("a CPKS solve did not reach its convergence threshold")
        if {round(v, 6) for v in record["grac_shifts_hartree"].values()} != {0.136}:
            problems.append("GRAC shifts are not the fixed 0.136")
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
    print(json.dumps({k: record.get(k) for k in ("arm", "system", "ok", "error", "wall_s", "cpks_solve_s", "nbf",
                                                  "naux_jkfit", "components_hartree", "policy_checks",
                                                  "ru_maxrss_gib")}), flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
