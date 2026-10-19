#!/usr/bin/env python
"""One fresh-process SAPT(DFT)-D4(I) energy under an XC grid-switching policy, on cuEST.

    python protein157_grid_policy_run.py --reference RESULT.json --output DIR [--smoke]

The workload (geometry, basis, options) is read from a reference result.json written by
devtools/benchmarks/saptdft_cuest_grac.py (Protein157 6-31+g**: CPU job 13868775), then
changed only where the GPU policy requires it; every changed key is recorded. --smoke runs
the same options on a water dimer first, to check the route before the long run.

The final-grid minimum is the Python-only scf_iterate(..., xc_final_min_iterations=N) guard,
loaded from --iterator (this branch's scf_iterator.py) over the staged driver, so the pinned
build is untouched; its sha256 is recorded. Timing is energy() wall time only: no memory
sampler, no microprofiling.
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def install_iterator(psi4, source, minimum, proofs):
    """HF.iterations = this branch's scf_iterate with the final-grid minimum, recording proofs."""
    name = "psi4.driver.procrouting.scf_proc._grid_policy_iterator"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @functools.wraps(module.scf_iterate)
    def iterate(wfn, *args, **kwargs):
        potential = wfn.V_potential()
        switched = bool(potential and potential.initial_grid())
        ok = False
        try:
            result = module.scf_iterate(wfn, *args, xc_final_min_iterations=minimum, **kwargs)
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
    return sha256(Path(source).read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterator", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu-smoke", action="store_true", help="script check only: no cuEST, no backend checks")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory", default="100 GiB")
    p.add_argument("--initial", default="35x194")
    p.add_argument("--final", default="75x302")
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--minimum", type=int, default=2)
    p.add_argument("--source-commit")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import psi4
    os.chdir(args.output)
    psi4.core.set_output_file(str(args.output / "psi4.out"), False)
    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)

    ref = json.loads(args.reference.read_text())
    initial = [int(x) for x in args.initial.split("x")]
    final = [int(x) for x in args.final.split("x")]
    options = dict(ref["options"])
    options.update({
        "USE_CUEST": True, "CUEST_XC": True, "CUEST_SAD": True, "CUEST_MIXED_PRECISION": False,
        "DFT_RADIAL_POINTS": final[0], "DFT_SPHERICAL_POINTS": final[1],
        "DFT_RADIAL_POINTS_INITIAL": initial[0], "DFT_SPHERICAL_POINTS_INITIAL": initial[1],
        "DFT_GRID_SWITCH_FACTOR": args.factor,
    })
    if args.cpu_smoke:
        args.smoke = True
        options.update({"USE_CUEST": False, "CUEST_XC": False, "CUEST_SAD": False})
    assert options["scf_type"] == "df" and options["SAPT_DFT_INDUCTION_TYPE"] == "NONE"
    assert options["SAPT_DFT_GRAC_COMPUTE"] == "ITERATIVE" and not options["SAPT_DFT_GRAC_SEED_CATION"]
    changed = {k: {"reference": ref["options"].get(k), "run": v}
               for k, v in options.items() if ref["options"].get(k) != v}
    geometry = WATER_DIMER if args.smoke else ref["geometry"]
    record = {
        "system": "water_dimer_smoke" if args.smoke else ref["system"], "basis": options["basis"],
        "reference_result_sha256": sha256(args.reference.read_bytes()),
        "reference_job": ref.get("slurm_job_id"), "reference_wall_s": ref.get("wall_s"),
        "geometry_sha256": sha256(geometry), "options": options, "options_changed_vs_reference": changed,
        "grid_policy": {"initial": args.initial, "final": args.final, "switch_factor": args.factor,
                        "minimum_final_iterations": args.minimum},
        "threads": args.threads, "memory": args.memory, "psi4_module": psi4.__file__,
        "source_commit": args.source_commit, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_step_id": os.getenv("SLURM_STEP_ID"), "ok": False,
    }
    proofs = []
    record["iterator_overlay"] = {"path": str(args.iterator),
                                  "sha256": install_iterator(psi4, args.iterator, args.minimum, proofs)}
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
        if not args.smoke:
            assert record["geometry_sha256"] == sha256(ref["geometry"])
            assert record["nbf"] == ref["nbf"], (record["nbf"], ref["nbf"])
        psi4.core.clean_timers()
        start = time.perf_counter()
        energy = psi4.energy("sapt(dft)-d4(i)", molecule=molecule)
        record["wall_s"] = time.perf_counter() - start
        record["timer_records"] = psi4.core.get_timer_records()
        record["returned_energy_hartree"] = energy
        record["components_hartree"] = {k: float(psi4.variable(k)) for k in COMPONENTS}
        record["grac_shifts_hartree"] = {s: float(psi4.variable(f"SAPT DFT GRAC SHIFT {s}")) for s in "AB"}
        if not args.smoke:
            record["vs_reference_hartree"] = {k: record["components_hartree"][k] - ref["components_hartree"][k]
                                              for k in COMPONENTS}
            record["grac_shift_diff_vs_reference"] = {
                s: record["grac_shifts_hartree"][s] - ref["grac_shifts_hartree"][s] for s in "AB"}
            record["speedup_vs_reference_wall"] = ref["wall_s"] / record["wall_s"]
        psi4.core.close_outfile()
        text = (args.output / "psi4.out").read_text()
        record["gpu_builder_seen"] = "cuESTJK: GPU-Accelerated" in text
        record["cpu_builder_seen"] = any(s in text for s in ("MemDFJK: Density-Fitted", "DiskDFJK: Density-Fitted"))
        record["direct_seen"] = "DirectJK" in text
        record["sad_gpu_builder_seen"] = "SAD J/K backend: cuESTJK" in text
        record["proofs"] = proofs
        switched = [q for q in proofs if q["switched"]]
        problems = []
        if not args.cpu_smoke and (not record["gpu_builder_seen"] or record["cpu_builder_seen"]
                                   or record["direct_seen"]):
            problems.append("J/K backend is not cuEST DF-JK only")
        if not args.cpu_smoke and not record["sad_gpu_builder_seen"]:
            problems.append("GPU SAD not observed")
        if not switched:
            problems.append("no SCF switched grids")
        if any(not q["converged"] or q["final"] < args.minimum or not q["final_grid_after"] or q["initial"] < 1
               for q in switched):
            problems.append("a switched SCF violated the grid policy")
        record["policy_checks"] = {"switched_scfs": len(switched), "problems": problems}
        record["ok"] = not problems
        if problems:
            record["error"] = "; ".join(problems)
    except Exception as exc:
        if start is not None:
            record.setdefault("wall_s", time.perf_counter() - start)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["proofs"] = proofs
    finally:
        psi4.core.close_outfile()
        (args.output / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: record.get(k) for k in ("system", "ok", "error", "wall_s", "components_hartree",
                                                  "grac_shifts_hartree", "policy_checks")}), flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
