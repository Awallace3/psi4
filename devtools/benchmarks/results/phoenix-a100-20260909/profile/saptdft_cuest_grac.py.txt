#!/usr/bin/env python3
"""Matched, fresh-process SAPT(DFT)-D4(I) CPU/cuEST benchmarks.

Run under the branch's PsiAPI environment, inside a GPU allocation:
  python saptdft_cuest_grac.py --output RUN --repeats 3
Each calculation gets a fresh process; energy() wall time includes backend
initialization but excludes Python import and molecule/basis construction.
The idealized benzene geometry is NOT the published S22 geometry. Fixed GRAC
shifts exercise the correction, not an ab initio ionization-potential model.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

COMPONENTS = [f"SAPT {term} ENERGY" for term in ("ELST", "EXCH", "IND", "DISP", "TOTAL")]
WATER = """0 1
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


def geometry(system):
    if system in ("peptide", "nanotube", "protein157"):
        return json.loads(Path(__file__).with_name("saptdft_suite_geometries.json").read_text())[system]
    if system == "water":
        return WATER
    blocks = []
    for dx, dz in [(0.0, 0.0), (1.6, 3.4)]:
        atoms = ["0 1"]
        for i in range(6):
            angle = math.pi * i / 3
            for element, radius in [("C", 1.3915), ("H", 2.4715)]:
                atoms.append(f"{element} {radius * math.cos(angle) + dx:.8f} "
                             f"{radius * math.sin(angle):.8f} {dz:.8f}")
        blocks.append("\n".join(atoms))
    return "\n--\n".join(blocks) + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com\n"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_case(args):
    import psi4

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {"system": args.system, "basis": args.basis, "mode": args.mode,
              "threads": args.threads, "shift_hartree": args.shift,
              "psi4_version": psi4.__version__, "psi4_module": psi4.__file__,
              "slurm_job_id": os.getenv("SLURM_JOB_ID"),
              "slurm_step_id": os.getenv("SLURM_STEP_ID"), "ok": False}
    os.chdir(output)  # Each process owns its timer.dat; Psi4 appends at exit.
    psi4.core.set_output_file(str(output / "psi4.out"), False)
    psi4.set_memory(args.memory)
    record["memory"] = args.memory
    psi4.set_num_threads(args.threads)
    options = {
        "basis": args.basis, "scf_type": "df", "reference": "rhf",
        "SAPT_DFT_FUNCTIONAL": "pbe0", "SAPT_DFT_GRAC_SHIFT_A": args.shift,
        "SAPT_DFT_GRAC_SHIFT_B": args.shift, "SAPT_DFT_GRAC_COMPUTE": "NONE",
        "SAPT_DFT_INDUCTION_TYPE": "NONE", "SAPT_DFT_DO_DHF": True,
        "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL", "SAPT_DFT_USE_EINSUMS": True,
        "USE_CUEST": args.mode == "gpu", "CUEST_XC": not args.cpu_xc,
        "CUEST_MIXED_PRECISION": args.mixed_precision, "E_CONVERGENCE": 9, "D_CONVERGENCE": 8,
        "DFT_RADIAL_POINTS": args.radial_points, "DFT_SPHERICAL_POINTS": args.spherical_points, "MAXITER": 150,
    }
    if args.system in ("peptide", "nanotube", "protein157"):
        # Pople's generated auxiliary basis is Cartesian, unsupported by cuEST.
        # Psi4 propagates the primary basis's puream into fitting bases here.
        # Explicitly use spherical orbital AND fitting bases in both arms;
        # these counts differ from the timing suite's Cartesian RHF references.
        options.update({"PUREAM": True, "DF_BASIS_SCF": "def2-universal-jkfit",
                        "DF_BASIS_MP2": "aug-cc-pvdz-ri"})
    record["options"] = options
    record["geometry"] = geometry(args.system)
    start = None
    try:
        molecule = psi4.geometry(record["geometry"])
        psi4.set_options(options)
        record["nbf"] = psi4.core.BasisSet.build(molecule, "BASIS", args.basis).nbf()
        record["nbf_monomer_a"] = psi4.core.BasisSet.build(molecule.extract_subsets(1), "BASIS", args.basis).nbf()
        record["nbf_monomer_b"] = psi4.core.BasisSet.build(molecule.extract_subsets(2), "BASIS", args.basis).nbf()
        expected_nbf = {("peptide", "6-31+g**"): 250, ("nanotube", "6-31+g**"): 548,
                        ("protein157", "6-31+g**"): 1786}
        expected = expected_nbf.get((args.system, args.basis.lower()))
        if expected is not None and record["nbf"] != expected:
            raise ValueError(f"Basis count {record['nbf']} differs from suite reference {expected}")
        start = time.perf_counter()
        energy = psi4.energy("sapt(dft)-d4(i)", molecule=molecule)
        record["wall_s"] = time.perf_counter() - start
        record["returned_energy_hartree"] = energy
        record["components_hartree"] = {key: float(psi4.variable(key)) for key in COMPONENTS}
        psi4.core.close_outfile()
        text = (output / "psi4.out").read_text()
        gpu_builder = "cuESTJK: GPU-Accelerated" in text
        cpu_builder = any(s in text for s in ("MemDFJK: Density-Fitted", "DiskDFJK: Density-Fitted"))
        record["gpu_builder_seen"] = gpu_builder
        record["cpu_builder_seen"] = cpu_builder
        if args.mode == "gpu" and (not gpu_builder or cpu_builder):
            raise RuntimeError("GPU backend missing or CPU J/K fallback detected")
        if args.mode == "cpu" and gpu_builder:
            raise RuntimeError("CPU baseline used cuEST")
        record["ok"] = True
    except Exception as exc:
        if start is not None:
            record.setdefault("wall_s", time.perf_counter() - start)
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        psi4.core.close_outfile()
        psi4.core.clean()
        atomic_json(output / "result.json", record)
    print(json.dumps(record), flush=True)
    return 0 if record["ok"] else 1


def campaign(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    manifest = {"command": sys.argv, "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "repeats": args.repeats, "threads": args.threads,
                "timing": "fresh-process energy() wall time, including backend initialization",
                "accuracy_tolerance_hartree": 1e-6, "records": []}
    atomic_json(output / "campaign.json", manifest)
    failed = False
    cases = [("water", ["cc-pvdz", "aug-cc-pvdz"]),
             ("benzene", ["cc-pvdz", "aug-cc-pvdz"]),
             ("peptide", ["6-31+g**"]), ("nanotube", ["6-31+g**"])]
    cases = [(system, bases) for system, bases in cases if system in args.systems]
    manifest["cases"] = cases
    manifest["geometry_sha256"] = hashlib.sha256(script.with_name("saptdft_suite_geometries.json").read_bytes()).hexdigest()
    atomic_json(output / "campaign.json", manifest)
    for system, bases in cases:
        for basis in bases:
            for repeat in range(args.repeats):
                for mode in (("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")):
                    name = f"{system}-{basis}-{mode}-{repeat + 1}"
                    directory = output / name
                    directory.mkdir()
                    command = [sys.executable, str(script), "--case", "--system", system,
                               "--basis", basis, "--mode", mode, "--output", str(directory),
                               "--threads", str(args.threads), "--memory", args.memory, "--shift", str(args.shift),
                               "--radial-points", str(args.radial_points),
                               "--spherical-points", str(args.spherical_points)]
                    if args.cpu_xc:
                        command.append("--cpu-xc")
                    if args.mixed_precision:
                        command.append("--mixed-precision")
                    print(f"START {name}", flush=True)
                    started = time.perf_counter()
                    try:
                        with (directory / "console.log").open("w") as log:
                            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                    timeout=args.case_timeout).returncode
                    except subprocess.TimeoutExpired:
                        status = 124
                    entry = {"name": name, "returncode": status,
                             "process_wall_s": time.perf_counter() - started}
                    manifest["records"].append(entry)
                    atomic_json(output / "campaign.json", manifest)
                    print(f"END {name} rc={status} elapsed={entry['process_wall_s']:.2f}s", flush=True)
                    failed |= status != 0
                    if status != 0:
                        # Preserve the failure, avoid spending the allocation on repeated failures.
                        return 1
    atomic_json(output / "COMPLETE.json", {"ok": not failed, "count": len(manifest["records"])})
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", default="24 GiB")
    parser.add_argument("--shift", type=float, default=0.136)
    parser.add_argument("--case-timeout", type=int, default=1200)
    parser.add_argument("--cpu-xc", action="store_true", help="Diagnostic: GPU J/K with CPU XC")
    parser.add_argument("--mixed-precision", action="store_true", help="Diagnostic: allow cuEST emulated mixed precision")
    parser.add_argument("--allow-protein157-cpu", action="store_true",
                        help="Explicit opt-in for a separately allocated protein157 CPU baseline")
    parser.add_argument("--radial-points", type=int, default=99)
    parser.add_argument("--spherical-points", type=int, default=590)
    parser.add_argument("--systems", nargs="+", choices=["water", "benzene", "peptide", "nanotube"],
                        default=["water", "benzene", "peptide", "nanotube"])
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--system", choices=["water", "benzene", "peptide", "nanotube", "protein157"], default="water")
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    if args.system == "protein157":
        if not args.case:
            parser.error("protein157 requires a separately allocated explicit --case")
        if args.mode == "cpu" and not args.allow_protein157_cpu:
            parser.error("protein157 CPU requires --allow-protein157-cpu and sufficient resources")
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    return run_case(args) if args.case else campaign(args)


if __name__ == "__main__":
    sys.exit(main())
