#!/usr/bin/env python
"""GPU memory held at each stage of a cuEST J/K object, per basis.

    python cuest_jk_memory_probe.py --suite GEOMETRIES_DIR --system protein157 --output OUT.json \
        [--basis aug-cc-pvdz heavy-aug-cc-pvdz] [--memory "100 GiB"]

Protein157 aug-cc-pVDZ stopped at the first SCF's cuEST J/K allocation (83.7 GiB requested,
12.2 of 139.8 GiB free); the XC plan was ruled out (<1 GiB). This reads device used memory
(nvidia-smi; the only GPU user in the allocation) after the SCF basis/aux build, JK.build,
JK.initialize and one compute with an occupied-like C, for each basis. A failure is recorded
with the memory at that point.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--basis", nargs="+", default=["heavy-aug-cc-pvdz", "aug-cc-pvdz"])
    p.add_argument("--memory", default="100 GiB")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sys.argv.append("--einsums:no-attach-debugger")
    import psi4
    sys.argv.pop()
    sys.path.insert(0, str(args.suite))
    from systems import SYSTEMS
    assert args.system in SYSTEMS and args.system != "pla15_3pe2"
    psi4.core.set_output_file(str(args.output.with_suffix(".out")), False)
    psi4.set_memory(args.memory)
    psi4.set_num_threads(8)
    molecule = psi4.geometry(SYSTEMS[args.system])
    molecule.update_geometry()
    rows = []
    for basis in args.basis:
        psi4.set_options({"basis": basis, "scf_type": "df", "use_cuest": True, "cuest_mixed_precision": False})
        row = {"basis": basis, "stages_mib": {"start": used_mib()}}
        t0 = time.perf_counter()
        try:
            primary = psi4.core.BasisSet.build(molecule, "ORBITAL", basis)
            aux = psi4.core.BasisSet.build(molecule, "DF_BASIS_SCF", "", "JKFIT", basis)
            row.update(nbf=primary.nbf(), naux=aux.nbf())
            row["stages_mib"]["bases"] = used_mib()
            jk = psi4.core.JK.build(primary, aux)
            row["jk_name"] = jk.name()
            row["stages_mib"]["jk_build"] = used_mib()
            jk.set_memory(int(psi4.get_memory() * 0.8 / 8))
            jk.initialize()
            row["stages_mib"]["jk_initialize"] = used_mib()
            C = psi4.core.Matrix.from_array(np.linalg.qr(np.random.default_rng(1).normal(
                size=(primary.nbf(), 300)))[0])
            jk.C_left_add(C)
            jk.compute()
            row["stages_mib"]["jk_compute"] = used_mib()
            jk.finalize()
            del jk
            row["stages_mib"]["jk_deleted"] = used_mib()
            row["ok"] = True
        except Exception as exc:
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}"
            row["stages_mib"]["at_error"] = used_mib()
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps({"system": args.system, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
