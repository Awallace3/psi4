#!/usr/bin/env python
"""Time ExternalPotential.computePotentialMatrix for point charges, CPU (Libint) vs cuEST.

    python extern_potential_timing.py --suite GEOMETRIES_DIR --output OUT.json \
        [--system peptide nanotube] [--basis aug-cc-pvdz] [--ncharge 1000 10000] [--threads 8]

Charges are random (seeded) in a box 15 bohr beyond the molecule. Each cell reports the best of
--repeat calls; the first cuEST call per basis also pays the basis/plan setup and is reported apart.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--system", nargs="+", default=["peptide", "nanotube"])
    p.add_argument("--basis", default="aug-cc-pvdz")
    p.add_argument("--ncharge", type=int, nargs="+", default=[1000, 10000])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sys.argv.append("--einsums:no-attach-debugger")
    import psi4
    sys.argv.pop()
    sys.path.insert(0, str(args.suite))
    from systems import SYSTEMS
    psi4.set_num_threads(args.threads)
    psi4.core.set_output_file(str(args.output.with_suffix(".out")), False)
    rows = []
    for system in args.system:
        assert system in SYSTEMS and system != "pla15_3pe2"
        mol = psi4.geometry(SYSTEMS[system])
        mol.update_geometry()
        basis = psi4.core.BasisSet.build(mol, "ORBITAL", args.basis)
        xyz = mol.geometry().np
        lo, hi = xyz.min(axis=0) - 15., xyz.max(axis=0) + 15.
        for n in args.ncharge:
            rng = np.random.default_rng(n)
            pos = rng.uniform(lo, hi, size=(n, 3))
            q = rng.normal(size=n)
            extern = psi4.core.ExternalPotential()
            for c, (x, y, z) in zip(q, pos):
                extern.addCharge(float(c), float(x), float(y), float(z))
            result = {}
            for route in ("cpu", "cuest"):
                psi4.set_options({"use_cuest": route == "cuest"})
                times = []
                for _ in range(args.repeat + (route == "cuest")):
                    t = time.perf_counter()
                    V = extern.computePotentialMatrix(basis).np.copy()
                    times.append(time.perf_counter() - t)
                result[route] = V
                if route == "cuest":
                    rows_first = times.pop(0)
                result[route + "_s"] = min(times)
            psi4.set_options({"use_cuest": False})
            row = {"system": system, "basis": args.basis, "nbf": basis.nbf(), "ncharge": n,
                   "cpu_s": result["cpu_s"], "cuest_s": result["cuest_s"], "cuest_first_call_s": rows_first,
                   "speedup": result["cpu_s"] / result["cuest_s"],
                   "max_abs_diff": float(np.abs(result["cpu"] - result["cuest"]).max()),
                   "max_abs_V": float(np.abs(result["cpu"]).max())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"threads": args.threads, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
