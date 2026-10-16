#!/usr/bin/env python
"""GPU memory held by a cuEST-grid DFT potential (XC integral plan), per basis.

    python cuest_xc_plan_memory_probe.py --suite GEOMETRIES_DIR --system protein157 --output OUT.json \
        [--basis aug-cc-pvdz heavy-aug-cc-pvdz] [--grid 75x302] [--functional pbe0]

Protein157 aug-cc-pVDZ SAPT(DFT) stopped at the first SCF's cuEST J/K allocation (83.7 GiB
requested, 12.2 of 139.8 GiB free), the same with 102 and 47 GiB of Psi4 memory. The DFT
potential is the only GPU object built before J/K. This reads the device's used memory
(nvidia-smi, this process is the only GPU user in the allocation) before V.initialize(),
after it, and after V.finalize(), for each basis.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split(",")
    return int(out[0]), int(out[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--basis", nargs="+", default=["heavy-aug-cc-pvdz", "aug-cc-pvdz"])
    p.add_argument("--grid", default="75x302")
    p.add_argument("--functional", default="pbe0")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sys.argv.append("--einsums:no-attach-debugger")
    import psi4
    sys.argv.pop()
    sys.path.insert(0, str(args.suite))
    from systems import SYSTEMS
    assert args.system in SYSTEMS and args.system != "pla15_3pe2"
    psi4.core.set_output_file(str(args.output.with_suffix(".out")), False)
    psi4.set_num_threads(8)
    radial, angular = (int(x) for x in args.grid.split("x"))
    molecule = psi4.geometry(SYSTEMS[args.system])
    molecule.update_geometry()
    rows = []
    for basis in args.basis:
        psi4.set_options({"basis": basis, "use_cuest": True, "cuest_xc": True, "cuest_mixed_precision": False,
                          "dft_radial_points": radial, "dft_spherical_points": angular})
        bs = psi4.core.BasisSet.build(molecule, "BASIS", basis)
        functional = psi4.driver.dft.build_superfunctional(args.functional, True)[0]
        before, total = used_mib()
        V = psi4.core.VBase.build(bs, functional, "RV")
        t = time.perf_counter()
        V.initialize()
        init_s = time.perf_counter() - t
        after, _ = used_mib()
        assert V.uses_cuest_xc()
        V.finalize()
        freed, _ = used_mib()
        row = {"basis": basis, "nbf": bs.nbf(), "grid": args.grid, "used_before_mib": before,
               "used_after_initialize_mib": after, "used_after_finalize_mib": freed,
               "xc_plan_mib": after - before, "initialize_s": init_s, "device_total_mib": total}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del V
    args.output.write_text(json.dumps({"system": args.system, "functional": args.functional, "rows": rows},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
