#!/usr/bin/env python3
"""Generate SAPT0/aug-cc-pVDZ references for the rigid S22 water-dimer PES.

Run under the built Psi4 environment. Individual distances can be evaluated in parallel,
then merged deterministically::

    seq ... | xargs ... python -P sapt0_pes.py --distance
    python -P sapt0_pes.py --combine

All reported energies use Psi4's SAPT0 component totals: SAPT0 IND ENERGY includes
exchange-induction and delta-HF; SAPT0 DISP ENERGY includes exchange-dispersion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import psi4

HARTREE_TO_KCAL = 627.5094740631
DISTANCES = [2.5, 2.6, 2.7, 2.8, 2.912, 3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0,
             4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
ELEMENTS = ("O", "H", "H", "O", "H", "H")
S22_WATER_DIMER_ANGSTROM = np.array([
    [-1.551007, -0.114520, 0.000000],
    [-1.934259,  0.762503, 0.000000],
    [-0.599677,  0.040712, 0.000000],
    [ 1.350625,  0.111469, 0.000000],
    [ 1.680398, -0.373741, -0.758561],
    [ 1.680398, -0.373741,  0.758561],
])
HERE = Path(__file__).resolve().parent
WORK = HERE / "sapt0-water.work"
OUT = HERE / "sapt0-water.tsv"


def geometry(distance: float) -> np.ndarray:
    xyz = S22_WATER_DIMER_ANGSTROM.copy()
    axis = xyz[3] - xyz[0]
    xyz[3:] += (distance / np.linalg.norm(axis) - 1.0) * axis
    return xyz


def molecule(distance: float):
    xyz = geometry(distance)
    lines = ["0 1"]
    lines += [f"{e} {x:.12f} {y:.12f} {z:.12f}" for e, (x, y, z) in zip(ELEMENTS[:3], xyz[:3])]
    lines += ["--", "0 1"]
    lines += [f"{e} {x:.12f} {y:.12f} {z:.12f}" for e, (x, y, z) in zip(ELEMENTS[3:], xyz[3:])]
    lines += ["units angstrom", "symmetry c1", "no_com", "no_reorient"]
    return psi4.geometry("\n".join(lines))


def evaluate(distance: float, threads: int, memory: str) -> dict[str, float]:
    WORK.mkdir(exist_ok=True)
    tag = str(distance).replace(".", "p")
    psi4.set_num_threads(threads)
    psi4.set_memory(memory)
    psi4.core.set_output_file(str(WORK / f"R-{tag}.out"), False)
    psi4.set_options({
        "basis": "aug-cc-pvdz",
        "scf_type": "df",
        "freeze_core": True,
        "e_convergence": 1.0e-8,
        "d_convergence": 1.0e-8,
    })
    psi4.energy("sapt0", molecule=molecule(distance))
    ind_h = float(psi4.core.variable("SAPT0 IND ENERGY"))
    disp_h = float(psi4.core.variable("SAPT0 DISP ENERGY"))
    row = {
        "distance": distance,
        "ind_hartree": ind_h,
        "disp_hartree": disp_h,
        "ind_kcal_mol": ind_h * HARTREE_TO_KCAL,
        "disp_kcal_mol": disp_h * HARTREE_TO_KCAL,
    }
    (WORK / f"R-{tag}.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def combine() -> list[dict[str, float]]:
    rows = []
    for distance in DISTANCES:
        tag = str(distance).replace(".", "p")
        path = WORK / f"R-{tag}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        rows.append(json.loads(path.read_text()))
    header = "distance\tind_hartree\tdisp_hartree\tind_kcal_mol\tdisp_kcal_mol"
    lines = [
        "# method: SAPT0/aug-cc-pVDZ",
        f"# psi4_version: {psi4.__version__}",
        "# induction: SAPT0 IND ENERGY",
        "# dispersion: SAPT0 DISP ENERGY",
        "# system: S22 #2 rigid water dimer; O-O translation at fixed orientation",
        header,
    ]
    for row in rows:
        lines.append("\t".join(f"{row[key]:.16g}" for key in header.split("\t")))
    OUT.write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=float)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory", default="8 GB")
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.combine:
        rows = combine()
        print(f"wrote {OUT} ({len(rows)} points)")
    elif args.distance is not None:
        print(json.dumps(evaluate(args.distance, args.threads, args.memory)))
    else:
        for distance in DISTANCES:
            print(json.dumps(evaluate(distance, args.threads, args.memory)))
        combine()


if __name__ == "__main__":
    main()
