#!/usr/bin/env python3
"""CPU-only, local replay: isolate geometry-unit conversion in saved spectra.

Uses the same PySCF/basis implementation for both probes, changing only atomic
positions. Does not change the benchmark gate or run SCF/CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyscf import gto
from protocol import molecular_geometry, capture_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True,
                        help="Case containing cc-pvtz/campaign and aug-cc-pvtz/campaign")
    args = parser.parse_args()
    lines = args.xyz.read_text().splitlines()[2:]
    atom = "\n".join(lines)
    config = dict(geometry_angstrom_per_bohr=0.52917721067)
    common_bohr, common_unit = molecular_geometry(atom, config)
    reports = []
    for basis in ("cc-pvtz", "aug-cc-pvtz"):
        base = args.campaign_root / basis / "campaign/results"
        saved = {}
        for engine in ("psi4", "pyscf"):
            path = base / f"{engine}-audit/overlap-spectrum.npz"
            info = json.loads(path.with_suffix(".json").read_text())
            if hashlib.sha256(args.xyz.read_bytes()).hexdigest() != info["geometry_sha256"]:
                raise ValueError("Geometry differs from saved spectrum provenance")
            with np.load(path, allow_pickle=False) as archive:
                saved[engine] = archive["eigenvalues"]
        report = dict(basis=basis)
        for label, atoms, unit in (("native_pyscf", atom, "Angstrom"),
                                   ("psi4_bohr_conversion", common_bohr, common_unit)):
            mol = gto.M(atom=atoms, unit=unit, basis=basis, cart=False,
                        charge=0, spin=0, verbose=0)
            spectrum = np.linalg.eigvalsh(mol.intor_symmetric("int1e_ovlp"))
            if label == "psi4_bohr_conversion":
                record = dict(config=config)
                capture_geometry(mol.atom_coords(unit="Bohr"), atom, record)
                report["live_bohr_coordinates_sha256"] = record["geometry_bohr_sha256"]
            report[label] = {engine: dict(
                maximum_absolute_difference=float(np.max(np.abs(spectrum - values))),
                original_spectrum_gate_passes=bool(np.allclose(
                    spectrum, values, atol=1e-9, rtol=1e-10)))
                for engine, values in saved.items()}
        reports.append(report)
    print(json.dumps(reports, indent=2), flush=True)
    # First reproduce the installed PySCF spectrum; then remove *only* its
    # geometry-conversion difference and demand the unchanged Psi4 audit gate.
    return 0 if all(
        r["native_pyscf"]["pyscf"]["maximum_absolute_difference"] < 1e-11 and
        r["psi4_bohr_conversion"]["psi4"]["original_spectrum_gate_passes"]
        for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
