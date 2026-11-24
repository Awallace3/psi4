"""Export one prepared CDK2 SPARCLE-QC system for d4i_timing.py."""
import argparse
import json
import math
from pathlib import Path


def export(system, destination):
    record = json.loads((system / "qca.json").read_text())
    meta = record["meta"]
    dimer, ligand, protein = (record[key]["molecule"] for key in ("cx", "lig", "prot"))
    boundary = meta["n_qm_ligand"]
    atoms = [line for line in (system / "sparcle/input/CAPPED_qm.pdb").read_text().splitlines()
             if line.startswith(("ATOM  ", "HETATM"))]
    if not 0 < boundary < len(atoms) or len(atoms) != len(dimer["symbols"]):
        raise ValueError("Capped PDB and fragment sizes disagree")
    if ligand["molecular_charge"] + protein["molecular_charge"] != dimer["molecular_charge"]:
        raise ValueError("Fragment charges do not sum to the complex charge")
    geometry = []
    groups = {"A": {}, "B": {}}
    for i, line in enumerate(atoms):
        on_a = i < boundary
        if ligand["real"][i] != on_a or protein["real"][i] == on_a or not dimer["real"][i]:
            raise ValueError(f"Inconsistent counterpoise atom masks at atom {i + 1}")
        if i in (0, boundary):
            mol = ligand if on_a else protein
            if i:
                geometry.append("--")
            geometry.append(f'{int(mol["molecular_charge"])} {mol["molecular_multiplicity"]}')
        symbol = line[76:78].strip().capitalize()
        xyz = [line[j:j + 8].strip() for j in (30, 38, 46)]
        # qca.json stores positions in bohr. Retain the original PDB angstrom precision.
        reference = dimer["geometry"][3*i:3*i+3]
        if symbol != dimer["symbols"][i] or any(abs(float(v) - b * 0.52917721067) > 1e-5
                                                 for v, b in zip(xyz, reference)):
            raise ValueError(f"PDB element or coordinates differ from qca.json at atom {i + 1}")
        geometry.append(" ".join([symbol, *xyz]))
        label = " ".join(x.strip() for x in (line[21], line[17:20], line[22:27]))
        groups["A" if on_a else "B"].setdefault(label, []).append(i + 1)
    field = record["cx"]["external_potential"]
    if (not field or field != record["prot"]["external_potential"] or len(field) != meta["n_mm_charges"]
            or any(len(row) != 4 or not all(map(math.isfinite, row)) for row in field)):
        raise ValueError("Invalid B-owned MM field")
    geometry.extend(("units angstrom", "no_reorient", "no_com", "symmetry c1"))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "dimer.geometry").write_text("\n".join(geometry) + "\n")
    for name, payload in (("field", {"B": field}), ("groups", groups), ("meta", meta)):
        (destination / f"{name}.json").write_text(json.dumps(payload, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", type=Path, help="Directory with qca.json and sparcle/input/CAPPED_qm.pdb")
    parser.add_argument("destination", type=Path, help="New input directory")
    args = parser.parse_args()
    export(args.system, args.destination)
