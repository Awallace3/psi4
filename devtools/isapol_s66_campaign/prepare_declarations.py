"""Persist explicit geometry-only native inputs; no SCF or server connection."""
import argparse
import hashlib
import json
from pathlib import Path

from declarations import declare_geometry


def geometry_digest(monomer):
    # Match prepare_entries.py; formula/name alone do not bind atom order.
    molecule = dict(
        schema_name='qcschema_molecule', schema_version=2,
        symbols=monomer['symbols'],
        geometry=[value for xyz in monomer['geometry_bohr'] for value in xyz],
        molecular_charge=monomer['charge'], molecular_multiplicity=monomer['multiplicity'],
        fix_com=True, fix_orientation=True, fix_symmetry='c1')
    return hashlib.sha256(json.dumps(molecule, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare(inventory):
    rows = []
    for monomer in inventory['monomers']:
        if monomer['charge'] != 0 or monomer['multiplicity'] != 1:
            raise ValueError('profile requires neutral singlets')
        declaration, policy = declare_geometry(monomer['symbols'], monomer['geometry_bohr'])
        rows.append(dict(
            monomer_id=monomer['id'], formula=monomer['formula'],
            geometry_sha256=geometry_digest(monomer),
            declaration=declaration, policy=policy))
    if len({row['monomer_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate monomer ids')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    validation = json.loads(args.inventory.with_name('inventory-validation.json').read_text())
    if validation['status'] != 'PASS' or validation['inventory_sha256'] != digest:
        raise ValueError('qualified exact inventory required')
    inventory = json.loads(raw)
    rows = prepare(inventory)
    if len(rows) != 131:
        raise ValueError('complete131-monomer inventory required')
    payload = dict(
        schema='native-s66x8-molecular-declaration-preparation/1',
        status='PREPARED_NOT_EXECUTED_OR_SUBMITTED',
        inventory_sha256=digest, source_pins=inventory['source_pins'],
        policy_source_sha256=hashlib.sha256(
            Path(__file__).with_name('declarations.py').read_bytes()).hexdigest(),
        queue_policy='EMBERS_ONLY_NO_INFERNO', monomers=rows,
        qualification='Input preparation only; no installed molecular, QCSchema, QCArchive artifact-roundtrip or accuracy qualification.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(
        status=payload['status'], monomers=len(rows),
        max_independent_parameters=max(row['policy']['independent_parameters'] for row in rows),
        lab_axis_fallback_monomers=[
            row['monomer_id'] for row in rows if row['policy']['lab_axis_fallback_sites']],
        output=str(args.output),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
