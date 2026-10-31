"""Prepare JSON singlepoint entries; never connect to or submit to QCArchive.

The user's SAPT-reference metadata is deliberately separate from molecule and
additional_keywords. These are dimer-context labels, not monomer targets or
inputs to the native calculation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def entries_from_inventory(inventory, references):
    entries = []
    reference_columns = sorted({key for row in references.values() for key in row
                                if key.startswith('SAPT')})
    for monomer in inventory['monomers']:
        contexts = []
        for member in monomer['members']:
            row = references[member['dimer_id']]
            energies = {}
            for key in reference_columns:
                raw = row.get(key, '')
                value = float(raw) if raw.strip() else float('nan')
                if math.isinf(value):
                    raise ValueError('infinite SAPT reference energy')
                energies[key] = None if math.isnan(value) else value
            contexts.append(dict(
                dimer_id=member['dimer_id'], system=member['system'], role=member['role'],
                R=member['R'], fragment_indices=member['fragment_indices'],
                rotation=member['rotation'], translation_bohr=member['translation_bohr'],
                placement_max_abs_error_bohr=member['max_abs_error_bohr'],
                sapt_reference_components_kcal_mol=energies))
        molecule = dict(
            schema_name='qcschema_molecule', schema_version=2,
            symbols=monomer['symbols'],
            geometry=[value for xyz in monomer['geometry_bohr'] for value in xyz],
            molecular_charge=monomer['charge'], molecular_multiplicity=monomer['multiplicity'],
            fix_com=True, fix_orientation=True, fix_symmetry='c1')
        geometry_sha = digest(json.dumps(molecule, sort_keys=True, allow_nan=False).encode())
        entries.append(dict(
            name=monomer['id']+'-'+geometry_sha[:12], molecule=molecule,
            attributes=dict(
                monomer_id=monomer['id'], formula=monomer['formula'],
                geometry_sha256=geometry_sha, atom_order=monomer['atom_order'],
                representative=monomer['representative'], dimer_contexts=contexts,
                reference_units='kcal/mol',
                reference_semantics='Dimer SAPT component metadata; grouped DISP includes exchange-dispersion. Never monomer targets or runtime keywords.',
                input_roundoff_tolerance_bohr=inventory['summary']['tolerance_bohr'],
                source_pins=inventory['source_pins']),
            # Frames/bonds/scientific declarations will be populated only after
            # the shipped generic profile/QCSchema seam is qualified.
            additional_keywords={}))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory_bytes = args.inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    validation = json.loads(args.inventory.with_name('inventory-validation.json').read_text())
    assert validation['status'] == 'PASS'
    assert validation['inventory_sha256'] == digest(inventory_bytes)
    csv_path = Path(inventory['source'])/'reference_energies.csv'
    csv_bytes = csv_path.read_bytes()
    assert digest(csv_bytes) == inventory['source_pins']['reference_energies.csv']['sha256']
    references = {r['id']: r for r in csv.DictReader(csv_bytes.decode().splitlines()) if r['DB'] == 's66x8'}
    entries = entries_from_inventory(inventory, references)
    assert len(entries) == len(inventory['monomers']) == 131
    assert len({entry['name'] for entry in entries}) == 131
    assert sum(len(entry['attributes']['dimer_contexts']) for entry in entries) == 1056
    assert all(not e['additional_keywords'] and 'extras' not in e['molecule'] for e in entries)
    output = dict(
        schema='native-s66x8-singlepoint-entry-preparation/1',
        status='PREPARED_NOT_REGISTERED_OR_SUBMITTED',
        queue_policy='EMBERS_ONLY_NO_INFERNO', deployment='existing_user_deployment',
        dataset_type='singlepoint', project='native-ccasp-s66x8',
        proposed_dataset_name='S66x8-native-CCASP-monomers-v1',
        inventory_sha256=digest(inventory_bytes), reference_csv_sha256=digest(csv_bytes),
        include_sapt_reference_attributes=True, entries=entries,
        blockers=['Generic installed native property producer and QCSchema artifact roundtrip not yet qualified.',
                  'Existing deployment/client configuration and pinned immutable computational specification must be verified.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(
        status=output['status'], monomer_entries=len(entries), dimer_contexts=1056,
        reference_attribute_columns=len(entries[0]['attributes']['dimer_contexts'][0]['sapt_reference_components_kcal_mol']),
        reference_energies_in_runtime_keywords=False, output=str(args.output),
        sha256=digest(args.output.read_bytes())), indent=2))


if __name__ == '__main__':
    main()
