"""Cross-check raw inventory, drafted entries and explicit declarations.

Geometry/QCSchema preparation only: no Psi4 execution or QCArchive connection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import qcelemental as qcel


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(inventory, entries, declarations):
    expected = {m['id']: m for m in inventory['monomers']}
    prepared = {row['attributes']['monomer_id']: row for row in entries['entries']}
    declared = {row['monomer_id']: row for row in declarations['monomers']}
    if (set(expected) != set(prepared) or set(expected) != set(declared)
            or len(prepared) != len(entries['entries'])
            or len(declared) != len(declarations['monomers'])):
        raise ValueError('monomer preparation inventory differs or contains duplicates')
    max_qcel_roundoff = 0.
    contexts = 0
    for monomer_id, raw in expected.items():
        entry, inputs = prepared[monomer_id], declared[monomer_id]
        molecule = entry['molecule']
        attrs = entry['attributes']
        if entry['additional_keywords'] or 'extras' in molecule:
            raise ValueError('draft preparation must not contain runtime/reference keywords')
        if (molecule['symbols'] != raw['symbols']
                or molecule['geometry'] != [v for xyz in raw['geometry_bohr'] for v in xyz]
                or molecule['molecular_charge'] != raw['charge']
                or molecule['molecular_multiplicity'] != raw['multiplicity']
                or not molecule['fix_com'] or not molecule['fix_orientation']
                or molecule['fix_symmetry'] != 'c1'):
            raise ValueError(f'{monomer_id}: raw ordered geometry/state/locks differ')
        geometry_sha = digest(json.dumps(molecule, sort_keys=True, allow_nan=False).encode())
        if geometry_sha != attrs['geometry_sha256'] or geometry_sha != inputs['geometry_sha256']:
            raise ValueError(f'{monomer_id}: declaration/entry geometry digest mismatch')
        model = qcel.models.Molecule(**molecule)
        error = float(np.max(np.abs(model.geometry-np.asarray(raw['geometry_bohr']))))
        if (list(model.symbols) != raw['symbols'] or error > 1e-8
                or not model.fix_com or not model.fix_orientation
                or model.fix_symmetry.lower() != 'c1'):
            raise ValueError(f'{monomer_id}: QCSchema conversion differs beyond input roundoff')
        max_qcel_roundoff = max(max_qcel_roundoff, error)
        definition = inputs['declaration']
        count = len(raw['symbols'])
        frames = np.asarray(definition['frames'])
        if (set(definition) != {'schema', 'profile', 'frames', 'bonds'}
                or definition['schema'] != 'native-monomer-declaration/1'
                or definition['profile'] != 's66-chno-isa-a-v1'
                or frames.shape != (count, 3, 3) or not np.isfinite(frames).all()
                or not np.allclose(frames.transpose(0, 2, 1) @ frames, np.eye(3), rtol=0, atol=1e-12)
                or not np.allclose(np.linalg.det(frames), 1., rtol=0, atol=1e-12)
                or inputs['policy']['independent_parameters'] != 36*count):
            raise ValueError(f'{monomer_id}: improper declared frame/parameter inventory')
        edges = definition['bonds']
        if (any(len(edge) != 2 or any(type(i) is not int or not 0 <= i < count for i in edge)
                or edge[0] == edge[1] for edge in edges)
                or len({tuple(sorted(edge)) for edge in edges}) != len(edges)):
            raise ValueError(f'{monomer_id}: invalid declared bonds')
        if (attrs['source_pins'] != inventory['source_pins']
                or len(attrs['dimer_contexts']) != len(raw['members'])):
            raise ValueError(f'{monomer_id}: source or placement context differs')
        contexts += len(attrs['dimer_contexts'])
    return dict(
        status='PASS_INPUT_PREPARATION_ONLY', monomers=len(expected),
        fragment_contexts=contexts, qcelemental_version=qcel.__version__,
        max_qcel_coordinate_roundoff_bohr=max_qcel_roundoff,
        qualification='Raw input and canonicalized QCSchema coordinates are distinguished; no molecular, installed-runtime, QCArchive or accuracy acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('entries', type=Path)
    parser.add_argument('declarations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    blobs = {name: getattr(args, name).read_bytes() for name in ('inventory', 'entries', 'declarations')}
    documents = {name: json.loads(raw) for name, raw in blobs.items()}
    inventory_sha = digest(blobs['inventory'])
    if any(documents[name]['inventory_sha256'] != inventory_sha for name in ('entries', 'declarations')):
        raise ValueError('drafts are not bound to the same raw inventory')
    result = validate(**documents)
    if result['monomers'] != 131 or result['fragment_contexts'] != 1056:
        raise ValueError('incomplete S66x8 input coverage')
    result['input_sha256'] = {name: digest(raw) for name, raw in blobs.items()}
    result['validator_sha256'] = digest(Path(__file__).read_bytes())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
