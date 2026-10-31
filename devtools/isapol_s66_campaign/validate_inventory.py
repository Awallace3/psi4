"""Independently reconstruct every inventory placement against raw S66x8."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    data = json.loads(raw)
    source = Path(data['source'])
    for relative, pin in data['source_pins'].items():
        content = (source/relative).read_bytes()
        assert len(content) == pin['bytes']
        assert hashlib.sha256(content).hexdigest() == pin['sha256']
    with (source/'reference_energies.csv').open() as stream:
        rows = {r['id']: r for r in csv.DictReader(stream) if r['DB'] == 's66x8'}
    with gzip.open(source/'qcel/dimers.json.gz', 'rt') as stream:
        molecules = json.load(stream)
    assert set(rows) == set(data['dimers']) and len(rows) == 528
    groups = {g['id']: g for g in data['monomers']}
    assert len(groups) == len(data['monomers']) == data['summary']['unique_ordered_geometry_monomers']
    assert sum(len(g['members']) for g in groups.values()) == 1056
    count, maximum = 0, 0.
    for name, dimer in data['dimers'].items():
        molecule = molecules[name]
        xyz = np.asarray(molecule['geometry']).reshape(-1, 3)
        assert dimer['R'] == float(rows[name]['R'])
        assert set(dimer['fragments']) == {'A', 'B'}
        for index, role in enumerate(('A', 'B')):
            item = dimer['fragments'][role]
            group = groups[item['monomer_id']]
            indices = molecule['fragments'][index]
            assert item['fragment_indices'] == indices
            assert [molecule['symbols'][i] for i in indices] == group['symbols']
            assert molecule['fragment_charges'][index] == group['charge']
            assert molecule['fragment_multiplicities'][index] == group['multiplicity']
            rotation = np.asarray(item['rotation'])
            assert np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-12)
            assert abs(np.linalg.det(rotation)-1) < 1e-12
            placed = np.asarray(group['geometry_bohr'])@rotation.T+item['translation_bohr']
            error = float(np.max(np.abs(placed-xyz[indices])))
            assert error <= data['summary']['tolerance_bohr']
            assert abs(error-item['max_abs_error_bohr']) < 1e-14
            maximum = max(maximum, error)
            count += 1
    result = dict(
        status='PASS', dimers=len(rows), reconstructed_fragments=count, monomers=len(groups),
        max_abs_error_bohr=maximum, tolerance_bohr=data['summary']['tolerance_bohr'],
        inventory_sha256=hashlib.sha256(raw).hexdigest(),
        checks=['raw input file hashes', 'all528 dimer IDs and1056 fragment roles',
                'ordered symbols, charge, multiplicity and atom indices',
                'proper rotations only; reconstructed every original fragment within recorded tolerance'],
    )
    args.inventory.with_name('inventory-validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
