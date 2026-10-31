"""Geometry-only S66x8 monomer inventory; never imports or executes Psi4.

Reuse requires equal ordered symbols/charge/spin and a verified proper rigid
placement within the recorded input-roundoff tolerance. Chemical names, formula
and an element match alone never authorize monomer reuse or site-level COPY.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def proper_placement(source, target):
    """Return Q,t,maxerror,RMSD for column-vector target=Q@source+t."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('equal Nx3 geometries required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('finite geometries required')
    a, b = source-source.mean(axis=0), target-target.mean(axis=0)
    u, singular, vt = np.linalg.svd(a.T@b)
    reflection = np.eye(3)
    reflection[-1, -1] = 1. if np.linalg.det(u@vt) > 0 else -1.
    rotation = (u@reflection@vt).T
    translation = target.mean(axis=0)-rotation@source.mean(axis=0)
    residual = source@rotation.T+translation-target
    assert np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-12)
    return dict(rotation=rotation.tolist(), translation_bohr=translation.tolist(),
                max_abs_error_bohr=float(np.max(np.abs(residual))),
                rmsd_bohr=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                coordinate_rank=int(np.sum(singular > max(float(singular[0]), 1.)*1e-12)))


def distances(xyz):
    xyz = np.asarray(xyz)
    return np.linalg.norm(xyz[:, None]-xyz[None, :], axis=-1)


def formula(symbols):
    counts = Counter(symbols)
    order = [s for s in ('C', 'H') if s in counts]+sorted(set(counts)-{'C', 'H'})
    return ''.join(s+(str(counts[s]) if counts[s] != 1 else '') for s in order)


def group_fragments(occurrences, tolerance):
    groups, by_identity = [], {}
    for occurrence in occurrences:
        key = (tuple(occurrence['symbols']), occurrence['charge'], occurrence['multiplicity'])
        xyz = np.asarray(occurrence['geometry_bohr'])
        d = distances(xyz)
        found = None
        for index in by_identity.get(key, []):
            group = groups[index]
            # Cheap rejection only. Membership is decided by the proper fit.
            if np.max(np.abs(group['_distances']-d)) > 4*tolerance:
                continue
            placement = proper_placement(group['geometry_bohr'], xyz)
            if placement['max_abs_error_bohr'] <= tolerance:
                found = index, placement
                break
        if found is None:
            index = len(groups)
            group = dict(
                id=f'monomer-{index+1:04d}', symbols=occurrence['symbols'],
                charge=occurrence['charge'], multiplicity=occurrence['multiplicity'],
                geometry_bohr=occurrence['geometry_bohr'], formula=formula(occurrence['symbols']),
                representative=dict(dimer_id=occurrence['dimer_id'], role=occurrence['role'],
                                    R=occurrence['R'], fragment_indices=occurrence['fragment_indices']),
                atom_order='Exactly the representative source fragment order; no atom permutation.',
                members=[], _distances=d)
            groups.append(group)
            by_identity.setdefault(key, []).append(index)
            found = index, proper_placement(xyz, xyz)
        index, placement = found
        groups[index]['members'].append(dict(
            dimer_id=occurrence['dimer_id'], system=occurrence['system'], R=occurrence['R'],
            role=occurrence['role'], fragment_indices=occurrence['fragment_indices'],
            source_fragment_sha256=occurrence['source_fragment_sha256'], **placement))
    for group in groups:
        group.pop('_distances')
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/home/awallace43/data/los-ii'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tolerance-bohr', type=float, default=1e-7)
    args = parser.parse_args()
    if not 0 < args.tolerance_bohr <= 1e-7:
        raise ValueError('inventory tolerance must be explicit and no larger than1e-7bohr')
    pins = {}
    def read(relative):
        raw = (args.data/relative).read_bytes()
        pins[relative] = dict(bytes=len(raw), sha256=digest(raw))
        return raw
    manifest = json.loads(read('manifest.json'))
    csv_bytes = read('reference_energies.csv')
    geometry_bytes = read('qcel/dimers.json.gz')
    for name in ('reference_energies.csv', 'qcel/dimers.json.gz'):
        assert pins[name] == manifest['files'][name]
    rows = sorted((r for r in csv.DictReader(csv_bytes.decode().splitlines()) if r['DB'] == 's66x8'),
                  key=lambda r: (r['system_label'], float(r['R'])))
    assert len(rows) == 528 and len({r['system_label'] for r in rows}) == 66
    molecules = json.loads(gzip.decompress(geometry_bytes))
    occurrences, dimer_map = [], {}
    for row in rows:
        molecule = molecules[row['id']]
        xyz = np.asarray(molecule['geometry']).reshape(-1, 3)
        fragments = molecule['fragments']
        assert len(fragments) == 2 and molecule['fix_com'] and molecule['fix_orientation']
        assert sorted(fragments[0]+fragments[1]) == list(range(len(xyz)))
        dimer_map[row['id']] = dict(system=row['system_label'], R=float(row['R']),
            geometry_sha256=digest(canonical_bytes(molecule)), fragments={})
        for role, indices, charge, multiplicity in zip(
                ('A', 'B'), fragments, molecule['fragment_charges'],
                molecule['fragment_multiplicities'], strict=True):
            record = dict(symbols=[molecule['symbols'][i] for i in indices],
                          geometry_bohr=xyz[indices].tolist(), charge=float(charge),
                          multiplicity=int(multiplicity))
            occurrences.append(dict(
                **record, dimer_id=row['id'], system=row['system_label'], R=float(row['R']),
                role=role, fragment_indices=indices, source_fragment_sha256=digest(canonical_bytes(record))))
    assert len(occurrences) == 1056
    groups = group_fragments(occurrences, args.tolerance_bohr)
    for group in groups:
        for member in group['members']:
            dimer_map[member['dimer_id']]['fragments'][member['role']] = dict(
                monomer_id=group['id'], fragment_indices=member['fragment_indices'],
                rotation=member['rotation'], translation_bohr=member['translation_bohr'],
                max_abs_error_bohr=member['max_abs_error_bohr'])
    assert all(set(d['fragments']) == {'A', 'B'} for d in dimer_map.values())
    # A stricter sensitivity count exposes any use of input-roundoff tolerance.
    strict_groups = group_fragments(occurrences, 1e-10)
    summary = dict(
        dimers=528, curves=66, fragment_occurrences=1056, unique_ordered_geometry_monomers=len(groups),
        strict_1e10_bohr_groups=len(strict_groups), tolerance_bohr=args.tolerance_bohr,
        elements=sorted({s for o in occurrences for s in o['symbols']}),
        formula_counts=dict(sorted(Counter(g['formula'] for g in groups).items())),
        atoms_min=min(len(g['symbols']) for g in groups), atoms_max=max(len(g['symbols']) for g in groups),
        nonneutral=sum(g['charge'] != 0 for g in groups),
        nonsinglet=sum(g['multiplicity'] != 1 for g in groups),
        max_placement_error_bohr=max(m['max_abs_error_bohr'] for g in groups for m in g['members']),
        rank_deficient_placements=sum(m['coordinate_rank'] < 2 for g in groups for m in g['members']),
        independent_rank2_uppertriangle_parameters_max=36*max(len(g['symbols']) for g in groups),
    )
    output = dict(
        schema='native-s66x8-monomer-inventory/1', generated=datetime.now(timezone.utc).isoformat(),
        source=str(args.data), source_pins=pins, summary=summary, monomers=groups, dimers=dimer_map,
        campaign_status='GEOMETRY_INVENTORY_ONLY_NO_CALCULATIONS_SUBMITTED',
        queue_policy='EMBERS_ONLY_NO_INFERNO',
        qualifications=[
            'Ordered-symbol/charge/spin equality plus proper rigid fit; no formula/name-only reuse or element-level site COPY.',
            'Numerical coordinate reuse tolerance is recorded; distinct conformations outside it remain separate.',
            'A stored placement proves geometry matching only, not rotational covariance of the native LW/property-grid algorithm.',
            'Atom-permuted equivalent monomers are conservatively not deduplicated.',
            'Rank-deficient placements need explicit frame handling before anisotropic dimer contractions.',
            'No native SCF/ISA/response/PFIT calculation is performed by this inventory.',
        ])
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output/'inventory.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False))
    for group in groups:
        record = dict(
            schema='native-monomer-campaign-input/1', units='bohr', monomer_id=group['id'],
            symbols=group['symbols'], geometry=group['geometry_bohr'],
            molecular_charge=group['charge'], molecular_multiplicity=group['multiplicity'],
            source_pins=pins, representative=group['representative'])
        (args.output/(group['id']+'.json')).write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps(dict(status=output['campaign_status'], **summary,
                         inventory_sha256=digest(path.read_bytes()), output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
