"""Prepare one immutable native pilot entry, without server writes or SCF.

Preparation binds the installed implementation and geometry declaration.
Only the shipped producer validates numerical/model admission; only a later
QCArchive retrieval and runtime audit can qualify the molecular calculation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_pilot(entries, declarations, qualification, pins, resources, *,
                  monomer_id, max_replay_work):
    if (qualification.get('status') != 'passed'
            or qualification.get('runtime_unchanged') is not True
            or qualification.get('expected_test_count') != 197
            or qualification.get('test_summary') != dict(
                tests=197, failures=0, errors=0, skipped=0)):
        raise ValueError('complete sealed installed qualification required')
    for key in ('build_commit', 'build_tree', 'test_source_commit'):
        if re.fullmatch('[0-9a-f]{40}', qualification.get(key, '')) is None:
            raise ValueError('exact scientific and test source pins required')
    if (type(pins) is not dict or set(pins) != {'core', 'isapol_python'}
            or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None
                   for v in pins.values())):
        raise ValueError('exact installed implementation pins required')
    if (type(resources) is not dict
            or set(resources) != {'max_bytes', 'max_work', 'max_io_bytes', 'capacity'}
            or resources['capacity'] != 'large'
            or any(type(resources[k]) is not int or resources[k] <= 0
                   for k in ('max_bytes', 'max_work', 'max_io_bytes'))
            or type(max_replay_work) is not int or not 0 < max_replay_work <= 2_000_000_000_000):
        raise ValueError('explicit finite molecular resource limits required')
    if (not entries.get('inventory_sha256')
            or entries['inventory_sha256'] != declarations.get('inventory_sha256')):
        raise ValueError('entry/declaration inventory pins differ')
    selected = [e for e in entries['entries'] if e['attributes']['monomer_id'] == monomer_id]
    declared = [d for d in declarations['monomers'] if d['monomer_id'] == monomer_id]
    if len(selected) != 1 or len(declared) != 1:
        raise ValueError('exactly one bound monomer entry/declaration required')
    entry, declaration = copy.deepcopy(selected[0]), declared[0]
    geometry_sha = json_digest(entry['molecule'])
    if (entry.get('additional_keywords') != {}
            or entry['attributes']['geometry_sha256'] != geometry_sha
            or declaration['geometry_sha256'] != geometry_sha):
        raise ValueError('raw draft geometry or empty runtime inputs changed')
    molecule = entry['molecule']
    if (molecule.get('fix_com') is not True or molecule.get('fix_orientation') is not True
            or molecule.get('fix_symmetry') != 'c1'
            or molecule.get('molecular_charge') != 0
            or molecule.get('molecular_multiplicity') != 1
            or set(molecule)-{'schema_name', 'schema_version', 'symbols', 'geometry',
                              'molecular_charge', 'molecular_multiplicity',
                              'fix_com', 'fix_orientation', 'fix_symmetry'}):
        raise ValueError('only raw fixed neutral singlet geometry is permitted')
    request = dict(
        schema='native-monomer-request/2', point_group='c1',
        declaration=copy.deepcopy(declaration['declaration']),
        resources=copy.deepcopy(resources), max_replay_work=max_replay_work,
        implementation_sha256=copy.deepcopy(pins))
    entry['additional_keywords'] = dict(function_kwargs=dict(
        native_monomer=request, properties=['dipole']))
    entry['attributes']['native_campaign'] = dict(
        build_commit=qualification['build_commit'], build_tree=qualification['build_tree'],
        implementation_sha256=copy.deepcopy(pins),
        qualification='Installed contracts only; fresh molecular acceptance pending.')
    specification = dict(
        program='psi4', driver='properties', method='pbe0', basis='aug-cc-pvtz',
        keywords=dict(reference='rhf', puream=True, scf_type='pk', guess='sad',
                      e_convergence=1e-10, d_convergence=1e-10, maxiter=100, print=1))
    return dict(
        schema='native-s66x8-pilot-preparation/1',
        status='PREPARED_NOT_REGISTERED_OR_SUBMITTED',
        inventory_sha256=entries['inventory_sha256'], monomer_id=monomer_id,
        entry=entry, specification=specification,
        submission=dict(find_existing=False, maximum_concurrency=1, qos='embers'),
        qualification='No molecular execution, QCEngine/server roundtrip or independence audit yet.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('entries', 'declarations', 'qualification', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--monomer-id', required=True)
    for name in ('max-bytes', 'max-work', 'max-io-bytes', 'max-replay-work'):
        parser.add_argument('--'+name, type=int, required=True)
    args = parser.parse_args()
    qpath = args.qualification
    qualification = json.loads(qpath.read_text())
    identity_path = qpath.with_name('imported-identity.json')
    junit_path = qpath.with_name('junit.xml')
    if (qualification.get('imported_identity_sha256') != file_digest(identity_path)
            or qualification.get('junit_sha256') != file_digest(junit_path)):
        raise ValueError('qualification identity/JUnit file hashes differ')
    entries, declarations, identity = [
        json.loads(p.read_text()) for p in (args.entries, args.declarations, identity_path)]
    result = prepare_pilot(
        entries, declarations, qualification, identity['implementation']['sha256'],
        dict(max_bytes=args.max_bytes, max_work=args.max_work,
             max_io_bytes=args.max_io_bytes, capacity='large'),
        monomer_id=args.monomer_id, max_replay_work=args.max_replay_work)
    result['source_files_sha256'] = {
        name: file_digest(path) for name, path in (
            ('entries', args.entries), ('declarations', args.declarations),
            ('installed_qualification', qpath), ('imported_identity', identity_path),
            ('junit', junit_path))}
    result['scientific_build_commit'] = qualification['build_commit']
    result['installed_test_commit'] = qualification['test_source_commit']
    result['build_identity_sha256'] = qualification['build_identity_sha256']
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], monomer=args.monomer_id,
                          output=str(args.output), sha256=file_digest(args.output))))


if __name__ == '__main__':
    main()
