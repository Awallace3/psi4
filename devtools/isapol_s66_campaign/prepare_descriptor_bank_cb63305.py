"""Prepare all131 descriptor inputs; never SCF, QCArchive or job submission.

Geometry/declaration inputs and installed receipts are closed immutable pins.
Resource declarations are explicit input data, not inferred or promoted here.
The resulting bank still needs actual descriptors and allocation admission.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

INPUT_SHA256 = {
    'inventory': '878c327424913be10e5822a8c9854bd26e38900cea2f299dbdf2d00a0a819c5b',
    'entries': '5c363f5750aea53db0c451408e0a921b330f77fbd88cf3381f9001522cb9d0d1',
    'declarations': 'a30685178a1c0fab1f70626d4c5e40ac0ea839a7fea31ecec04fc7f0bf88cf98',
    'input_validation': 'b7de0fc47fa49a0f0b909a4d6110bf36bb2086ce1442c7f03796359338ed492c',
}


def helper(name):
    if name not in ('qualified_cb63305', 'resource_plan_cb63305',
                    'qualified_ed561666', 'resource_plan_ed561666',
                    'qualified_grid400', 'resource_plan_grid400'):
        raise ValueError('closed descriptor preparation helpers required')
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_expanded_prepare_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def prepare_bank(inputs, receipts, resource_bytes, *, producer='cb63305'):
    """Return unsubmitted runtime-only inputs, never a qualified descriptor bank."""
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed installed producer profile required')
    return _prepare_bank(inputs, receipts, resource_bytes, producer=producer)


def prepare_grid400_pilot_bank(inputs, receipts, resource_bytes):
    """Prepare only the authorized fresh monomer0008, never an all131 bank.

    Candidate installed receipts must exist and verify before construction.
    This entry point does not accept arbitrary selections or producer modules.
    """
    return _prepare_bank(inputs, receipts, resource_bytes, producer='grid400',
                         selected=('monomer-0008',))


def prepare_grid400_full_bank(inputs, receipts, resource_bytes):
    """Explicitly prepare all131 candidate geometries; never descriptor admission."""
    return _prepare_bank(inputs, receipts, resource_bytes, producer='grid400', scope='full-bank')


def _prepare_bank(inputs, receipts, resource_bytes, *, producer, selected=None, scope='default'):
    if (scope not in ('default', 'full-bank')
            or scope == 'full-bank' and (producer != 'grid400' or selected is not None)
            or (producer == 'grid400' and scope == 'default' and selected != ('monomer-0008',))
            or (producer in ('cb63305', 'ed561666') and selected is not None)
            or producer not in ('cb63305', 'ed561666', 'grid400')):
        raise ValueError('closed descriptor producer/selection required')
    if type(inputs) is not dict or set(inputs) != set(INPUT_SHA256):
        raise ValueError('exact immutable input roles required')
    for role, expected in INPUT_SHA256.items():
        if type(inputs[role]) is not bytes or digest(inputs[role]) != expected:
            raise ValueError('validated raw input hash differs: '+role)
    profile, planner = helper('qualified_'+producer), helper('resource_plan_'+producer)
    producer = profile.verify_receipts(receipts)
    docs = {key: json.loads(raw) for key, raw in inputs.items()}
    inventory, entries, declarations, validation = (
        docs[key] for key in ('inventory', 'entries', 'declarations', 'input_validation'))
    if (validation.get('status') != 'PASS_INPUT_PREPARATION_ONLY'
            or validation.get('monomers') != 131 or validation.get('fragment_contexts') != 1056
            or validation.get('input_sha256') != {
                key: INPUT_SHA256[key] for key in ('inventory', 'entries', 'declarations')}
            or entries.get('inventory_sha256') != INPUT_SHA256['inventory']
            or declarations.get('inventory_sha256') != INPUT_SHA256['inventory']):
        raise ValueError('exact131-geometry/1056-context input validation required')
    expected = {m['id']: m for m in inventory['monomers']}
    drafted = {e['attributes']['monomer_id']: e for e in entries['entries']}
    declared = {d['monomer_id']: d for d in declarations['monomers']}
    ids = {f'monomer-{i:04d}' for i in range(1, 132)}
    if (set(expected) != ids or set(drafted) != ids or set(declared) != ids
            or len(inventory['monomers']) != 131 or len(entries['entries']) != 131
            or len(declarations['monomers']) != 131):
        raise ValueError('complete unique131-monomer input identity set required')
    if sum(len(entry['attributes']['dimer_contexts']) for entry in drafted.values()) != 1056:
        raise ValueError('complete1056-context bookkeeping required')
    active = ids if selected is None else set(selected)
    if type(resource_bytes) is not bytes:
        raise ValueError('immutable explicit resource declaration bytes required')
    resources = json.loads(resource_bytes)
    expected_keys = {'schema', 'status', 'inventory_sha256', 'policy_id', 'monomers'}
    if selected is not None:
        expected_keys.add('selected_monomers')
    if (set(resources) != expected_keys
            or resources['schema'] != ('native-s66x8-resource-declarations/2'
                if selected is None else 'native-s66x8-selected-resource-declarations/1')
            or resources['status'] != 'DECLARED_NOT_ADMITTED'
            or resources['inventory_sha256'] != INPUT_SHA256['inventory']
            or resources['policy_id'] != planner.POLICY
            or type(resources['monomers']) is not dict or set(resources['monomers']) != active
            or (selected is not None and resources['selected_monomers'] != list(selected))):
        raise ValueError('exact explicit131-monomer resource declarations required'
                         if selected is None else 'exact selected monomer0008 resource declarations required')
    pfit = planner.shared_pfit_plan()
    rows, contexts = [], 0
    for monomer in sorted(active):
        entry, declaration, contract = drafted[monomer], declared[monomer], resources['monomers'][monomer]
        geometry_hash = json_digest(entry['molecule'])
        if (entry.get('additional_keywords') != {}
                or geometry_hash != entry['attributes']['geometry_sha256']
                or geometry_hash != declaration['geometry_sha256']
                or contract.get('geometry_sha256') != geometry_hash):
            raise ValueError('raw geometry or empty runtime draft differs: '+monomer)
        if set(contract) != {'geometry_sha256', 'resources', 'max_replay_work',
                             'pfit_maximum_work_bytes', 'intended_memory_bytes'}:
            raise ValueError('exact finite monomer resource contract required')
        budget = contract['resources']
        if (type(budget) is not dict or set(budget) != {
                'max_bytes', 'max_work', 'max_io_bytes', 'capacity'}
                or type(budget['capacity']) is not str or budget['capacity'] not in ('large', 'extended')):
            raise ValueError('explicit large or extended resources required')
        for key, ceiling in (('max_bytes', None),
                ('max_work', (100 if budget['capacity'] == 'large' else 128)*10**12),
                ('max_io_bytes', 64*1024**3)):
            value = budget[key]
            if type(value) is not int or value <= 0 or (ceiling is not None and value > ceiling):
                raise ValueError('invalid finite resource '+key)
        memory = contract['intended_memory_bytes']
        if type(memory) is not int or memory < budget['max_bytes']:
            raise ValueError('intended Psi4 memory must cover the numeric budget')
        if selected is not None and (budget != dict(
                max_bytes=48*1024**3, max_work=75*10**12,
                max_io_bytes=16*1024**3, capacity='large') or memory != 60*1024**3):
            raise ValueError('exact approved fresh48GiB numeric/60GiB science contract required')
        cap = contract['pfit_maximum_work_bytes']
        if cap is None:
            raise ValueError('explicit finite native workspace required')
        if selected is not None and (
                type(cap) is not int or cap != 2*1024**3
                or type(contract['max_replay_work']) is not int
                or contract['max_replay_work'] != 10**12):
            raise ValueError('exact original replay/native workspace limits required')
        pfit.check_replay_work(contract['max_replay_work'])
        count = len(entry['molecule']['symbols'])
        if (count != len(declaration['declaration']['frames'])
                or declaration['declaration']['schema'] != 'native-monomer-declaration/1'
                or declaration['declaration']['profile'] != 's66-chno-isa-a-v1'
                or declaration['policy']['independent_parameters'] != 36*count):
            raise ValueError('full independent-site declaration differs')
        fit = pfit.combined_fit_plan(2000, 9*count, 36*count, 11, maximum_work_bytes=cap)
        pfit.packed_design_plan(npoint=2000, channel_count=9*count, parameter_count=36*count,
            used_channels=8*count, block_rows=4096, max_bytes=budget['max_bytes'],
            max_work=contract['max_replay_work'])
        if (fit['complete_refinement_bytes'] > budget['max_bytes']
                or fit['complete_refinement_work'] > budget['max_work']):
            raise ValueError('complete refinement alone exceeds explicit resources')
        request = dict(schema='native-monomer-request/3', point_group='c1',
            declaration=copy.deepcopy(declaration['declaration']), resources=copy.deepcopy(budget),
            max_replay_work=contract['max_replay_work'], pfit_maximum_work_bytes=cap,
            implementation_sha256=copy.deepcopy(producer['implementation_sha256']))
        runtime = dict(molecule=copy.deepcopy(entry['molecule']), specification=dict(
            driver='properties', model=dict(method='pbe0', basis='aug-cc-pvtz'),
            keywords=dict(reference='rhf', puream=True, scf_type='pk', guess='sad',
                e_convergence=1e-10, d_convergence=1e-10, maxiter=100, print=1,
                function_kwargs=dict(native_monomer=request, properties=['dipole']))))
        if 'sapt_reference' in json.dumps(runtime):
            raise ValueError('reference attributes entered executable inputs')
        contexts += len(entry['attributes']['dimer_contexts'])
        rows.append(dict(monomer_id=monomer, entry_name=entry['name'],
            geometry_sha256=geometry_hash, intended_memory_bytes=memory,
            runtime_input=runtime, runtime_input_sha256=json_digest(runtime)))
    if selected is None and contexts != 1056:
        raise ValueError('complete1056-context bookkeeping required')
    result = dict(schema='native-s66x8-descriptor-preparation/2',
        status='PREPARED_FOR_DESCRIPTORS_ONLY', policy_id=planner.POLICY,
        input_sha256={k: digest(v) for k, v in inputs.items()},
        resource_declarations_sha256=digest(resource_bytes), producer=producer,
        monomer_count=len(active), fragment_contexts=contexts, monomers=rows,
        qualification='No descriptors, SCF, molecular calculations, manager discovery, '
                      'QCArchive writes or jobs. Resource declarations are not admission.')
    if selected is not None:
        result.update(schema='native-s66x8-selected-descriptor-preparation/1',
            status='PREPARED_SELECTED_MONOMER_FOR_DESCRIPTORS_ONLY',
            selected_monomers=list(selected), inventory_monomers=131, inventory_fragment_contexts=1056)
    if scope == 'full-bank':
        result['campaign_scope'] = 'full-bank'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_SHA256, 'resources', 'build_identity', 'qualification', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--producer', choices=('cb63305', 'ed561666', 'grid400'), default='cb63305')
    parser.add_argument('--scope', choices=('default', 'full-bank'), default='default')
    args = parser.parse_args()
    scope = getattr(args, 'scope', 'default')
    if scope == 'full-bank' and args.producer != 'grid400':
        raise ValueError('full-bank scope is explicit candidate-only')
    qpath = args.qualification
    receipt_paths = dict(build_identity=args.build_identity, installed_qualification=qpath,
        imported_identity=qpath.with_name('imported-identity.json'), junit=qpath.with_name('junit.xml'))
    inputs = {role: getattr(args, role).read_bytes() for role in INPUT_SHA256}
    receipts = {role: p.read_bytes() for role, p in receipt_paths.items()}
    resources = args.resources.read_bytes()
    result = (prepare_grid400_full_bank(inputs, receipts, resources)
        if scope == 'full-bank' else prepare_grid400_pilot_bank(inputs, receipts, resources)
        if args.producer == 'grid400'
        else prepare_bank(inputs, receipts, resources, producer=args.producer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], monomers=result['monomer_count'],
        contexts=result['fragment_contexts'], sha256=digest(args.output.read_bytes()))))


if __name__ == '__main__':
    main()
