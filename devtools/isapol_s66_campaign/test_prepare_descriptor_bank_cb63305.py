"""Synthetic input-bank bookkeeping; never real descriptor/molecular acceptance."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'prepare_descriptor_bank_cb63305', Path(__file__).with_name('prepare_descriptor_bank_cb63305.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def encode(data):
    return json.dumps(data, sort_keys=True, allow_nan=False).encode()


def fixture():
    molecule = dict(schema_name='qcschema_molecule', schema_version=2,
        symbols=['O', 'H', 'H'], geometry=[0., 0., 0., 1., 0., 1., -1., 0., 1.],
        molecular_charge=0, molecular_multiplicity=1,
        fix_com=True, fix_orientation=True, fix_symmetry='c1')
    geometry = prepare.json_digest(molecule)
    frame = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    inventory, entries, declarations, contracts = [], [], [], {}
    for i in range(1, 132):
        mid = f'monomer-{i:04d}'
        inventory.append(dict(id=mid))
        contexts = [{'sapt_reference_components_kcal_mol': {'SAPT0/adz DISP': -1.}}]*(16 if i == 131 else 8)
        entries.append(dict(name=mid+'-'+geometry[:12], molecule=copy.deepcopy(molecule),
            additional_keywords={}, attributes=dict(monomer_id=mid, geometry_sha256=geometry,
                                                    dimer_contexts=contexts)))
        declarations.append(dict(monomer_id=mid, geometry_sha256=geometry,
            declaration=dict(schema='native-monomer-declaration/1', profile='s66-chno-isa-a-v1',
                             frames=[copy.deepcopy(frame) for _ in range(3)], bonds=[[0, 1], [0, 2]]),
            policy=dict(independent_parameters=108)))
        contracts[mid] = dict(geometry_sha256=geometry, resources=dict(max_bytes=12*1024**3,
            max_work=6*10**12, max_io_bytes=16*1024**3, capacity='large'),
            max_replay_work=10**12, pfit_maximum_work_bytes=256*1024**2,
            intended_memory_bytes=20*1024**3)
    return dict(inventory={'monomers': inventory}, entries={'entries': entries},
                declarations={'monomers': declarations}), contracts


def bind_inputs(monkeypatch, documents):
    """Mock closed raw input pins only, to isolate preparation shape guards."""
    docs = copy.deepcopy(documents)
    raw = {'inventory': encode(docs['inventory'])}
    inventory_hash = prepare.digest(raw['inventory'])
    for key in ('entries', 'declarations'):
        docs[key]['inventory_sha256'] = inventory_hash
        raw[key] = encode(docs[key])
    docs['input_validation'] = dict(status='PASS_INPUT_PREPARATION_ONLY', monomers=131,
        fragment_contexts=1056, input_sha256={k: prepare.digest(v) for k, v in raw.items()})
    raw['input_validation'] = encode(docs['input_validation'])
    monkeypatch.setattr(prepare, 'INPUT_SHA256', {k: prepare.digest(v) for k, v in raw.items()})
    original = prepare.helper
    def helper(name):
        module = original(name)
        if name == 'qualified_cb63305':
            module.verify_receipts = lambda _: dict(status='SYNTHETIC_UNIT_RECEIPTS_ONLY',
                                                    implementation_sha256=dict(module.IMPLEMENTATION))
        return module
    monkeypatch.setattr(prepare, 'helper', helper)
    return raw


def resources(raw_inputs, contracts):
    return encode(dict(schema='native-s66x8-resource-declarations/2',
        status='DECLARED_NOT_ADMITTED', inventory_sha256=prepare.digest(raw_inputs['inventory']),
        policy_id='cb63305-reference-h2h1-ledger-v2', monomers=contracts))


@pytest.mark.parametrize('producer', [None, True, 'latest', '../qualified_ed561666'])
def test_unknown_or_caller_selected_producer_paths_are_rejected(producer):
    with pytest.raises(ValueError, match='closed installed producer'):
        prepare.prepare_bank({}, {}, b'', producer=producer)


def test_new_closed_producer_uses_new_identity_and_requires_its_own_policy(monkeypatch):
    documents, contracts = fixture()
    raw = bind_inputs(monkeypatch, documents)
    original = prepare.helper
    def helper(name):
        module = original(name)
        if name == 'qualified_ed561666':
            module.verify_receipts = lambda _: dict(status='SYNTHETIC_UNIT_RECEIPTS_ONLY',
                scientific_commit=module.SCIENTIFIC_COMMIT,
                implementation_sha256=dict(module.IMPLEMENTATION))
        return module
    monkeypatch.setattr(prepare, 'helper', helper)
    old_resources = resources(raw, contracts)
    new_resources = json.loads(old_resources)
    new_resources['policy_id'] = 'ed561666-reference-h2h1-ledger-v3'
    with pytest.raises(ValueError, match='resource declarations'):
        prepare.prepare_bank(raw, {}, old_resources, producer='ed561666')
    with pytest.raises(ValueError, match='resource declarations'):
        prepare.prepare_bank(raw, {}, encode(new_resources))
    bank = prepare.prepare_bank(raw, {}, encode(new_resources), producer='ed561666')
    assert bank['monomer_count'] == 131 and bank['fragment_contexts'] == 1056
    assert bank['status'] == 'PREPARED_FOR_DESCRIPTORS_ONLY'
    assert bank['producer']['scientific_commit'] == 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
    identity = helper('qualified_ed561666').IMPLEMENTATION
    assert all(row['runtime_input']['specification']['keywords']['function_kwargs']
               ['native_monomer']['implementation_sha256'] == identity for row in bank['monomers'])
    assert not any('sapt_reference' in json.dumps(row['runtime_input']) for row in bank['monomers'])


def test_actual_input_constants_are_exact_previous_geometry_preparation_not_models():
    assert prepare.INPUT_SHA256['inventory'] == (
        '878c327424913be10e5822a8c9854bd26e38900cea2f299dbdf2d00a0a819c5b')
    assert prepare.INPUT_SHA256['input_validation'] == (
        'b7de0fc47fa49a0f0b909a4d6110bf36bb2086ce1442c7f03796359338ed492c')


def test_complete_bank_preserves_context_counts_but_never_sends_reference_attributes(monkeypatch):
    docs, contracts = fixture()
    raw = bind_inputs(monkeypatch, docs)
    result = prepare.prepare_bank(raw, {}, resources(raw, contracts))
    assert result['status'] == 'PREPARED_FOR_DESCRIPTORS_ONLY'
    assert (result['monomer_count'], result['fragment_contexts']) == (131, 1056)
    assert len(result['monomers']) == 131
    for row in result['monomers']:
        runtime = row['runtime_input']
        assert prepare.json_digest(runtime) == row['runtime_input_sha256']
        assert 'sapt_reference' not in json.dumps(runtime)
        request = runtime['specification']['keywords']['function_kwargs']['native_monomer']
        assert request['schema'] == 'native-monomer-request/3'
        assert request['pfit_maximum_work_bytes'] == 256*1024**2
        assert request['resources']['max_work'] == 6*10**12
        assert len(request['declaration']['frames']) == 3
    assert 'Resource declarations are not admission' in result['qualification']


@pytest.mark.parametrize('role', ['inventory', 'entries', 'declarations', 'input_validation'])
def test_changed_input_bytes_are_refused_before_preparation(monkeypatch, role):
    docs, contracts = fixture()
    raw = bind_inputs(monkeypatch, docs)
    resource_blob = resources(raw, contracts)
    raw[role] += b' '
    with pytest.raises(ValueError, match='input hash differs'):
        prepare.prepare_bank(raw, {}, resource_blob)


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate_entry', 'crosswired_geometry'])
def test_partial_extra_duplicate_or_cross_wired_banks_cannot_pass(monkeypatch, change):
    docs, contracts = fixture()
    if change == 'missing':
        contracts.pop('monomer-0001')
    elif change == 'extra':
        contracts['monomer-9999'] = copy.deepcopy(contracts['monomer-0001'])
    elif change == 'duplicate_entry':
        docs['entries']['entries'].append(copy.deepcopy(docs['entries']['entries'][0]))
    else:
        contracts['monomer-0001']['geometry_sha256'] = '0'*64
    raw = bind_inputs(monkeypatch, docs)
    with pytest.raises(ValueError):
        prepare.prepare_bank(raw, {}, resources(raw, contracts))


@pytest.mark.parametrize('field,value', [
    ('pfit_maximum_work_bytes', None), ('pfit_maximum_work_bytes', 1),
    ('pfit_maximum_work_bytes', True), ('max_replay_work', None), ('max_replay_work', False),
    ('max_replay_work', 2*10**12+1), ('intended_memory_bytes', 1),
])
def test_finite_contracts_cannot_be_inferred_or_promoted(monkeypatch, field, value):
    docs, contracts = fixture()
    contracts['monomer-0001'][field] = value
    raw = bind_inputs(monkeypatch, docs)
    with pytest.raises(ValueError):
        prepare.prepare_bank(raw, {}, resources(raw, contracts))
