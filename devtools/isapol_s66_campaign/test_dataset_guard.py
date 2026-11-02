"""Synthetic dataset snapshots only: no server connection or quantum work."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('dataset_guard', Path(__file__).with_name('dataset_guard.py'))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def baseline():
    policy = dict(authenticated_owner='owner', implementation={'core': 'core', 'isapol_python': 'python'},
                  specification=dict(program='psi4', driver='properties', method='pbe0',
                                     basis='aug-cc-pvtz', keywords={},
                                     protocols={'error_correction': {'default_policy': False}}),
                  water_envelope_sha256='accepted-water')
    water = dict(id=guard.WATER_RECORD, status='complete', tensor_sha256=guard.WATER_TENSOR,
                 native_envelope_sha256='accepted-water',
                 history=[{'manager_name': 'ccasp-native-s66-water-14948991-owned'}])
    before = dict(metadata=dict(id=353, dataset_type='singlepoint', name=guard.DATASET_NAME,
                                owner_user='owner', owner_group='unchanged-group',
                                provenance=dict(scientific_build_commit=guard.SCIENTIFIC_COMMIT,
                                                implementation_sha256=policy['implementation'])),
                  specifications={guard.SPECIFICATION: {'specification': policy['specification']}},
                  entries={guard.WATER_ENTRY: {'attributes': {
                      'monomer_id': 'monomer-0001', 'geometry_sha256': 'water-geometry'}}},
                  associations={guard.WATER_ENTRY: {guard.SPECIFICATION: guard.WATER_RECORD}},
                  records={str(guard.WATER_RECORD): water})
    return before, policy


def draft():
    return dict(monomer_id='monomer-0004', entry=dict(name='methanol-new', attributes=dict(
        monomer_id='monomer-0004', geometry_sha256='new-geometry')))


def appended(before, *, record_id=None):
    after = copy.deepcopy(before)
    after['entries']['methanol-new'] = draft()['entry']
    after['associations']['methanol-new'] = {guard.SPECIFICATION: record_id}
    if record_id is not None:
        after['records'][str(record_id)] = dict(id=record_id, status='waiting', history=[])
    return after


def test_owned_identity_water_and_spec_are_required_before_any_write():
    before, policy = baseline()
    guard.validate_owned(before, **policy)
    guard.validate_new_entry(before, draft())


@pytest.mark.parametrize('field', [
    'id', 'owner', 'group-provenance', 'name', 'type', 'specification', 'correction',
    'water-association', 'water-status', 'water-history', 'water-tensor', 'water-envelope',
])
def test_wrong_owned_state_refused(field):
    before, policy = baseline()
    if field in ('id', 'name'):
        before['metadata'][field] = 'wrong'
    elif field == 'owner':
        before['metadata']['owner_user'] = 'not-owner'
    elif field == 'type':
        before['metadata']['dataset_type'] = 'manybody'
    elif field == 'group-provenance':
        before['metadata']['provenance'] = {}
    elif field == 'specification':
        before['specifications']['extra'] = {}
    elif field == 'correction':
        before['specifications'][guard.SPECIFICATION]['specification'] = {
            'protocols': {'error_correction': {'default_policy': True}}}
    elif field == 'water-association':
        before['associations'][guard.WATER_ENTRY][guard.SPECIFICATION] = 5
    else:
        key = {'water-status': 'status', 'water-history': 'history',
               'water-tensor': 'tensor_sha256', 'water-envelope': 'native_envelope_sha256'}[field]
        before['records'][str(guard.WATER_RECORD)][key] = [] if key == 'history' else 'wrong'
    with pytest.raises(ValueError):
        guard.validate_owned(before, **policy)


def test_entry_then_singleton_record_are_only_allowed_deltas():
    before, _ = baseline()
    guard.assert_append_delta(before, appended(before), entry_name='methanol-new')
    guard.assert_append_delta(before, appended(before, record_id=123), entry_name='methanol-new', record_id=123)


@pytest.mark.parametrize('changed', ['old-record', 'old-association', 'old-entry',
                                    'metadata', 'specifications', 'extra-entry',
                                    'extra-record', 'deleted-record', 'new-association'])
def test_all_preexisting_state_is_immutable(changed):
    before, _ = baseline()
    after = appended(before, record_id=123)
    if changed == 'old-record':
        after['records'][str(guard.WATER_RECORD)]['status'] = 'waiting'
    elif changed == 'old-association':
        after['associations'][guard.WATER_ENTRY][guard.SPECIFICATION] = None
    elif changed == 'old-entry':
        after['entries'][guard.WATER_ENTRY]['attributes']['geometry_sha256'] = 'different'
    elif changed == 'metadata':
        after['metadata']['owner_group'] = 'changed'
    elif changed == 'specifications':
        after['specifications'][guard.SPECIFICATION]['description'] = 'changed'
    elif changed == 'extra-entry':
        after['entries']['unexpected'] = {}
    elif changed == 'extra-record':
        after['records']['999'] = {}
    elif changed == 'deleted-record':
        del after['records'][str(guard.WATER_RECORD)]
    else:
        after['associations']['methanol-new'][guard.SPECIFICATION] = guard.WATER_RECORD
    with pytest.raises(ValueError):
        guard.assert_append_delta(before, after, entry_name='methanol-new', record_id=123)


@pytest.mark.parametrize('changed', ['name', 'monomer', 'geometry', 'next', 'incomplete'])
def test_no_existing_entry_identity_reuse_or_unqualified_next_pilot(changed):
    before, _ = baseline()
    new = draft()
    if changed == 'name':
        new['entry']['name'] = guard.WATER_ENTRY
    elif changed == 'monomer':
        new['entry']['attributes']['monomer_id'] = 'monomer-0001'
    elif changed == 'geometry':
        new['entry']['attributes']['geometry_sha256'] = 'water-geometry'
    elif changed == 'next':
        new['monomer_id'] = 'monomer-0006'
    else:
        before['records'][str(guard.WATER_RECORD)]['status'] = 'error'
    with pytest.raises(ValueError):
        guard.validate_new_entry(before, new)


def test_multiple_preexisting_records_are_all_preserved_not_just_water():
    before, _ = baseline()
    before['entries']['another'] = {'attributes': {'monomer_id': 'prior', 'geometry_sha256': 'prior'}}
    before['associations']['another'] = {guard.SPECIFICATION: 9}
    before['records']['9'] = {'id': 9, 'status': 'complete', 'body_sha256': 'kept'}
    after = appended(before, record_id=123)
    guard.assert_append_delta(before, after, entry_name='methanol-new', record_id=123)
    del after['records']['9']
    with pytest.raises(ValueError):
        guard.assert_append_delta(before, after, entry_name='methanol-new', record_id=123)


def test_fresh_record_uses_effective_per_entry_keywords_not_only_base_specification():
    spec = {'keywords': {'reference': 'rhf'}, 'protocols': {'error_correction': {'default_policy': False}}}
    entry = {'additional_keywords': {'function_kwargs': {'native_monomer': 'new-request'}}}
    effective = copy.deepcopy(spec)
    effective['keywords'].update(entry['additional_keywords'])
    record = dict(status='waiting', history=[], task={'compute_tag': 'owned'}, specification=effective)
    guard.validate_fresh_record(record, tag='owned', specification=spec, entry=entry)
    with pytest.raises(ValueError):
        guard.validate_fresh_record(dict(record, specification=spec), tag='owned', specification=spec, entry=entry)
    with pytest.raises(ValueError):
        guard.validate_fresh_record(dict(record, history=[{}]), tag='owned', specification=spec, entry=entry)


def test_stored_entry_uses_real_raw_geometry_without_repairing_cached_symmetry():
    expected = dict(name='new', attributes={'reference': 'attributes-only'}, additional_keywords={'runtime': 'new'},
                    molecule=dict(symbols=['O', 'H', 'H'], geometry=[0., 0., 0., 1., 0., 0., 0., 1., 0.],
                                  molecular_charge=0, molecular_multiplicity=1, fix_com=True,
                                  fix_orientation=True, fix_symmetry='c1'))
    stored = copy.deepcopy(expected)
    stored['molecule']['geometry'] = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    stored['molecule']['fix_symmetry'] = None
    guard.validate_stored_entry(stored, expected)
    assert stored['molecule']['fix_symmetry'] is None
    stored['molecule']['geometry'][0][0] = .1
    with pytest.raises(ValueError):
        guard.validate_stored_entry(stored, expected)


def test_snapshot_proves_absences_through_fresh_forced_enumerations_not_getter_caches():
    from types import SimpleNamespace as NS
    class Model:
        def __init__(self, body):
            self.body = body
        def dict(self, **kw):
            return self.body
    class Record(Model):
        id, molecule_id = 1, 2
        status, extras, compute_history = NS(value='complete'), {}, []
        specification = {}
        def __init__(self):
            super().__init__({'id': 1, 'task': None, 'molecule': {
                'symbols': ['H'], 'geometry': [0., 0., 0.]}})
    entry = Model({'name': 'entry', 'attributes': {}})
    entry.name = 'entry'
    class Dataset:
        entry_names = ['entry', 'empty-entry']
        specifications = {'spec': Model({'specification': {}})}
        def fetch_specifications(self, **kw):
            assert kw == {'force_refetch': True}
        def fetch_entry_names(self):
            pass
        def iterate_entries(self, **kw):
            assert kw == {'force_refetch': True}
            empty = Model({'name': 'empty-entry', 'attributes': {}})
            empty.name = 'empty-entry'
            return iter([entry, empty])
        def iterate_records(self, **kw):
            assert kw['force_refetch'] is True and kw['include'] == ['compute_history', 'task', 'molecule']
            return iter([('entry', 'spec', Record())])
        def get_entry(self, *a, **kw):
            raise AssertionError('cached entry getter forbidden for snapshot')
        def get_record(self, *a, **kw):
            raise AssertionError('cached absence fallback forbidden for snapshot')
    class Client:
        def make_request(self, method, path, response):
            assert method == 'get' and path == 'api/v1/datasets/353'
            return {'id': 353}
        def get_dataset_by_id(self, identifier):
            assert identifier == 353
            return Dataset()
        def query_singlepoints(self, **kw):
            assert kw == {'dataset_id': 353}
            return iter([NS(id=1)])
    snapshot = guard.snapshot_dataset(Client())
    assert snapshot['associations'] == {'entry': {'spec': 1}, 'empty-entry': {'spec': None}}
    assert set(snapshot['records']) == {'1'}
    assert snapshot['records']['1']['body_sha256'] == guard.digest(Record().body)


def test_compressed_record_task_bytes_are_hashed_not_decompressed_or_serialized_raw():
    value = guard.plain({'task': {'compute_tag': 'owned', 'function_kwargs_compressed': b'fixture'}})
    compressed = value['task']['function_kwargs_compressed']
    assert compressed['size_bytes'] == 7 and len(compressed['bytes_sha256']) == 64
    assert value['task']['compute_tag'] == 'owned'
