"""Real QCPortal0.63 stack, mocked HTTP/transactions only; never a live write."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict

import pytest
import requests
from qcportal.client_base import PortalRequestError
from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fresh = load('fresh_transport_fixture', 'dataset_fresh_cb63305.py')
legacy_fixtures = load('reviewed_transport_fixtures', 'transport_tests.py')
ENDPOINTS = ['api/v1/datasets/singlepoint',
    'api/v1/datasets/singlepoint/9999/entries/bulkCreate',
    'api/v1/datasets/singlepoint/9999/specifications',
    'api/v1/datasets/singlepoint/9999/submit']


def client(results, *, producer='cb63305'):
    value = object.__new__(fresh.client_class(
        **({} if producer == 'cb63305' else dict(producer=producer))))
    original = legacy_fixtures.client(results, adapted=False)
    value.__dict__.update(original.__dict__)
    return value


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('result', [(401, {'msg': 'Token has expired'}),
    requests.exceptions.ConnectionError('synthetic lost response')])
def test_every_fresh_write_disables_both_retry_paths(endpoint, result):
    value = client([result])
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        value.make_request('post', endpoint, Dict[str, Any], body={'synthetic': 'not a live request'})
    assert len(value._req_session.sent) == 1 and not value.refreshes


@pytest.mark.parametrize('endpoint', [
    'api/v1/datasets/singlepoint/353/submit', 'api/v1/datasets/353',
    'api/v1/datasets/singlepoint/353/entries/bulkCreate'])
def test_historical_dataset_mutations_are_rejected_before_http(endpoint):
    value = client([])
    with pytest.raises(ValueError, match='historical dataset353'):
        value.make_request('post', endpoint, Dict[str, Any], body={})
    assert not value._req_session.sent


def draft(*, producer='cb63305'):
    result = dict(schema='native-expanded-molecular-pilot/1',
        status='PREPARED_NOT_REGISTERED_OR_SUBMITTED', monomer_id='monomer-0006',
        dataset_name='S66x8-native-CCASP-cb63305f41-request3-v1',
        specification_name='pbe0-atz-native-request3', producer={'synthetic': 'unit-only'},
        descriptor_bank_sha256='synthetic-only',
        queue_policy=dict(find_existing=False, existing_ok=False, concurrency=1, maximum_records=1),
        specification=dict(program='psi4', driver='properties', method='pbe0', basis='aug-cc-pvtz',
                           keywords={'reference': 'rhf'}),
        entry=dict(name='synthetic-nitrogen-pilot', attributes=dict(monomer_id='monomer-0006',
            sapt_reference_components_kcal_mol={'SAPT0/adz DISP': -1.}),
            additional_keywords=dict(function_kwargs={'native_monomer': {'schema': 'native-monomer-request/3'}}),
            molecule=dict(symbols=['C', 'N']+['H']*5, geometry=[
                v for i in range(7) for v in (i*3., 0., 0.)],
                molecular_charge=0, molecular_multiplicity=1,
                fix_com=True, fix_orientation=True, fix_symmetry='c1')))
    if producer == 'ed561666':
        pins = fresh.helper('pilot_cb63305').producer_pins(producer)
        result.update(schema=pins['schema'], producer_profile=producer, dataset_mode='fresh',
            monomer_id=pins['monomer'], dataset_name=pins['dataset_name'],
            runtime_input_sha256=pins['runtime_input_sha256'],
            descriptor_bank_sha256=pins['bank_sha256'],
            descriptor_allocation_sha256=pins['allocation_sha256'],
            execution_profile={'name': 's40'})  # transport fixture, not scientific admission
        result['entry']['attributes']['monomer_id'] = pins['monomer']
    return result


def transaction(case=None, *, producer='cb63305'):
    value = client([], producer=producer)
    data, calls, saved = draft(producer=producer), [], []
    dataset = NS(id=9999, name=data['dataset_name'], _client=value)
    def create(*args, **kwargs):
        calls.append(('create', args, kwargs))
        return dataset
    def entries(rows):
        calls.append(('entries', rows))
        return InsertMetadata(inserted_idx=[0])
    def specification(name, model):
        calls.append(('specification', name, model))
        return InsertMetadata(inserted_idx=[0])
    def submit(**kwargs):
        calls.append(('submit', kwargs))
        return InsertCountsMetadata(n_inserted=1, n_existing=0)
    value.add_dataset = create
    dataset.add_entries, dataset.add_specification, dataset.submit = entries, specification, submit
    stored = NS(attributes=data['entry']['attributes'],
                additional_keywords=data['entry']['additional_keywords'])
    dataset.get_entry = lambda *a, **kw: stored
    dataset.get_record = lambda *a, **kw: None
    if case == 'old_dataset':
        dataset.id = 353
    elif case == 'protected354':
        dataset.id = 354
    elif case == 'wrong_name':
        dataset.name = 'foreign'
    elif case == 'different_client':
        dataset._client = client([], producer=producer)
    elif case == 'changed_attributes':
        stored.attributes = {}
    elif case == 'existing_record':
        dataset.get_record = lambda *a, **kw: NS(id=1)
    elif case == 'existing_entry':
        dataset.add_entries = lambda _: InsertMetadata(existing_idx=[0])
    elif case == 'lost_create_response':
        def lost(*args, **kwargs):
            calls.append(('create', args, kwargs))
            raise requests.exceptions.ConnectionError('synthetic ambiguous write')
        value.add_dataset = lost
    save = lambda name, record: saved.append((name, copy.deepcopy(record)))
    return value, dataset, data, calls, saved, save


def test_new_dataset_singleton_preserves_attributes_and_never_reuses_records():
    value, expected, data, calls, saved, save = transaction()
    dataset, specification = fresh.register_once(value, data, 'ccasp-native-cb63305-monomer-0006-123', save)
    assert dataset is expected
    assert calls[0][2]['existing_ok'] is False
    assert calls[1][1][0].attributes == data['entry']['attributes']
    assert specification.dict()['protocols']['error_correction']['default_policy'] is False
    fresh.submit_once(value, dataset, data, 'ccasp-native-cb63305-monomer-0006-123', save)
    submitted = calls[-1][1]
    assert submitted == dict(entry_names=[data['entry']['name']],
        specification_names=['pbe0-atz-native-request3'],
        compute_tag='ccasp-native-cb63305-monomer-0006-123', find_existing=False)
    assert [row[0] for row in saved] == ['dataset-create-attempt.json', 'dataset-created.json',
        'entry-insert-attempt.json', 'specification-insert-attempt.json', 'record-submit-attempt.json']


@pytest.mark.parametrize('case', ['old_dataset', 'wrong_name', 'different_client',
    'changed_attributes', 'existing_record', 'existing_entry', 'lost_create_response'])
def test_failed_or_ambiguous_registration_stops_without_submission_or_retry(case):
    value, dataset, data, calls, saved, save = transaction(case)
    with pytest.raises((ValueError, RuntimeError, requests.exceptions.ConnectionError)):
        fresh.register_once(value, data, 'ccasp-native-cb63305-monomer-0006-123', save)
    assert sum(row[0] == 'create' for row in calls) == 1
    assert not any(row[0] == 'submit' for row in calls)
    assert not any(row[0] == 'record-submit-attempt.json' for row in saved)


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('result', [(401, {'msg': 'Token has expired'}),
    requests.exceptions.ConnectionError('synthetic cap32 lost response')])
def test_ed_writes_are_one_shot_without_token_or_network_replay(endpoint, result):
    value = client([result], producer='ed561666')
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        value.make_request('post', endpoint, Dict[str, Any], body={'synthetic': 'not live'})
    assert len(value._req_session.sent) == 1 and not value.refreshes


@pytest.mark.parametrize('result', [(401, {'msg': 'Token has expired'}),
    requests.exceptions.ConnectionError('synthetic manager lost response')])
def test_ed_manager_claim_disables_token_recursion_and_network_retry(result):
    from qcportal.manager_client import ManagerClient
    from qcportal.managers.models import ManagerName
    value = object.__new__(ManagerClient)
    value.__dict__.update(legacy_fixtures.client([result], adapted=False).__dict__)
    value.manager_name_data = ManagerName(cluster='synthetic', hostname='unit-only', uuid='123')
    fresh.guard_manager_client(value)
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        value.claim(programs={'psi4': ['synthetic']},
                    tags=['ccasp-native-ed561666-monomer-0047-123'], limit=1)
    assert len(value._req_session.sent) == 1 and not value.refreshes
    with pytest.raises(ValueError, match='unmodified'):
        fresh.guard_manager_client(value)


def test_ed_manager_source_is_bound_without_changing_legacy_transport_inventory():
    assert 'manager_client.py' not in fresh.transport_binding()['source_sha256']
    binding = fresh.transport_binding(producer='ed561666')
    assert binding['manager_source_sha256'] == {
        'manager_client.py': 'fd4147c5cd163c4431d24b7afd5b1e2edd398d452aa3cff6fc79450ed127e6b3'}


@pytest.mark.parametrize('endpoint', [
    'api/v1/datasets/singlepoint/353/submit', 'api/v1/datasets/354',
    'api/v1/datasets/singlepoint/354/entries/bulkCreate',
    'api/v1/datasets/singlepoint/354?synthetic=query',
    'api/v1/records/bulkModify', 'api/v1/records/bulkDelete',
    'api/v1/records/singlepoint/5234317', 'api/v1/records/5234316/comments',
    '/api/v1/datasets/354', 'api/v1/datasets/%33%35%34',
    'api/v1/datasets/singlepoint/9999/../354/submit',
    'api//v1/datasets/354',
])
def test_ed_protected_dataset_and_all_record_mutations_stop_before_http(endpoint):
    value = client([], producer='ed561666')
    with pytest.raises(ValueError):
        value.make_request('post', endpoint, Dict[str, Any],
            body={'record_ids': [5234314, 5234315, 5234316, 5234317]})
    assert not value._req_session.sent


@pytest.mark.parametrize('endpoint', ['api/v1/records/bulkGet', 'api/v1/records/singlepoint/bulkGet',
                                     'api/v1/records/query'])
def test_ed_keeps_read_only_record_posts_without_permitting_mutation(endpoint):
    value = client([(200, {'synthetic_read': True})], producer='ed561666')
    assert value.make_request('post', endpoint, Dict[str, Any], body={'ids': [9999]}) == {
        'synthetic_read': True}
    assert len(value._req_session.sent) == 1


def test_ed_client_is_separate_while_default_client_retains_owned354_route():
    cb = client([(200, {'synthetic_cb_owned': True})])
    ed = client([], producer='ed561666')
    assert type(cb) is not type(ed)
    fresh.require_client(cb)
    fresh.require_client(ed, producer='ed561666')
    with pytest.raises(ValueError):
        fresh.require_client(cb, producer='ed561666')
    with pytest.raises(ValueError):
        fresh.require_client(ed)
    assert cb.make_request('post', 'api/v1/datasets/singlepoint/354/submit',
                          Dict[str, Any], body={}) == {'synthetic_cb_owned': True}
    assert 'producer' not in fresh.transport_binding()
    assert fresh.transport_binding(producer='ed561666')['producer'] == 'ed561666'


def test_ed_s40_is_a_fresh_transaction_not_owned354_append():
    value, expected, data, calls, saved, save = transaction(producer='ed561666')
    tag = 'ccasp-native-ed561666-monomer-0047-123'
    dataset, _ = fresh.register_once(value, data, tag, save, producer='ed561666')
    assert dataset is expected and data['execution_profile']['name'] == 's40'
    assert calls[0][2]['existing_ok'] is False
    assert calls[1][1][0].attributes == data['entry']['attributes']
    fresh.submit_once(value, dataset, data, tag, save, producer='ed561666')
    assert calls[-1][1]['find_existing'] is False and calls[-1][1]['compute_tag'] == tag
    assert all(row[1].get('dataset_id') not in (353, 354) for row in saved)


@pytest.mark.parametrize('case', ['old_dataset', 'protected354', 'wrong_name', 'different_client',
    'changed_attributes', 'existing_record', 'existing_entry', 'lost_create_response'])
def test_ed_failed_or_ambiguous_registration_never_submits_or_retries(case):
    value, dataset, data, calls, saved, save = transaction(case, producer='ed561666')
    with pytest.raises((ValueError, RuntimeError, requests.exceptions.ConnectionError)):
        fresh.register_once(value, data, 'ccasp-native-ed561666-monomer-0047-123',
                            save, producer='ed561666')
    assert sum(row[0] == 'create' for row in calls) == 1
    assert not any(row[0] == 'submit' for row in calls)


@pytest.mark.parametrize('change', [dict(dataset_mode='owned354'), dict(dataset_id=354),
    dict(owned_checkpoint='not permitted'), dict(producer_profile='cb63305')])
def test_ed_rejects_append_markers_before_any_api_operation(change):
    value, _, data, calls, _, save = transaction(producer='ed561666')
    data.update(change)
    with pytest.raises(ValueError):
        fresh.register_once(value, data, 'ccasp-native-ed561666-monomer-0047-123',
                            save, producer='ed561666')
    assert not calls and not value._req_session.sent
