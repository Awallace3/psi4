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


def client(results):
    value = object.__new__(fresh.client_class())
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


def draft():
    return dict(schema='native-expanded-molecular-pilot/1',
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


def transaction(case=None):
    value = client([])
    data, calls, saved = draft(), [], []
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
    elif case == 'wrong_name':
        dataset.name = 'foreign'
    elif case == 'different_client':
        dataset._client = client([])
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
