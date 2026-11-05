"""Real reviewed QCPortal with fake HTTP; no server or scientific execution."""
import copy
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

import pytest
import requests
from qcportal.cache import PortalCache
from qcportal.client_base import PortalRequestError
from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_http_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transaction(results=(), *, mode='fresh'):
    module = load('bank_dataset')
    cases, inputs, pool = load('test_grid400_bank_case').fixture()
    # Non-overlapping toy coordinates only; no wavefunction or quantum result.
    entries = json.loads(inputs['entries'])
    geometry = [0., 0., 0., 3., 0., 0., 6., 0., 0.]
    index = 0 if mode == 'fresh' else 1
    entries['entries'][index]['molecule']['geometry'] = geometry
    inputs['entries'] = json.dumps(entries).encode()
    row, report = pool['bank']['monomers'][index], pool['allocation']['reports'][index]
    row['runtime_input']['molecule']['geometry'] = geometry
    row['runtime_input_sha256'] = cases.helper('prepare_descriptor_bank_cb63305').json_digest(
        row['runtime_input'])
    report['runtime_input_sha256'] = row['runtime_input_sha256']
    pool['bank']['input_sha256']['entries'] = hashlib.sha256(inputs['entries']).hexdigest()
    case = cases.prepare_case(inputs, pool, row['monomer_id'])
    tag = 'ccasp-native-grid400-bank-'+row['monomer_id']+'-123'
    raw = None
    if mode == 'append':
        snapshot = dict(metadata=dict(id=9999))
        checkpoint = dict(schema='native-audited-bank-checkpoint/1',
            status='READ_ONLY_INITIAL_BASELINE', campaign=case['campaign'], dataset_id=9999,
            eligibility_sha256=case['eligibility_sha256'], bootstrap_context_sha256='b'*64,
            generation=1, snapshot=snapshot,
            snapshot_sha256=module.helper('bank_transport').helper('dataset_guard').digest(snapshot))
        raw = json.dumps(checkpoint).encode()
        case.update(dataset_mode='append', dataset_id=9999,
            bootstrap_context_sha256='b'*64, checkpoint_sha256=hashlib.sha256(raw).hexdigest())
    scope = module.prepare_transaction(case, copy.deepcopy(case), tag, checkpoint_raw=raw)
    cls = module.client_class(scope)
    client = object.__new__(cls)
    fixtures = load('transport_cb63305_tests')
    client.__dict__.update(fixtures.legacy_fixtures.client(list(results), adapted=False).__dict__)
    return module, scope, client, case


def payload(module, scope, case, role):
    from qcportal.dataset_models import DatasetAddBody, DatasetSubmitBody
    from qcportal.singlepoint import QCSpecification, SinglepointDatasetNewEntry
    from qcportal.singlepoint.dataset_models import SinglepointDatasetSpecification
    if role == 'create':
        return DatasetAddBody(name=case['dataset_name'], default_tag=scope.compute_tag,
            description='SYNTHETIC_ONLY', tagline='', tags=[], default_priority=1, extras={},
            existing_ok=False, provenance=dict(producer=case['producer'],
                descriptor_bank_sha256=case['descriptor_bank_sha256']))
    if role == 'entry':
        return [SinglepointDatasetNewEntry(**case['entry'])]
    if role == 'specification':
        return [SinglepointDatasetSpecification(name=case['specification_name'],
            specification=QCSpecification(**module.helper('dataset_fresh_cb63305').helper(
                'phoenix_pilot').execution_specification(case['specification'])))]
    return DatasetSubmitBody(entry_names=[case['entry']['name']],
        specification_names=[case['specification_name']], compute_tag=scope.compute_tag, find_existing=False)


@pytest.mark.parametrize('role', ['create', 'entry', 'specification', 'submit'])
@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic lost response')])
def test_failed_or_ambiguous_write_has_one_http_attempt_and_no_replay(role, error):
    module, scope, client, case = transaction([error])
    module.helper('bank_transport').require_client(client, scope)
    if role != 'create':
        scope.request('post', 'api/v1/datasets/singlepoint')
        scope.bind_created(9999)
    if role == 'submit':
        for suffix in ('entries/bulkCreate', 'specifications'):
            scope.request('post', 'api/v1/datasets/singlepoint/9999/'+suffix)
    endpoint = ('api/v1/datasets/singlepoint' if role == 'create' else
                'api/v1/datasets/singlepoint/9999/'+dict(
                    entry='entries/bulkCreate', specification='specifications', submit='submit')[role])
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        client.make_request('post', endpoint, Dict[str, Any], body=payload(module, scope, case, role))
    with pytest.raises(ValueError, match='already attempted'):
        client.make_request('post', endpoint, Dict[str, Any], body={'synthetic': True})
    assert len(client._req_session.sent) == 1 and not client.refreshes


@pytest.mark.parametrize('identifier', [353, 354, 355, 356, 9998])
def test_foreign_or_protected_target_refuses_before_http(identifier):
    _, scope, client, _ = transaction()
    scope.request('post', 'api/v1/datasets/singlepoint')
    scope.bind_created(9999)
    with pytest.raises(ValueError):
        client.make_request('post', f'api/v1/datasets/singlepoint/{identifier}/submit',
                            Dict[str, Any], body={})
    assert not client._req_session.sent


def test_real_positive_creation_entry_specification_and_fresh_record_submission():
    module, scope, client, case = transaction()
    client.cache = PortalCache(client.address, None, 16*1024**2)
    client.server_info = {'api_limits': defaultdict(lambda: 100)}
    client.api_limits = client.server_info['api_limits']
    entry = copy.deepcopy(case['entry'])
    entry['molecule']['id'] = 9997
    metadata = dict(id=9999, dataset_type='singlepoint', name=case['dataset_name'],
        description='Synthetic HTTP-only fixture', tagline='', tags=[], group='default',
        visibility=True, provenance={}, extras={}, default_tag=scope.compute_tag, default_priority=1,
        owner_user='synthetic-unit-user', owner_group=None)
    observed = []

    def response(request, **kwargs):
        route = urlsplit(request.url).path.lstrip('/')
        observed.append((request.method, route))
        client._req_session.sent.append((request, kwargs))
        if request.method == 'POST' and route == 'api/v1/datasets/singlepoint':
            body = 9999
        elif request.method == 'GET' and route == 'api/v1/datasets/9999':
            body = metadata
        elif route.endswith('/entries/bulkCreate') or (
                route.endswith('/specifications') and request.method == 'POST'):
            body = asdict(InsertMetadata(inserted_idx=[0]))
        elif route.endswith('/entries/bulkFetch'):
            body = {entry['name']: entry}
        elif route.endswith('/entry_names'):
            body = [entry['name']]
        elif route.endswith('/specifications/bulkFetch'):
            body = {case['specification_name']: dict(name=case['specification_name'],
                specification=case['specification'], description=None)}
        elif route.endswith('/records/bulkFetch'):
            body = []
        elif route.endswith('/submit'):
            body = asdict(InsertCountsMetadata(n_inserted=1, n_existing=0))
        else:
            raise AssertionError('unexpected bank HTTP route: '+str((request.method, route)))
        result = requests.Response()
        result.status_code, result.request = 200, request
        result.headers['Content-Type'] = 'application/json'
        result._content = json.dumps(body).encode()
        return result

    client._req_session.send = response
    saved = []
    save = lambda name, data: saved.append((name, data))
    dataset, _ = module.register_once(client, scope, case, save)
    assert dataset.id == scope.dataset_id == 9999
    result = module.submit_once(client, scope, dataset, case, save)
    assert result.success and result.n_inserted == 1 and result.n_existing == 0
    assert scope.attempted == {'create', 'entry', 'specification', 'submit'}
    for suffix in ('singlepoint', '9999/entries/bulkCreate', '9999/specifications', '9999/submit'):
        assert sum(method == 'POST' and route.endswith('/'+suffix) for method, route in observed) == 1
    assert not client.refreshes
    assert all(data.get('dataset_id') not in (353, 354, 355, 356) for _, data in saved)


@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic authentication failure')])
def test_real_constructor_login_failure_is_one_post(monkeypatch, error):
    from qcportal.exceptions import AuthenticationFailure
    _, _, client, _ = transaction()
    fixtures = load('transport_cb63305_tests')
    session = fixtures.legacy_fixtures.MockSession([error])
    monkeypatch.setattr(requests, 'Session', lambda: session)
    with pytest.raises((AuthenticationFailure, requests.exceptions.ConnectionError)):
        type(client)('http://transport-fixture.invalid', username='synthetic', password='unit-only',
                     show_motd=False)
    assert len(session.sent) == 1
    assert urlsplit(session.sent[0][0].url).path == '/auth/v1/login'


@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic proactive refresh failure')])
def test_real_proactive_refresh_failure_cannot_retry_or_fall_back_to_login(error):
    import time
    from qcportal.exceptions import AuthenticationFailure
    _, _, client, _ = transaction([error])
    client.__dict__.pop('_refresh_JWT_token', None)
    client._jwt_access_exp, client._jwt_refresh_exp = 1, time.time()+3600
    client._jwt_refresh_token = 'synthetic-not-a-real-token'
    with pytest.raises((AuthenticationFailure, requests.exceptions.ConnectionError)):
        client.make_request('get', 'api/v1/datasets/9999', Dict[str, Any])
    assert len(client._req_session.sent) == 1
    assert urlsplit(client._req_session.sent[0][0].url).path == '/auth/v1/refresh'


def test_append_posts_are_single_entry_and_single_fresh_record():
    results = [(200, asdict(InsertMetadata(inserted_idx=[0]))),
               (200, asdict(InsertCountsMetadata(n_inserted=1, n_existing=0)))]
    module, scope, client, case = transaction(results, mode='append')
    for role, suffix, response in (('entry', 'entries/bulkCreate', InsertMetadata),
                                   ('submit', 'submit', InsertCountsMetadata)):
        result = client.make_request('post', 'api/v1/datasets/singlepoint/9999/'+suffix,
                                     response, body=payload(module, scope, case, role))
        assert result.n_inserted == 1 and result.n_existing == 0
    assert len(client._req_session.sent) == 2 and scope.attempted == {'entry', 'submit'}


@pytest.mark.parametrize('role', ['entry', 'submit'])
@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic ambiguous append')])
def test_append_failure_cannot_replay_after_any_http_outcome(role, error):
    module, scope, client, case = transaction([error], mode='append')
    if role == 'submit':
        scope.request('post', 'api/v1/datasets/singlepoint/9999/entries/bulkCreate')
    endpoint = 'api/v1/datasets/singlepoint/9999/'+('submit' if role == 'submit' else 'entries/bulkCreate')
    body = payload(module, scope, case, role)
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        client.make_request('post', endpoint, Dict[str, Any], body=body)
    with pytest.raises(ValueError):
        client.make_request('post', endpoint, Dict[str, Any], body=body)
    assert len(client._req_session.sent) == 1 and not client.refreshes


@pytest.mark.parametrize('change', ['multiple_entries', 'different_entry', 'runtime',
                                  'find_existing', 'tag', 'unrestricted_submit'])
def test_scope_rejects_wrong_or_multiple_mutation_payloads_before_http(change):
    module, scope, client, case = transaction(mode='append')
    role = 'entry' if change in ('multiple_entries', 'different_entry', 'runtime') else 'submit'
    body = payload(module, scope, case, role)
    if role == 'entry':
        if change == 'multiple_entries':
            body *= 2
        elif change == 'different_entry':
            body[0].name = 'different'
        else:
            body[0].additional_keywords['function_kwargs']['native_monomer']['resources']['max_work'] += 1
    else:
        scope.request('post', 'api/v1/datasets/singlepoint/9999/entries/bulkCreate')
        if change == 'find_existing':
            body.find_existing = True
        elif change == 'tag':
            body.compute_tag = 'foreign'
        else:
            body.entry_names = None
    endpoint = 'api/v1/datasets/singlepoint/9999/'+('submit' if role == 'submit' else 'entries/bulkCreate')
    with pytest.raises(ValueError, match='payload'):
        client.make_request('post', endpoint, Dict[str, Any], body=body)
    assert not client._req_session.sent


def test_two_clients_cannot_repeat_a_write_under_the_same_scope():
    module, scope, first, case = transaction([(200, asdict(InsertMetadata(inserted_idx=[0])))], mode='append')
    second = object.__new__(type(first))
    second.__dict__.update(load('transport_cb63305_tests').legacy_fixtures.client([], adapted=False).__dict__)
    endpoint, body = 'api/v1/datasets/singlepoint/9999/entries/bulkCreate', payload(module, scope, case, 'entry')
    first.make_request('post', endpoint, InsertMetadata, body=body)
    with pytest.raises(ValueError, match='already attempted'):
        second.make_request('post', endpoint, InsertMetadata, body=body)
    assert len(first._req_session.sent) == 1 and not second._req_session.sent
