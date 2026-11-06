"""Real QCPortal models/client with in-memory fake HTTP. No network or science."""
import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

import pytest
import requests
from qcportal.client_base import PortalRequestError
from qcportal.exceptions import AuthenticationFailure
from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata
from qcportal.singlepoint import SinglepointRecord, SinglepointDatasetNewEntry
from qcportal.dataset_models import DatasetSubmitBody

_spec = importlib.util.spec_from_file_location('_wave_sdk_fixture', Path(__file__).with_name('test_bank_wave.py'))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
load, fixture = _fixtures.load, _fixtures.fixture


def transaction(monkeypatch, tmp_path, results=(), *, readonly=False):
    f, transport = fixture(tmp_path), load('bank_wave_transport')
    monkeypatch.setattr(transport.helper('bank_wave_campaign'), 'revalidate_binding', lambda binding: None)
    scope = None
    if readonly:
        cls = transport.readonly_client_class(f.binding, monomer_id=f.binding['monomer_id'])
    else:
        binding = copy.deepcopy(f.binding)
        binding.update(registration=None, release_permit=None)
        monkeypatch.setenv('SLURM_JOB_ID', '8000')
        scope = transport.WaveScope(binding, f.cases, f.reg['members'], f.reg['registration_role'])
        scope.select(f.plan['member_ids'][0])
        cls = transport.client_class(scope)
    client = object.__new__(cls)
    legacy = load('transport_cb63305_tests').legacy_fixtures
    client.__dict__.update(legacy.client(list(results), adapted=False).__dict__)
    return f, transport, scope, client


def payload(f, role):
    mid = f.plan['member_ids'][0]
    if role == 'entries/bulkCreate':
        return [SinglepointDatasetNewEntry(**f.cases[mid]['entry'])]
    return DatasetSubmitBody(entry_names=[f.cases[mid]['entry']['name']],
        specification_names=[f.cases[mid]['specification_name']],
        compute_tag=f.reg['members'][mid]['compute_tag'], find_existing=False)


@pytest.mark.parametrize('role', ['entries/bulkCreate', 'submit'])
@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic lost reply')])
def test_registration_write_is_one_attempt_even_with_lost_reply(monkeypatch, tmp_path, role, error):
    f, transport, scope, client = transaction(monkeypatch, tmp_path, [error])
    if role == 'submit':
        scope.request('post', 'api/v1/datasets/singlepoint/357/entries/bulkCreate')
    endpoint = 'api/v1/datasets/singlepoint/357/'+role
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        client.make_request('post', endpoint, Dict[str, Any], body=payload(f, role))
    with pytest.raises(ValueError, match='already attempted'):
        client.make_request('post', endpoint, Dict[str, Any], body=payload(f, role))
    assert len(client._req_session.sent) == 1 and not client.refreshes
    transport.require_registration_client(client, scope)


def test_exact_one_entry_and_submit_typed_payloads(monkeypatch, tmp_path):
    results = [(200, asdict(InsertMetadata(inserted_idx=[0]))),
               (200, asdict(InsertCountsMetadata(n_inserted=1, n_existing=0)))]
    f, _, scope, client = transaction(monkeypatch, tmp_path, results)
    for role, model in (('entries/bulkCreate', InsertMetadata), ('submit', InsertCountsMetadata)):
        response = client.make_request('post', 'api/v1/datasets/singlepoint/357/'+role,
                                       model, body=payload(f, role))
        assert response.n_inserted == 1 and response.n_existing == 0
    scope.finish(f.plan['member_ids'][0])
    scope.select(f.plan['member_ids'][1])
    assert len(client._req_session.sent) == 2


@pytest.mark.parametrize('change', ['multiple', 'runtime', 'tag', 'find_existing', 'all_entries'])
def test_wrong_typed_payload_consumes_attempt_without_http(monkeypatch, tmp_path, change):
    f, _, scope, client = transaction(monkeypatch, tmp_path)
    role = 'entries/bulkCreate' if change in ('multiple', 'runtime') else 'submit'
    body = payload(f, role)
    if role == 'submit':
        scope.request('post', 'api/v1/datasets/singlepoint/357/entries/bulkCreate')
    if change == 'multiple':
        body *= 2
    elif change == 'runtime':
        body[0].additional_keywords['function_kwargs']['native_monomer']['resources']['max_work'] += 1
    elif change == 'tag':
        body.compute_tag += '-foreign'
    elif change == 'find_existing':
        body.find_existing = True
    else:
        body.entry_names = None
    endpoint = 'api/v1/datasets/singlepoint/357/'+role
    with pytest.raises(ValueError):
        client.make_request('post', endpoint, Dict[str, Any], body=body)
    with pytest.raises(ValueError, match='already attempted'):
        client.make_request('post', endpoint, Dict[str, Any], body=payload(f, role))
    assert not client._req_session.sent


@pytest.mark.parametrize('readonly', [False, True])
@pytest.mark.parametrize('endpoint', ['api/v1/datasets/singlepoint', 'api/v1/datasets/singlepoint/357/specifications',
    'api/v1/datasets/singlepoint/353/entries/bulkCreate', 'api/v1/records/reset',
    'api/v1/datasets/singlepoint/357/delete'])
def test_unapproved_mutation_has_no_http(monkeypatch, tmp_path, readonly, endpoint):
    _, _, _, client = transaction(monkeypatch, tmp_path, readonly=readonly)
    with pytest.raises(ValueError):
        client.make_request('post', endpoint, Dict[str, Any], body={})
    assert not client._req_session.sent


@pytest.mark.parametrize('readonly', [False, True])
@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic auth loss')])
def test_constructor_login_has_one_attempt(monkeypatch, tmp_path, readonly, error):
    _, _, _, client = transaction(monkeypatch, tmp_path, readonly=readonly)
    session = load('transport_cb63305_tests').legacy_fixtures.MockSession([error])
    monkeypatch.setattr(requests, 'Session', lambda: session)
    with pytest.raises((AuthenticationFailure, requests.exceptions.ConnectionError)):
        type(client)('http://transport-fixture.invalid', username='synthetic', password='unit-only',
                     show_motd=False)
    assert len(session.sent) == 1
    assert urlsplit(session.sent[0][0].url).path == '/auth/v1/login'


@pytest.mark.parametrize('readonly', [False, True])
@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic refresh loss')])
def test_proactive_refresh_cannot_retry_or_fall_back(monkeypatch, tmp_path, readonly, error):
    import time
    _, _, _, client = transaction(monkeypatch, tmp_path, [error], readonly=readonly)
    client.__dict__.pop('_refresh_JWT_token', None)
    client._jwt_access_exp, client._jwt_refresh_exp = 1, time.time()+3600
    client._jwt_refresh_token = 'synthetic-not-real'
    with pytest.raises((AuthenticationFailure, requests.exceptions.ConnectionError)):
        client.make_request('get', 'api/v1/datasets/357', Dict[str, Any])
    assert len(client._req_session.sent) == 1
    assert urlsplit(client._req_session.sent[0][0].url).path == '/auth/v1/refresh'


def test_readonly_scope_cache_is_exact_type_not_subclass(monkeypatch, tmp_path):
    f, transport, _, client = transaction(monkeypatch, tmp_path, readonly=True)
    assert transport.readonly_client_class(f.binding, monomer_id=f.binding['monomer_id']) is type(client)
    transport.require_readonly_client(client, f.binding, monomer_id=f.binding['monomer_id'])
    class Forged(type(client)):
        pass
    with pytest.raises(ValueError):
        transport.require_readonly_client(object.__new__(Forged), f.binding,
                                          monomer_id=f.binding['monomer_id'])
    f.binding['release_permit_sha256'] = 'a'*64
    with pytest.raises(ValueError):
        transport.require_readonly_client(client, f.binding, monomer_id=f.binding['monomer_id'])


def test_real_portal_record_alias_lifecycle_and_immutable_body():
    transport = load('bank_wave_transport')
    record = SinglepointRecord(id=123, is_service=False, status='waiting',
        created_on='2026-11-04T12:00:00+00:00', modified_on='2026-11-04T12:00:00+00:00',
        molecule_id=9997, specification=dict(program='psi4', driver='properties',
            method='pbe0', basis='aug-cc-pvtz', keywords={}), extras={}, compute_history=[],
        task=dict(id=456, record_id=123, compute_tag='synthetic', compute_priority=1,
                  required_programs=['psi4'], function='synthetic', function_kwargs_compressed=b'fixture'))
    body = transport.helper('dataset_guard').plain(record)
    assert 'task' in body and 'compute_history' in body and 'task_' not in body
    completed = record.copy(deep=True, update=dict(status='complete', manager_name='synthetic-manager',
        properties={'return_energy': -1}, extras={'psi4:native_monomer': {'synthetic': True}},
        task_=None, compute_history_=[dict(id=789, record_id=123, status='complete',
            modified_on='2026-11-04T13:00:00+00:00', manager_name='synthetic-manager')]))
    after = transport.helper('dataset_guard').plain(completed)
    assert transport.stable_record_identity(body) == transport.stable_record_identity(after)
    assert transport.helper('dataset_guard').digest(body) != transport.helper('dataset_guard').digest(after)
    for key in ('molecule_id', 'name', 'description', 'created_on'):
        modified = dict(after, **{key: 'changed'})
        assert transport.stable_record_identity(body) != transport.stable_record_identity(modified)
    body['extras']['user-owned'] = {'preserve': 1}
    after['extras']['user-owned'] = {'preserve': 1}
    assert transport.stable_record_identity(body) == transport.stable_record_identity(after)
    for change in ('add', 'change', 'remove'):
        modified = copy.deepcopy(after)
        if change == 'add':
            modified['extras']['unknown-new'] = 1
        elif change == 'change':
            modified['extras']['user-owned']['preserve'] = 2
        else:
            modified['extras'].pop('user-owned')
        assert transport.stable_record_identity(body) != transport.stable_record_identity(modified)


@pytest.mark.parametrize('failure', [None, 'entry_lost', 'existing_record'])
def test_registration_adapter_saves_actual_sequential_captures(monkeypatch, tmp_path, failure):
    f, transport = fixture(tmp_path), load('bank_wave_transport')
    controller = load('bank_wave_controller')
    controller._MODULES['bank_wave_transport'] = transport
    monkeypatch.setattr(controller.helper('bank_wave_campaign'), 'revalidate_binding', lambda binding: None)
    monkeypatch.setattr(transport.helper('bank_wave_campaign'), 'revalidate_binding', lambda binding: None)
    store = controller.helper('bank_store')
    monkeypatch.setattr(store, 'INTENTS', tmp_path/'intents')
    monkeypatch.setenv('SLURM_JOB_ID', '8000')
    binding = copy.deepcopy(f.binding)
    binding.update(registration=None, release_permit=None)
    members = {mid: {key: f.reg['members'][mid][key] for key in
        ('science_job_id', 'compute_tag', 'case_sha256', 'worker_preflight_sha256')}
        for mid in f.plan['member_ids']}
    scope = transport.WaveScope(binding, f.cases, members, f.reg['registration_role'])
    client = object.__new__(transport.client_class(scope))
    client.get_server_information = lambda: dict(name='CDS QCFractal Server', version='0.70')
    client.query_managers = lambda **kwargs: []
    observations = [f.previous['snapshot']]
    for mid in f.plan['member_ids']:
        observations.extend([f.captures[mid]['before'], f.captures[mid]['entry'],
                             f.captures[mid]['entry'], f.captures[mid]['record']])
    observations.append(f.registered)
    observed = iter(observations)
    monkeypatch.setattr(transport, '_snapshot_dataset', lambda *args: copy.deepcopy(next(observed)))
    calls, saved = [], {}
    def request(method, endpoint, model, body, **kwargs):
        scope.request(method, endpoint)
        calls.append((scope.active, endpoint.rsplit('/', 1)[-1]))
        assert kwargs['allow_retries'] is False
        if endpoint.endswith('bulkCreate'):
            assert len(body) == 1 and body[0].name == f.cases[scope.active]['entry']['name']
            if failure == 'entry_lost':
                raise requests.exceptions.ConnectionError('synthetic accepted-but-lost entry reply')
            return InsertMetadata(inserted_idx=[0])
        assert body.find_existing is False
        if failure == 'existing_record':
            return InsertCountsMetadata(n_inserted=0, n_existing=1)
        return InsertCountsMetadata(n_inserted=1, n_existing=0)
    client.make_request = request
    if failure:
        with pytest.raises((requests.exceptions.ConnectionError, RuntimeError)):
            controller.register_once(client, scope, save=lambda name, value: saved.update({name: value}))
        assert 'registration.json' not in saved and 'release-permit.json' not in saved
        assert len(calls) == (1 if failure == 'entry_lost' else 2)
        # Real canonical reservation survives; no automatic clearing/retry.
        assert (store.INTENTS/'357'/f.plan['predecessor_snapshot_sha256']/'intent.json').exists()
    else:
        result = controller.register_once(client, scope, save=lambda name, value: saved.update({name: value}))
        assert len(calls) == 16
        assert result['registered_snapshot_sha256'] == f.reg['registered_snapshot_sha256']
        assert saved['registration/registered-snapshot.json'] == f.registered
        for mid in f.plan['member_ids']:
            for role, filename in (('before', 'owned-before-entry.json'), ('entry', 'owned-after-entry.json'),
                                   ('record', 'owned-after-record.json')):
                assert saved['registration/'+mid+'/'+filename] == f.captures[mid][role]
        assert 'release-permit.json' not in saved
