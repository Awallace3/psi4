"""Mandatory real QCPortal0.63 request-stack regression, with mocked HTTP only.

Run explicitly with the p4qca Python, not the unrelated p4_ci0.64 client.
No HTTP connection, credentials, QCArchive write or quantum work occurs.
"""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pytest
import requests
from qcportal import PortalClient
from qcportal.client_base import PortalRequestError
from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata

spec = importlib.util.spec_from_file_location(
    'actual_append_transport', Path(__file__).with_name('dataset_append.py'))
append = importlib.util.module_from_spec(spec)
spec.loader.exec_module(append)


class MockSession(requests.Session):
    def __init__(self, results):
        super().__init__()
        self.trust_env = False
        self.results, self.sent = list(results), []

    def send(self, request, **kwargs):
        self.sent.append((request, kwargs))
        if not self.results:
            raise AssertionError('unexpected extra transport send')
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        status, body = result
        response = requests.Response()
        response.status_code = status
        response.reason = 'Unauthorized' if status == 401 else 'OK'
        response.headers['Content-Type'] = 'application/json'
        response._content = json.dumps(body).encode()
        response.request = request
        return response


def client(results, *, adapted=True):
    cls = append.client_class() if adapted else PortalClient
    value = object.__new__(cls)
    value._req_session = MockSession(results)
    value.address = 'http://transport-fixture.invalid/'
    value.encoding, value.timeout = 'application/json', 1
    value._verify, value.debug_requests = True, False
    value._jwt_refresh_exp = value._jwt_access_exp = None
    value.retry_max, value.retry_delay = 3, 0
    value.retry_jitter_fraction, value.retry_backoff = 0, 1
    value.refreshes = []
    value._refresh_JWT_token = lambda: value.refreshes.append('refresh')
    return value


def entry():
    return dict(
        name='methanol-transport-fixture', attributes={'monomer_id': 'monomer-0004'},
        molecule=dict(symbols=['C', 'O', 'H', 'H', 'H', 'H'],
                      geometry=[0., 0., 0., 2.6, 0., 0., -1., 1.7, 0.,
                                -1., -1.7, 0., 0., 0., 2., 3.2, 1.2, 0.],
                      molecular_charge=0, molecular_multiplicity=1,
                      fix_com=True, fix_orientation=True, fix_symmetry='c1'))


def write(value, stage):
    if stage == 'entry':
        return append.insert_entry_once(value, entry())
    return append.submit_record_once(
        value, entry_name=entry()['name'], compute_tag='ccasp-native-s66-monomer-0004-123')


@pytest.mark.parametrize('stage', ['entry', 'record'])
def test_expired_token_is_one_post_and_hard_failure_not_recursive_auth_retry(stage):
    value = client([(401, {'msg': 'Token has expired'}),
                    requests.exceptions.ConnectionError('must never send this second POST')])
    with pytest.raises(PortalRequestError) as error:
        write(value, stage)
    assert error.value.status_code == 401
    assert len(value._req_session.sent) == 1 and not value.refreshes
    assert value._req_session.sent[0][0].method == 'POST'


@pytest.mark.parametrize('stage', ['entry', 'record'])
def test_lost_response_is_propagated_after_exactly_one_post(stage):
    value = client([requests.exceptions.ConnectionError('synthetic lost response')])
    with pytest.raises(requests.exceptions.ConnectionError):
        write(value, stage)
    assert len(value._req_session.sent) == 1 and not value.refreshes


@pytest.mark.parametrize('stage', ['entry', 'record'])
def test_real_serialization_and_metadata_validation_survive_adapter(stage):
    metadata = InsertMetadata(inserted_idx=[0]) if stage == 'entry' else InsertCountsMetadata(n_inserted=1, n_existing=0)
    value = client([(200, asdict(metadata))])
    result = write(value, stage)
    assert result.success and result.n_inserted == 1 and result.n_existing == 0
    assert len(value._req_session.sent) == 1
    request, kwargs = value._req_session.sent[0]
    assert kwargs['allow_redirects'] is False
    body = json.loads(request.body)
    if stage == 'entry':
        assert len(body) == 1 and body[0]['attributes']['monomer_id'] == 'monomer-0004'
    else:
        assert body['entry_names'] == [entry()['name']]
        assert body['specification_names'] == ['pbe0-atz-native-request2']
        assert body['find_existing'] is False


def test_real_unadapted_client_reproduces_expired_token_resend():
    value = client([(401, {'msg': 'Token has expired'}), (200, {'result': 'fixture'})], adapted=False)
    result = value.make_request(
        'post', 'api/v1/datasets/singlepoint/353/submit', Dict[str, Any],
        body={'fixture_only': True}, allow_retries=False)
    assert result == {'result': 'fixture'}
    assert len(value._req_session.sent) == 2 and len(value.refreshes) == 1
    with pytest.raises(ValueError):
        append.require_one_shot_client(value)


def test_unrelated_read_request_keeps_normal_token_refresh_behavior():
    value = client([(401, {'msg': 'Token has expired'}), (200, {'record': 'fixture'})])
    result = value.make_request('get', 'api/v1/datasets/353', Dict[str, Any])
    assert result == {'record': 'fixture'}
    assert len(value._req_session.sent) == 2 and len(value.refreshes) == 1


@pytest.mark.parametrize('result', [(401, {'msg': 'Token has expired'}),
                                   requests.exceptions.ConnectionError('lost response')])
def test_adapter_forces_both_retry_flags_even_if_caller_omits_allow_retries(result):
    value = client([result])
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        value.make_request('post', 'api/v1/datasets/singlepoint/353/submit',
                           Dict[str, Any], body={'fixture_only': True})
    assert len(value._req_session.sent) == 1 and not value.refreshes


def test_changed_version_or_transport_sources_cannot_create_adapter(monkeypatch):
    import qcportal
    monkeypatch.setattr(qcportal, '__version__', '0.64')
    with pytest.raises(RuntimeError, match='0.63'):
        append.client_class()
    monkeypatch.setattr(qcportal, '__version__', '0.63')
    monkeypatch.setattr(append, 'TRANSPORT_SOURCES', {'client_base.py': 'not-approved'})
    with pytest.raises(RuntimeError, match='source'):
        append.client_class()
