"""Actual0.63 request-stack tests with mockedHTTP only; no credentials/server."""
import importlib.util
from pathlib import Path
from typing import Any, Dict

import pytest
import requests
from qcportal.client_base import PortalRequestError


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('_forensic_transport_module', 'forensic_grid400.py')
fixtures = load('_forensic_transport_fixtures', 'transport_tests.py')


def client(results):
    value = object.__new__(audit.readonly_client_class())
    value.__dict__.update(fixtures.client(results, adapted=False).__dict__)
    return value


@pytest.mark.parametrize('method,endpoint', [
    ('post', 'api/v1/datasets/singlepoint'), ('post', 'api/v1/datasets/singlepoint/356/submit'),
    ('patch', 'api/v1/datasets/356/records'), ('post', 'api/v1/records/bulkModify'),
    ('delete', 'api/v1/records/5234327'), ('post', 'compute/v1/tasks/return'),
    ('post', 'compute/v1/tasks/claim'), ('post', 'compute/v1/managers')])
def test_forensic_client_refuses_every_mutation_before_http(method, endpoint):
    value = client([])
    with pytest.raises(ValueError, match='forbids server mutations'):
        value.make_request(method, endpoint, Dict[str, Any], body={})
    assert not value._req_session.sent


@pytest.mark.parametrize('endpoint', ['auth/v1/login', 'auth/v1/refresh',
    'api/v1/records/bulkGet', 'api/v1/records/singlepoint/bulkGet'])
@pytest.mark.parametrize('error', [
    (401, {'msg': 'Token has expired'}), requests.exceptions.ConnectionError('synthetic lost read')])
def test_allowed_read_auth_posts_never_replay(endpoint, error):
    value = client([error])
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        value.make_request('post', endpoint, Dict[str, Any], body={})
    assert len(value._req_session.sent) == 1 and not value.refreshes


def test_read_only_client_preserves_real_serialization():
    value = client([(200, {'synthetic_record': 5234327})])
    result = value.make_request('post', 'api/v1/records/bulkGet', Dict[str, Any], body=[5234327])
    assert result == {'synthetic_record': 5234327}
    assert len(value._req_session.sent) == 1


def test_real_get_records_include_all_preserves_history_and_envelope():
    import json
    from collections import defaultdict
    from qcportal.cache import PortalCache
    from qcportal.singlepoint import SinglepointRecord
    record = SinglepointRecord(id=5234327, is_service=False, status='complete',
        created_on='2026-11-04T12:00:00+00:00', modified_on='2026-11-04T14:00:00+00:00',
        molecule_id=9997, specification=dict(program='psi4', driver='properties',
            method='pbe0', basis='aug-cc-pvtz', keywords={}),
        extras={'psi4:native_monomer': {'synthetic': 'HTTP-only'}},
        compute_history=[dict(id=9998, record_id=5234327, status='complete',
            modified_on='2026-11-04T14:00:00+00:00', manager_name='synthetic-owned-manager')])
    wire = json.loads(record.json(by_alias=True, exclude_none=True))
    value = client([(200, [wire]), (200, [wire])])
    value.cache = PortalCache(value.address, None, 16*1024**2)
    value.api_limits = defaultdict(lambda: 100)
    value.download_target_time, value.n_download_threads = 1., 1
    summary, envelope = audit.server_snapshot(value)
    assert summary == dict(record_id=5234327, status='complete', history_count=1,
                           history_managers=['synthetic-owned-manager'])
    assert envelope == {'synthetic': 'HTTP-only'}
    assert value._req_session.sent and not value.refreshes
    assert all(request.method == 'POST' and request.url.split('.invalid/')[-1].split('?')[0] in (
        'api/v1/records/bulkGet', 'api/v1/records/singlepoint/bulkGet')
        for request, _ in value._req_session.sent)
