"""Real SDK read-only allowlist with fake HTTP; no server or scientific work."""
import importlib.util
from pathlib import Path
from typing import Any, Dict

import pytest
import requests
from qcportal.client_base import PortalRequestError


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_readonly_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def client(results=()):
    fixtures = load('bank_transport_tests')
    adapter, scope, parent, case = fixtures.transaction(results)
    cls = load('bank_readonly').readonly_client_class(adapter, scope, 9999)
    value = object.__new__(cls)
    value.__dict__.update(parent.__dict__)
    return value, adapter, scope, case


@pytest.mark.parametrize('endpoint', ['api/v1/datasets/singlepoint',
    'api/v1/datasets/singlepoint/9999/entries/bulkCreate',
    'api/v1/datasets/singlepoint/9999/specifications',
    'api/v1/datasets/singlepoint/9999/submit', 'api/v1/records/bulkModify',
    'api/v1/records/bulkDelete', 'api/v1/datasets/singlepoint/356/records/bulkFetch'])
def test_mutation_or_foreign_dataset_post_is_blocked_before_http(endpoint):
    value, _, _, _ = client()
    with pytest.raises(ValueError, match='read-only'):
        value.make_request('post', endpoint, Dict[str, Any], body={})
    assert not value._req_session.sent


@pytest.mark.parametrize('endpoint', ['api/v1/records/singlepoint/bulkGet',
    'api/v1/records/singlepoint/query', 'api/v1/molecules/bulkGet',
    'api/v1/datasets/singlepoint/9999/entries/bulkFetch',
    'api/v1/datasets/singlepoint/9999/records/bulkFetch',
    'api/v1/datasets/singlepoint/9999/specifications/bulkFetch'])
def test_explicit_reads_survive_without_mutation_permission(endpoint):
    value, _, _, _ = client([(200, {'unit': 'metadata only'})])
    assert value.make_request('post', endpoint, Dict[str, Any], body={}) == {'unit': 'metadata only'}
    assert len(value._req_session.sent) == 1


@pytest.mark.parametrize('error', [(401, {'msg': 'Token has expired'}),
                                  requests.exceptions.ConnectionError('synthetic ambiguous read')])
def test_read_post_cannot_replay_after_token_or_connection_failure(error):
    value, _, _, _ = client([error])
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        value.make_request('post', 'api/v1/records/singlepoint/query', Dict[str, Any], body={})
    assert len(value._req_session.sent) == 1 and not value.refreshes


def test_audit_client_cannot_be_used_as_the_transaction_owned_mutation_client():
    value, adapter, scope, case = client()
    with pytest.raises(ValueError, match='transaction-owned'):
        adapter.register_once(value, scope, case, lambda *args: None)
    assert not value._req_session.sent
