"""Real0.63 clients with mocked HTTP only; no live QCA or quantum work."""
import copy
from pathlib import Path
from typing import Any, Dict

import pytest
import requests
from qcportal.client_base import PortalRequestError

import importlib.util


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load('_grid400_http_fixtures', 'transport_cb63305_tests.py')
pilot_fixture = load('_grid400_transport_pilot_fixtures', 'test_pilot_grid400.py')
fresh = fixtures.fresh
TAG = 'ccasp-native-grid400-monomer-0008-123'
ERRORS = [(401, {'msg': 'Token has expired'}),
          requests.exceptions.ConnectionError('synthetic lost response')]


def candidate_draft(monkeypatch):
    inputs, row, measurement, entry = pilot_fixture.fixture(monkeypatch)
    # Give qcelemental a nonoverlapping toy geometry; not a scientific fixture.
    geometry = [value for i in range(12) for value in (3.*i, 0., 0.)]
    row['runtime_input']['molecule']['geometry'] = geometry
    entry['molecule']['geometry'] = geometry
    import json
    inputs['entries'] = json.dumps(dict(entries=[entry])).encode()
    pilot = pilot_fixture.pilot
    sha = pilot.helper('prepare_descriptor_bank_cb63305').json_digest(row['runtime_input'])
    row['runtime_input_sha256'] = measurement['runtime_input_sha256'] = sha
    pins = pilot.producer_pins('grid400') | dict(runtime_input_sha256=sha)
    original_pins = pilot.producer_pins
    monkeypatch.setattr(pilot, 'producer_pins', lambda producer='cb63305':
        copy.deepcopy(pins) if producer == 'grid400' else original_pins(producer))
    draft = pilot_fixture.construct(inputs)
    original_helper = fresh.helper
    monkeypatch.setattr(fresh, 'helper', lambda name:
                        pilot if name == 'pilot_cb63305' else original_helper(name))
    return draft


def transaction(monkeypatch, case=None):
    data = candidate_draft(monkeypatch)
    monkeypatch.setattr(fixtures, 'draft', lambda **kwargs: copy.deepcopy(data))
    result = fixtures.transaction(case, producer='grid400')
    client = result[0]
    original_create = client.add_dataset
    def create(*args, **kwargs):
        # Stubbed add_dataset models the creation request. Actual HTTP-side
        # one-attempt/bound-ID behavior has separate real-stack tests below.
        client._grid400_create_attempted = True
        return original_create(*args, **kwargs)
    client.add_dataset = create
    return result


@pytest.mark.parametrize('endpoint', fixtures.ENDPOINTS)
@pytest.mark.parametrize('result', ERRORS)
def test_grid400_mutations_have_no_network_or_token_replay(endpoint, result):
    client = fixtures.client([result], producer='grid400')
    if '/9999/' in endpoint:
        client._grid400_create_attempted = True
        fresh._bind_grid400_dataset(client, 9999)
    with pytest.raises((PortalRequestError, requests.exceptions.ConnectionError)):
        client.make_request('post', endpoint, Dict[str, Any], body={'synthetic': True})
    assert len(client._req_session.sent) == 1 and not client.refreshes


@pytest.mark.parametrize('endpoint', [
    'api/v1/datasets/singlepoint/353/submit', 'api/v1/datasets/354',
    'api/v1/datasets/singlepoint/355/entries/bulkCreate', 'api/v1/datasets/355?query=1',
    'api/v1/records/bulkModify', 'api/v1/records/bulkDelete',
    'api/v1/records/singlepoint/5234326', 'api/v1/records/5234318/comments',
    '/api/v1/datasets/355', 'api/v1/datasets/%33%35%35',
    'api/v1/datasets/singlepoint/9999/../355/submit', 'api//v1/datasets/355'])
def test_all_old_datasets_and_record_mutations_refuse_before_http(endpoint):
    client = fixtures.client([], producer='grid400')
    with pytest.raises(ValueError):
        client.make_request('post', endpoint, Dict[str, Any], body={'ids': [5234326]})
    assert not client._req_session.sent


@pytest.mark.parametrize('endpoint', [
    'api/v1/datasets/singlepoint/0355/submit', 'api/v1/datasets/000354',
    'api/v1/datasets/bulkDelete', 'api/v1/datasets/clone',
    'api/v1/datasets/singlepoint/9998/entries/bulkCreate',
    'api/v1/datasets/singlepoint/09999/submit'])
def test_alias_bulk_clone_or_foreign_dataset_writes_refuse_before_http(endpoint):
    client = fixtures.client([], producer='grid400')
    client._grid400_create_attempted = True
    fresh._bind_grid400_dataset(client, 9999)
    with pytest.raises(ValueError):
        client.make_request('post', endpoint, Dict[str, Any], body={'ids': [355]})
    assert not client._req_session.sent


def test_lost_create_response_cannot_be_replayed_or_bound_to_an_old_dataset():
    client = fixtures.client([requests.exceptions.ConnectionError('synthetic ambiguous create')],
                             producer='grid400')
    endpoint = 'api/v1/datasets/singlepoint'
    with pytest.raises(requests.exceptions.ConnectionError):
        client.make_request('post', endpoint, Dict[str, Any], body={})
    with pytest.raises(ValueError, match='one fresh grid400 dataset creation attempt'):
        client.make_request('post', endpoint, Dict[str, Any], body={})
    assert len(client._req_session.sent) == 1
    with pytest.raises(ValueError):
        fresh._bind_grid400_dataset(client, 355)


def test_grid400_client_is_not_interchangeable_with_historical_clients():
    grid = fixtures.client([], producer='grid400')
    for producer in ('cb63305', 'ed561666'):
        old = fixtures.client([], producer=producer)
        assert type(old) is not type(grid)
        with pytest.raises(ValueError):
            fresh.require_client(old, producer='grid400')
        with pytest.raises(ValueError):
            fresh.require_client(grid, producer=producer)
    assert fresh.transport_binding(producer='grid400')['manager_activation_policy'] == (
        'owned-one-shot-client-factory-including-activation/1')


def test_one_fresh_transaction_preserves_metadata_and_never_reuses_a_record(monkeypatch):
    client, expected, data, calls, saved, save = transaction(monkeypatch)
    dataset, _ = fresh.register_once(client, data, TAG, save, producer='grid400')
    assert dataset is expected and calls[0][2]['existing_ok'] is False
    assert calls[1][1][0].attributes == data['entry']['attributes']
    fresh.submit_once(client, dataset, data, TAG, save, producer='grid400')
    assert calls[-1][1] == dict(entry_names=[data['entry']['name']],
        specification_names=[data['specification_name']], compute_tag=TAG, find_existing=False)
    assert data['execution_profile']['time_limit'] == '04:00:00'
    assert all(item[1].get('dataset_id') not in (353, 354, 355) for item in saved)


@pytest.mark.parametrize('case', ['old_dataset', 'protected354', 'wrong_name', 'different_client',
    'changed_attributes', 'existing_record', 'existing_entry', 'lost_create_response'])
def test_failed_or_ambiguous_fresh_registration_never_submits_or_retries(monkeypatch, case):
    client, dataset, data, calls, saved, save = transaction(monkeypatch, case)
    with pytest.raises((ValueError, RuntimeError, requests.exceptions.ConnectionError)):
        fresh.register_once(client, data, TAG, save, producer='grid400')
    assert sum(call[0] == 'create' for call in calls) == 1
    assert not any(call[0] == 'submit' for call in calls)


def test_owned355_cannot_become_the_new_dataset(monkeypatch):
    client, dataset, data, calls, saved, save = transaction(monkeypatch)
    dataset.id = 355
    with pytest.raises(RuntimeError, match='historical identity'):
        fresh.register_once(client, data, TAG, save, producer='grid400')
    assert [call[0] for call in calls] == ['create']


@pytest.mark.parametrize('change', ['append', 'shape', 'runtime', 'tag'])
def test_changed_draft_is_rejected_before_dataset_creation(monkeypatch, change):
    client, dataset, data, calls, saved, save = transaction(monkeypatch)
    tag = TAG
    if change == 'append':
        data['dataset_id'] = 355
    elif change == 'shape':
        data['queue_policy']['cpus'] = 12
    elif change == 'runtime':
        data['entry']['additional_keywords']['function_kwargs']['native_monomer']['resources']['max_bytes'] *= 2
    else:
        tag = 'ccasp-native-ed561666-monomer-0008-123'
    with pytest.raises(ValueError):
        fresh.register_once(client, data, tag, save, producer='grid400')
    assert not calls


@pytest.mark.parametrize('operation', ['activate', 'claim', 'return', 'deactivate'])
@pytest.mark.parametrize('result', ERRORS)
def test_owned_manager_including_activation_has_no_http_replay(operation, result):
    from qcportal.manager_client import ManagerClient
    from qcportal.managers.models import ManagerName
    import qcfractalcompute.compute_manager as compute
    original = compute.ManagerClient
    assert original is ManagerClient
    with fresh.grid400_manager_client_factory() as selected:
        assert compute.ManagerClient is selected
        client = object.__new__(selected)
        client.__dict__.update(fixtures.legacy_fixtures.client([result], adapted=False).__dict__)
        client.manager_name_data = ManagerName(cluster='synthetic', hostname='unit-only', uuid='123')
        client.username = 'synthetic-unit-user'
        client.server_info = {'api_limits': {'manager_tasks_return': 1}}
        expected_error = (RuntimeError if operation == 'return' else
                          (PortalRequestError, requests.exceptions.ConnectionError))
        with pytest.raises(expected_error):
            if operation == 'activate':
                client.activate('0.63', {'psi4': ['synthetic']}, [TAG])
            elif operation == 'claim':
                client.claim(programs={'psi4': ['synthetic']}, tags=[TAG], limit=1)
            elif operation == 'return':
                client.return_finished({9999: b'SYNTHETIC_NOT_A_RESULT'})
            else:
                client.deactivate(active_tasks=0, active_cores=0, active_memory=0., total_cpu_hours=0.)
        assert len(client._req_session.sent) == 1 and not client.refreshes
    assert compute.ManagerClient is original


def test_manager_factory_restores_original_binding_on_failure_and_rejects_nesting():
    import qcfractalcompute.compute_manager as compute
    original = compute.ManagerClient
    with pytest.raises(RuntimeError, match='synthetic construction failure'):
        with fresh.grid400_manager_client_factory():
            with pytest.raises(ValueError, match='unmodified manager client constructor'):
                with fresh.grid400_manager_client_factory():
                    raise AssertionError('nested replacement must not enter')
            raise RuntimeError('synthetic construction failure')
    assert compute.ManagerClient is original


@pytest.mark.parametrize('method,suffix', [
    ('post', 'background_submit'), ('post', 'copy_from'),
    ('patch', 'records'), ('post', 'records/bulkDelete'),
    ('post', 'records/revert'), ('delete', 'entries'),
    ('patch', 'specifications'), ('put', 'submit'),
    ('post', 'submit/'), ('post', 'entries/rename'),
])
def test_bound_dataset_cannot_reset_copy_cancel_retag_or_background_submit(method, suffix):
    client = fixtures.client([], producer='grid400')
    client._grid400_create_attempted = True
    fresh._bind_grid400_dataset(client, 9999)
    with pytest.raises(ValueError):
        client.make_request(method, 'api/v1/datasets/singlepoint/9999/'+suffix,
                            Dict[str, Any], body={'ids': [5234326]})
    assert not client._req_session.sent


def test_deferred_and_shutdown_updates_cannot_replay_ambiguous_delivery():
    """Exercise the actual compute-manager deferred path, not one HTTP call."""
    from collections import defaultdict
    import logging
    from types import SimpleNamespace
    from qcportal.managers.models import ManagerName
    from qcfractalcompute.compute_manager import ComputeManager
    with fresh.grid400_manager_client_factory() as selected:
        client = object.__new__(selected)
        client.__dict__.update(fixtures.legacy_fixtures.client([
            requests.exceptions.ConnectionError('synthetic ambiguous result return')
        ], adapted=False).__dict__)
        client.manager_name_data = ManagerName(cluster='synthetic', hostname='unit-only', uuid='123')
        client.username = 'synthetic-unit-user'
        client.server_info = {'api_limits': {'manager_tasks_return': 1}}
        manager = object.__new__(ComputeManager)
        manager.client = client
        manager.logger = logging.getLogger('synthetic-grid400-no-delivery-replay')
        manager._deferred_tasks = defaultdict(dict, {
            0: {9999: SimpleNamespace(result_compressed=b'SYNTHETIC_NOT_A_RESULT')}})
        # Both normal and shutdown updates use this same real deferred path.
        for _ in range(3):
            with pytest.raises(RuntimeError, match='grid400 result delivery'):
                manager._update_deferred_tasks()
        assert len(client._req_session.sent) == 1 and not client.refreshes


def test_real_dataset_methods_complete_one_fresh_transaction(monkeypatch):
    """No dataset methods stubbed: use real0.63 serialization and fetch paths."""
    from collections import defaultdict
    from dataclasses import asdict
    import json
    from urllib.parse import urlsplit
    from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata
    from qcportal.cache import PortalCache
    data = candidate_draft(monkeypatch)
    client = fixtures.client([], producer='grid400')
    client.cache = PortalCache(client.address, None, 16*1024**2)
    client.server_info = {'api_limits': defaultdict(lambda: 100)}
    client.api_limits = client.server_info['api_limits']
    observed = []
    entry = copy.deepcopy(data['entry'])
    entry['molecule']['id'] = 9997
    metadata = dict(id=9999, dataset_type='singlepoint', name=data['dataset_name'],
        description='Synthetic HTTP-only fixture', tagline='', tags=[], group='default',
        visibility=True, provenance={}, extras={}, default_tag=TAG, default_priority=1,
        owner_user='synthetic-unit-user', owner_group=None)

    def response(request, **kwargs):
        route = urlsplit(request.url).path.lstrip('/')
        observed.append((request.method, route))
        client._req_session.sent.append((request, kwargs))
        if request.method == 'POST' and route == 'api/v1/datasets/singlepoint':
            body = 9999
        elif request.method == 'GET' and route == 'api/v1/datasets/9999':
            body = metadata
        elif route.endswith('/entries/bulkCreate'):
            body = asdict(InsertMetadata(inserted_idx=[0]))
        elif route.endswith('/specifications') and request.method == 'POST':
            body = asdict(InsertMetadata(inserted_idx=[0]))
        elif route.endswith('/entries/bulkFetch'):
            body = {entry['name']: entry}
        elif route.endswith('/entry_names'):
            body = [entry['name']]
        elif route.endswith('/specifications/bulkFetch'):
            body = {data['specification_name']: dict(name=data['specification_name'],
                specification=data['specification'], description=None)}
        elif route.endswith('/records/bulkFetch'):
            body = []
        elif route.endswith('/submit'):
            body = asdict(InsertCountsMetadata(n_inserted=1, n_existing=0))
        else:
            raise AssertionError('unexpected real dataset route: '+str((request.method, route)))
        result = requests.Response()
        result.status_code = 200
        result.headers['Content-Type'] = 'application/json'
        result._content = json.dumps(body).encode()
        result.request = request
        return result
    client._req_session.send = response
    saved = []
    save = lambda name, document: saved.append((name, document))
    dataset, _ = fresh.register_once(client, data, TAG, save, producer='grid400')
    assert dataset.id == 9999 and client._grid400_dataset_id == 9999
    result = fresh.submit_once(client, dataset, data, TAG, save, producer='grid400')
    assert result.success and result.n_inserted == 1 and result.n_existing == 0
    assert not client.refreshes
    writes = [route for method, route in observed if method != 'GET']
    for suffix in ('singlepoint', '9999/entries/bulkCreate', '9999/specifications', '9999/submit'):
        assert sum(route.endswith('/'+suffix) for route in writes) == 1


def test_successful_acknowledgment_through_real_single_claim_updates():
    from collections import defaultdict
    from dataclasses import asdict
    import logging
    from types import SimpleNamespace as NS
    from qcportal.managers.models import ManagerName
    from qcportal.metadata_models import TaskReturnMetadata
    from qcfractalcompute.compute_manager import ComputeManager, ManagerStatistics
    mixin = load('_grid400_successful_ack_mixin', 'single_claim.py').SingleClaimMixin

    class LocalFixtureManager(mixin, ComputeManager):
        # No executor or quantum work. Only result acquisition and stopping
        # are fixtures; _update/_return_finished and HTTP are real methods.
        @property
        def n_active_tasks(self):
            return {'pilot': 0}

        @property
        def n_total_active_tasks(self):
            return 0

        def _acquire_complete_tasks(self):
            return self.synthetic_results.pop(0) if self.synthetic_results else {}

        def stop(self):
            self.stopped = True

    with fresh.grid400_manager_client_factory() as selected:
        client = object.__new__(selected)
        client.__dict__.update(fixtures.legacy_fixtures.client([
            (200, None), (200, asdict(TaskReturnMetadata(accepted_ids=[9999]))), (200, None)
        ], adapted=False).__dict__)
        client.manager_name_data = ManagerName(cluster='synthetic', hostname='unit-only', uuid='123')
        client.username = 'synthetic-unit-user'
        client.server_info = {'api_limits': {'manager_tasks_return': 1}}
        client.activate('0.63', {'psi4': ['synthetic']}, [TAG])
        manager = object.__new__(LocalFixtureManager)
        manager.client = client
        manager.logger = logging.getLogger('synthetic-grid400-successful-ack')
        manager._deferred_tasks = defaultdict(dict)
        manager._record_id_map = {9999: 9998}
        manager.statistics = ManagerStatistics()
        manager.manager_config = NS(executors={'pilot': NS(cores_per_worker=16, memory_per_worker=60)})
        manager.target_task_id = 9999
        manager.delivery_outcome = None
        manager.synthetic_results = [{'pilot': {9999: NS(
            result_compressed=b'SYNTHETIC_NOT_A_RESULT', success=True, walltime=1.)}}]
        manager.stopped = False
        for _ in range(3):
            manager._update(new_tasks=False)
        assert manager.delivery_outcome == 'accepted' and manager.stopped
        assert manager.statistics.total_successful_tasks == 1
        assert not manager._deferred_tasks
        client.deactivate(active_tasks=0, active_cores=0, active_memory=0., total_cpu_hours=0.)
        assert len(client._req_session.sent) == 3 and not client.refreshes
