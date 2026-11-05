"""Real reviewed client/scope + mocked snapshots/HTTP, never live QCA or SCI."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_append_http_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_append_uses_the_real_reviewed_adapter_scope_and_singleton_wire_bodies(monkeypatch, tmp_path):
    fixtures = load('test_bank_append')
    append, _, _, case, kwargs, _, _, _ = fixtures.fixture(monkeypatch, tmp_path)
    adapter = load('bank_dataset')
    tag = 'ccasp-native-grid400-bank-monomer-0002-502'
    scope = adapter.prepare_transaction(case, case, tag, checkpoint_raw=kwargs['checkpoint_raw'])
    cls = adapter.client_class(scope)
    client = object.__new__(cls)
    legacy = load('transport_cb63305_tests').legacy_fixtures
    client.__dict__.update(legacy.client([
        (200, asdict(InsertMetadata(inserted_idx=[0]))),
        (200, asdict(InsertCountsMetadata(n_inserted=1, n_existing=0))),
    ], adapted=False).__dict__)
    client.get_server_information = lambda: dict(name='CDS QCFractal Server', version='0.70')
    client.query_managers = lambda **kw: []
    kwargs['dataset_adapter'] = adapter
    result = append.append_once(client, scope, case, **kwargs)
    assert result['record_id'] == 10002 and result['n_inserted'] == 1 and result['n_existing'] == 0
    assert scope.attempted == {'entry', 'submit'} and len(client._req_session.sent) == 2
    entry, submission = (json.loads(request.body) for request, _ in client._req_session.sent)
    assert len(entry) == 1 and entry[0]['name'] == case['entry']['name']
    assert submission['entry_names'] == [case['entry']['name']]
    assert submission['specification_names'] == [case['specification_name']]
    assert submission['compute_tag'] == tag and submission['find_existing'] is False
