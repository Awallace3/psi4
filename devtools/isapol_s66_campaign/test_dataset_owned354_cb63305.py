"""Synthetic singleton append orchestration, never real records or server calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_owned_append_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


append = load('dataset_owned354_cb63305')
fixtures = load('test_owned_dataset_cb63305')
TAG = 'ccasp-native-cb63305-monomer-0047-12345'


def setup(monkeypatch, tmp_path, *, change=None, producer='cb63305'):
    guard = fixtures.owned
    selected = {} if producer == 'cb63305' else dict(producer=producer)
    before, accepted, specification = (fixtures.fixture() if not selected else fixtures.fixture_ed())
    case = fixtures.new_case()
    tag, first_record = TAG, guard.FIRST_RECORD
    if selected:
        context = guard.context(producer)
        case.update(dataset_id=355, dataset_name=context['dataset_name'], monomer_id='monomer-0006',
            producer_profile=producer, dataset_mode='append',
            required_first_pilot_audit_sha256=context['accepted_sha256']['audit'])
        case['entry']['name'] = 'monomer-0006-unit-only'
        case['entry']['attributes']['monomer_id'] = 'monomer-0006'
        tag, first_record = 'ccasp-native-ed561666-monomer-0006-12345', context['first_record']
        monkeypatch.setattr(append, 'INTENT_ROOT', tmp_path/'dispatch-intents')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
    case['specification'] = copy.deepcopy(specification)
    del case['specification']['protocols']
    case['entry']['molecule'].update(molecular_charge=0, molecular_multiplicity=1,
                                    fix_com=True, fix_orientation=True)
    path = tmp_path/'case.json'
    supplied = copy.deepcopy(case)
    if change == 'tampered_case':
        supplied['entry']['molecule']['geometry'][0] = .25
    path.write_text(json.dumps(supplied))
    checkpoint = guard.prepare_initial_checkpoint(before, authenticated_owner='awallace43',
        specification=specification, accepted=accepted, **selected)
    raw = json.dumps(checkpoint).encode()
    if selected:
        # Explicit fake anchors only: these snapshots are not molecular evidence.
        monkeypatch.setattr(guard, 'SEED_CHECKPOINT_SHA256', hashlib.sha256(raw).hexdigest())
        monkeypatch.setattr(guard, 'SEED_SNAPSHOT_SHA256', guard.base().digest(before))
    after_entry = copy.deepcopy(before)
    entry = case['entry']['name']
    after_entry['entries'][entry] = copy.deepcopy(case['entry'])
    after_entry['associations'][entry] = {guard.SPECIFICATION: None}
    after_record = copy.deepcopy(after_entry)
    effective = copy.deepcopy(specification)
    effective['keywords'].update(case['entry']['additional_keywords'])
    after_record['associations'][entry][guard.SPECIFICATION] = 9999
    after_record['records']['9999'] = dict(id=9999, molecule_id=8888, status='waiting',
        history=[], task={'compute_tag': tag}, specification=effective)
    if change == 'old_record_changed':
        after_record['records'][str(first_record)]['status'] = 'error'
    elif change == 'not_fresh':
        after_record['records']['9999']['history'] = [{'manager_name': 'foreign'}]
    elif change == 'wrong_tag':
        after_record['records']['9999']['task']['compute_tag'] = '*'
    elif change == 'stored_entry_changed':
        after_entry['entries'][entry]['molecule']['geometry'][0] = .1
    frames = [before, before, after_entry, after_entry, after_record]
    if change == 'concurrent':
        frames[1] = dict(before, metadata=dict(before['metadata'], name='changed'))
    iterator = iter(copy.deepcopy(frames))
    monkeypatch.setattr(guard, 'snapshot_dataset', lambda client, **kw: next(iterator))
    monkeypatch.setattr(guard, 'verify_first_acceptance', lambda blobs, **kw: accepted)
    transport = NS(require_client=lambda client, **kw: None)
    prep = NS(verify_prepared_from_paths=lambda *args, **kw: copy.deepcopy(case))
    original = append.helper
    monkeypatch.setattr(append, 'helper', lambda name:
        {'owned_dataset_cb63305': guard, 'dataset_fresh_cb63305': transport,
         'case_cb63305': prep}.get(name) or original(name))
    class Model(NS):
        def dict(self, **kwargs):
            return vars(self)
    for name in ('qcportal', 'qcportal.singlepoint', 'qcportal.dataset_models', 'qcportal.metadata_models'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['qcportal.singlepoint'].QCSpecification = Model
    sys.modules['qcportal.singlepoint'].SinglepointDatasetNewEntry = Model
    sys.modules['qcportal.dataset_models'].DatasetSubmitBody = Model
    sys.modules['qcportal.metadata_models'].InsertMetadata = Model
    sys.modules['qcportal.metadata_models'].InsertCountsMetadata = Model
    class Client:
        def __init__(self):
            self.calls = []
        def get_server_information(self):
            return dict(name='CDS QCFractal Server', version='0.70')
        def query_managers(self, **kwargs):
            assert kwargs == dict(status=['active'], limit=200)
            if change == 'wildcard_manager':
                return [NS(compute_tags=['*'])]
            if change == 'matching_manager':
                return [NS(compute_tags=[tag])]
            if change == 'manager_limit':
                return [NS(compute_tags=['unrelated']) for _ in range(200)]
            return []
        def make_request(self, method, endpoint, response, **kwargs):
            self.calls.append((method, endpoint, kwargs))
            if change == 'lost_reply' or change == 'submit_lost_reply' and len(self.calls) == 2:
                raise TimeoutError('synthetic lost reply; no retry')
            existing = change == 'existing' or change == 'submit_existing' and len(self.calls) == 2
            return NS(success=True, n_inserted=1, n_existing=int(existing))
    saved = {}
    def save(name, value):
        if name in saved:
            raise FileExistsError(name)
        saved[name] = copy.deepcopy(value)
    kwargs = dict(case_path=path, inputs=None, build_identity=None, qualification=None,
        descriptors=None, first_receipt_blobs={'synthetic': b'not receipts'},
        checkpoint_raw=raw, checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        authenticated_owner='awallace43', compute_tag=tag, save=save, **selected)
    return Client(), kwargs, saved, guard


def test_only354_entry_and_request3_singleton_submit_preserve_audited_firstcase(monkeypatch, tmp_path):
    client, kwargs, saved, guard = setup(monkeypatch, tmp_path)
    result = append.append_once(client, **kwargs)
    assert result['dataset_id'] == 354 and result['record_id'] == 9999
    assert result['molecule_id'] == 8888 and result['n_existing'] == 0
    assert [(method, endpoint) for method, endpoint, _ in client.calls] == [
        ('post', 'api/v1/datasets/singlepoint/354/entries/bulkCreate'),
        ('post', 'api/v1/datasets/singlepoint/354/submit')]
    assert all(kw['allow_retries'] is False for _, _, kw in client.calls)
    body = client.calls[1][2]['body']
    assert body.find_existing is False and body.compute_tag == TAG
    assert body.specification_names == ['pbe0-atz-native-request3']
    assert saved['owned-before-entry.json']['records'][str(guard.FIRST_RECORD)] == (
        saved['owned-after-record.json']['records'][str(guard.FIRST_RECORD)])


@pytest.mark.parametrize('change,posts', [
    ('tampered_case', 0), ('wildcard_manager', 0), ('matching_manager', 0),
    ('manager_limit', 0), ('concurrent', 0),
    ('lost_reply', 1), ('existing', 1), ('stored_entry_changed', 1),
    ('submit_lost_reply', 2), ('submit_existing', 2),
    ('old_record_changed', 2), ('not_fresh', 2), ('wrong_tag', 2)])
def test_ambiguous_or_changed_states_stop_without_retry_or_other_mutation(
        monkeypatch, tmp_path, change, posts):
    client, kwargs, _, _ = setup(monkeypatch, tmp_path, change=change)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        append.append_once(client, **kwargs)
    assert len(client.calls) == posts
    assert all('/354/' in endpoint for _, endpoint, _ in client.calls)


@pytest.mark.parametrize('change', ['bytes', 'checkpoint_status', 'checkpoint_extra_record',
                                   'tag', 'owner'])
def test_checkpoint_and_owner_are_not_inferred_from_case_labels(monkeypatch, tmp_path, change):
    client, kwargs, _, _ = setup(monkeypatch, tmp_path)
    if change == 'bytes':
        kwargs['checkpoint_raw'] += b' '
    elif change in ('checkpoint_status', 'checkpoint_extra_record'):
        checkpoint = json.loads(kwargs['checkpoint_raw'])
        if change == 'checkpoint_status':
            checkpoint['status'] = 'PREPARED'
        else:
            checkpoint['snapshot']['records']['7777'] = {}
        kwargs['checkpoint_raw'] = json.dumps(checkpoint).encode()
        kwargs['checkpoint_sha256'] = hashlib.sha256(kwargs['checkpoint_raw']).hexdigest()
    elif change == 'tag':
        kwargs['compute_tag'] = '*'
    else:
        kwargs['authenticated_owner'] = 'foreign'
    with pytest.raises(ValueError):
        append.append_once(client, **kwargs)
    assert client.calls == []


def test_client_and_transport_aliases_keep_the_same_reviewed_client_class(monkeypatch):
    class SyntheticClient:
        pass
    shared = NS(client_class=lambda: SyntheticClient, transport_binding=lambda: {'synthetic': 'transport'})
    def helper(name):
        assert name == 'dataset_fresh_cb63305'
        return shared
    monkeypatch.setattr(append, 'helper', helper)
    assert append.client_class() is SyntheticClient
    assert append.transport_binding() == {'synthetic': 'transport'}


def test_cap32_first_append_only355_preserves_audited_benzene_and_creates_fresh_methylamine(
        monkeypatch, tmp_path):
    client, kwargs, saved, guard = setup(monkeypatch, tmp_path, producer='ed561666')
    result = append.append_once(client, **kwargs)
    assert result['dataset_id'] == 355 and result['record_id'] == 9999
    assert [endpoint for _, endpoint, _ in client.calls] == [
        'api/v1/datasets/singlepoint/355/entries/bulkCreate',
        'api/v1/datasets/singlepoint/355/submit']
    assert client.calls[1][2]['body'].find_existing is False
    assert saved['owned-before-entry.json']['records']['5234318'] == (
        saved['owned-after-record.json']['records']['5234318'])
    assert guard.DATASET_ID == 354


@pytest.mark.parametrize('change,posts', [
    ('tampered_case', 0), ('wildcard_manager', 0), ('matching_manager', 0),
    ('manager_limit', 0), ('concurrent', 0),
    ('lost_reply', 1), ('existing', 1), ('stored_entry_changed', 1),
    ('submit_lost_reply', 2), ('submit_existing', 2),
    ('old_record_changed', 2), ('not_fresh', 2), ('wrong_tag', 2)])
def test_cap32_ambiguous_append_never_repeats_or_mutates_old_targets(
        monkeypatch, tmp_path, change, posts):
    client, kwargs, _, _ = setup(monkeypatch, tmp_path, producer='ed561666', change=change)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        append.append_once(client, **kwargs)
    assert len(client.calls) == posts
    assert all('/355/' in endpoint and kw['allow_retries'] is False
               for _, endpoint, kw in client.calls)


def test_cap32_initial_checkpoint_cannot_bless_an_unaudited_second_record(monkeypatch, tmp_path):
    client, kwargs, _, _ = setup(monkeypatch, tmp_path, producer='ed561666')
    checkpoint = json.loads(kwargs['checkpoint_raw'])
    checkpoint['generation'] = 2
    checkpoint['audited_records'].append(9999)
    kwargs['checkpoint_raw'] = json.dumps(checkpoint).encode()
    kwargs['checkpoint_sha256'] = hashlib.sha256(kwargs['checkpoint_raw']).hexdigest()
    with pytest.raises(ValueError, match='audited checkpoint'):
        append.append_once(client, **kwargs)
    assert not client.calls


def test_atomic_checkpoint_intent_blocks_second_writer_without_repeating_posts(monkeypatch, tmp_path):
    client, kwargs, saved, _ = setup(monkeypatch, tmp_path, producer='ed561666')
    append.append_once(client, **kwargs)
    saved_intent = saved['dispatch-intent.json']
    case = json.loads(kwargs['case_path'].read_bytes())
    assert append.verify_dispatch_intent(saved_intent,
        checkpoint_sha256=kwargs['checkpoint_sha256'],
        snapshot_sha256=json.loads(kwargs['checkpoint_raw'])['snapshot_sha256'],
        case_sha256=hashlib.sha256(kwargs['case_path'].read_bytes()).hexdigest(),
        monomer_id=case['monomer_id'], entry_name=case['entry']['name'],
        compute_tag=kwargs['compute_tag']) == saved_intent['sha256']
    with pytest.raises(FileExistsError):
        append.reserve_dispatch_intent(checkpoint_sha256=kwargs['checkpoint_sha256'],
            snapshot_sha256=json.loads(kwargs['checkpoint_raw'])['snapshot_sha256'],
            case_sha256='f'*64, monomer_id='monomer-0001', entry_name='synthetic-other-case',
            compute_tag='ccasp-native-ed561666-monomer-0001-12345')
    assert len(client.calls) == 2
    assert (append.INTENT_ROOT/json.loads(kwargs['checkpoint_raw'])['snapshot_sha256']/'intent.json').exists()
    with pytest.raises(FileExistsError):
        append.reserve_dispatch_intent(checkpoint_sha256='e'*64,
            snapshot_sha256=json.loads(kwargs['checkpoint_raw'])['snapshot_sha256'],
            case_sha256='f'*64, monomer_id='monomer-0001', entry_name='synthetic-byte-alias',
            compute_tag='ccasp-native-ed561666-monomer-0001-12345')
