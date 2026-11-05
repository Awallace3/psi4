"""Mocked append transaction, real pure proof/intent guards; no live API or SCI."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_append_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(monkeypatch, tmp_path, change='success'):
    module, fixtures = load('bank_append'), load('test_audited_bank')
    core, pool, cases = fixtures.setup()
    # Supply non-overlapping toy model input to the QCPortal metadata parser.
    for index, case in enumerate(cases.values()):
        case['entry']['molecule']['geometry'] = [float(index), 0., 0., float(index+3), 0., 0.,
                                                float(index+6), 0., 0.]
        runtime = load('phoenix_worker_probe').runtime_input(case)
        case['runtime_input_sha256'] = core.digest(core.encode(runtime))
    receipts, captured, tensor = fixtures.delivered(core, cases['monomer-0001'])
    context, checkpoint = core.initialize(pool, cases, receipts, core.digest(receipts['audit']),
                                          captured['delivered'], tensor)
    proofs = {core.digest(raw): raw for raw in receipts.values()}
    raw, context_sha = core.encode(checkpoint), core.digest(context)
    case = core.bind_next_case(cases['monomer-0002'], raw, core.digest(raw), context, context_sha, proofs)
    _, captured, _ = fixtures.delivered(core, case, checkpoint['snapshot'])
    scope = NS(mode='append', dataset_id=9999, case_sha256=core.digest(core.encode(case)),
               compute_tag='ccasp-native-grid400-bank-monomer-0002-502')
    module.helper('audited_bank')  # Cache the real source implementation.
    base, store = module.helper('dataset_guard'), module.helper('bank_store')
    monkeypatch.setattr(store, 'INTENTS', tmp_path/'intents')
    monkeypatch.setenv('SLURM_JOB_ID', '502')
    snapshots = [copy.deepcopy(captured[k]) for k in ('before', 'before', 'entry', 'entry', 'record')]
    if change == 'preentry_race':
        snapshots[1]['metadata']['name'] = 'raced'
    elif change == 'presubmit_race':
        snapshots[3]['metadata']['name'] = 'raced'
    elif change == 'old_record_changed':
        snapshots[2]['records']['10001']['body_sha256'] = 'f'*64
    elif change == 'old_record_changed_after_submit':
        snapshots[4]['records']['10001']['body_sha256'] = 'f'*64
    elif change == 'new_entry_comment_changed':
        snapshots[4]['entries'][case['entry']['name']]['comment'] = 'changed after insertion'
    elif change == 'new_entry_signed_zero':
        snapshots[4]['entries'][case['entry']['name']]['molecule']['geometry'][1] = -0.0
    elif change == 'record_nonwaiting':
        snapshots[4]['records']['10002']['status'] = 'running'
    elif change == 'record_wrong_tag':
        snapshots[4]['records']['10002']['task']['compute_tag'] = 'foreign'
    elif change == 'record_wrong_spec':
        snapshots[4]['records']['10002']['specification']['method'] = 'hf'
    def snapshot(client, **kwargs):
        assert kwargs == dict(dataset_id=9999, max_entries=3)
        return snapshots.pop(0)
    monkeypatch.setattr(base, 'snapshot_dataset', snapshot)
    requests, saved, bindings = [], {}, []

    class Client:
        def get_server_information(self):
            return dict(name='CDS QCFractal Server', version='0.70')

        def query_managers(self, **kwargs):
            return [NS(compute_tags=['*'])] if change == 'wildcard_manager' else []

        def make_request(self, method, endpoint, response, *, body, allow_retries):
            assert method == 'post' and allow_retries is False
            requests.append((endpoint, body))
            if endpoint.endswith('/entries/bulkCreate'):
                assert len(body) == 1 and body[0].name == case['entry']['name']
                if change == 'lost_entry':
                    raise RuntimeError('synthetic ambiguous entry reply')
                return response(existing_idx=[0]) if change == 'existing_entry' else response(inserted_idx=[0])
            assert endpoint.endswith('/submit')
            assert body.entry_names == [case['entry']['name']] and body.find_existing is False
            assert body.specification_names == [case['specification_name']]
            assert body.compute_tag == scope.compute_tag
            if change == 'lost_submit':
                raise RuntimeError('synthetic ambiguous submit reply')
            if change == 'existing_record_count':
                return response(n_inserted=0, n_existing=1)
            if change == 'incorrect_record_count':
                return response(n_inserted=2, n_existing=0)
            return response(n_inserted=1, n_existing=0)

    client = Client()
    adapter = NS(_validate=lambda candidate, binding, draft: bindings.append(
        candidate is client and binding is scope and draft is case))
    def save(name, value):
        assert name not in saved
        failures = dict(save_intent='dispatch-intent.json', save_after_entry='owned-after-entry.json',
                        save_after_record='owned-after-record.json')
        if name == failures.get(change):
            raise OSError('synthetic metadata save failure after reservation')
        saved[name] = copy.deepcopy(value)
    kwargs = dict(dataset_adapter=adapter, prepared_case=cases['monomer-0002'],
        checkpoint_raw=raw, checkpoint_sha256=core.digest(raw), context_raw=context,
        context_sha256=context_sha, proof_blobs=proofs, authenticated_owner='awallace43', save=save)
    return module, client, scope, case, kwargs, requests, saved, bindings


def test_exact_one_entry_one_fresh_record_and_preserved_predecessor(monkeypatch, tmp_path):
    module, client, scope, case, kwargs, requests, saved, bindings = fixture(monkeypatch, tmp_path)
    result = module.append_once(client, scope, case, **kwargs)
    assert result['dataset_id'] == 9999 and result['record_id'] == 10002
    assert result['dispatch_intent_sha256'] == saved['dispatch-intent.json']['sha256']
    assert len(requests) == 2 and bindings == [True]
    assert saved['owned-after-record.json']['records']['10001'] == saved[
        'owned-before-entry.json']['records']['10001']
    with pytest.raises(FileExistsError):
        # Identical canonical state cannot acquire a second writer.
        module.helper('bank_store').reserve_intent(
            dataset_id=9999, checkpoint_sha256=kwargs['checkpoint_sha256'],
            snapshot_sha256=module.helper('dataset_guard').digest(saved['owned-before-entry.json']),
            context_sha256=kwargs['context_sha256'], case_sha256=scope.case_sha256,
            eligibility_sha256=case['eligibility_sha256'], monomer_id=case['monomer_id'],
            compute_tag=scope.compute_tag)


@pytest.mark.parametrize('change,count', [('preentry_race', 0), ('presubmit_race', 1),
    ('old_record_changed', 1), ('lost_entry', 1), ('lost_submit', 2), ('existing_entry', 1),
    ('wildcard_manager', 0)])
def test_race_refusal_or_ambiguous_reply_stops_without_replay(monkeypatch, tmp_path, change, count):
    module, client, scope, case, kwargs, requests, saved, _ = fixture(monkeypatch, tmp_path, change)
    with pytest.raises((ValueError, RuntimeError)):
        module.append_once(client, scope, case, **kwargs)
    assert len(requests) == count
    assert 'owned-after-record.json' not in saved
    if change != 'wildcard_manager':
        assert 'dispatch-intent.json' in saved


@pytest.mark.parametrize('change', ['case', 'owner', 'checkpoint', 'context', 'scope'])
def test_unbound_or_historical_append_never_contacts_mutation_endpoint(monkeypatch, tmp_path, change):
    module, client, scope, case, kwargs, requests, _, _ = fixture(monkeypatch, tmp_path)
    if change == 'case':
        case['entry']['attributes']['reference_energy'] = 'changed'
    elif change == 'owner':
        kwargs['authenticated_owner'] = 'foreign'
    elif change == 'checkpoint':
        kwargs['checkpoint_raw'] += b' '
    elif change == 'context':
        kwargs['context_sha256'] = 'f'*64
    else:
        scope.dataset_id = 356
    with pytest.raises(ValueError):
        module.append_once(client, scope, case, **kwargs)
    assert not requests


@pytest.mark.parametrize('change,count', [('old_record_changed_after_submit', 2),
    ('new_entry_comment_changed', 2), ('new_entry_signed_zero', 2),
    ('record_nonwaiting', 2), ('record_wrong_tag', 2),
    ('record_wrong_spec', 2), ('existing_record_count', 2), ('incorrect_record_count', 2),
    ('save_intent', 0), ('save_after_entry', 1), ('save_after_record', 2)])
def test_post_reservation_failures_keep_intent_and_refuse_a_fresh_attempt(
        monkeypatch, tmp_path, change, count):
    module, client, scope, case, kwargs, requests, _, _ = fixture(monkeypatch, tmp_path, change)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        module.append_once(client, scope, case, **kwargs)
    assert len(requests) == count
    root = module.helper('bank_store').INTENTS
    intent = root/'9999'/module.helper('dataset_guard').digest(
        json.loads(kwargs['checkpoint_raw'])['snapshot'])/'intent.json'
    original = intent.read_bytes()
    # Even a new process with the same apparently restored before-state may
    # not repeat an unresolved mutation. No automatic intent clearing.
    other, client2, scope2, case2, kwargs2, requests2, _, _ = fixture(monkeypatch, tmp_path)
    with pytest.raises(FileExistsError):
        other.append_once(client2, scope2, case2, **kwargs2)
    assert not requests2 and intent.read_bytes() == original


def test_failed_ancestor_sync_prevents_any_entry_mutation(monkeypatch, tmp_path):
    module, client, scope, case, kwargs, requests, _, _ = fixture(monkeypatch, tmp_path)
    def fail(path):
        raise OSError('synthetic ancestor fsync failure')
    monkeypatch.setattr(module.helper('bank_store'), '_sync', fail)
    with pytest.raises(OSError):
        module.append_once(client, scope, case, **kwargs)
    assert not requests
