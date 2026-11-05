"""Synthetic captured/live audit publication, never QCA or quantum evidence."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_audit_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(monkeypatch, tmp_path, append=False):
    module, fixture_module = load('bank_audit'), load('test_audited_bank')
    core, pool, cases = fixture_module.setup()
    store, base = module.helper('bank_store'), module.helper('dataset_guard')
    monkeypatch.setattr(store, 'INTENTS', tmp_path/'intents')
    monkeypatch.setattr(store, 'PROOFS', tmp_path/'proofs')

    def reserve(case, receipts, views, bound):
        outcome, audit = json.loads(receipts['outcome']), json.loads(receipts['audit'])
        monkeypatch.setenv('SLURM_JOB_ID', outcome['allocation']['JobId'])
        intent = store.reserve_intent(
            dataset_id=None if bound is None else 9999,
            checkpoint_sha256=None if bound is None else bound['checkpoint_sha256'],
            snapshot_sha256=None if bound is None else base.digest(views['before']),
            context_sha256=None if bound is None else bound['context_sha256'],
            case_sha256=core.digest(core.encode(case)), eligibility_sha256=case['eligibility_sha256'],
            monomer_id=case['monomer_id'], compute_tag=outcome['compute_tag'])
        outcome['dispatch_intent_sha256'] = intent['sha256']
        receipts['outcome'] = core.encode(outcome)
        audit['case_outcome_sha256'] = core.digest(receipts['outcome'])
        if bound is not None:
            audit['captured_preservation']['dispatch_intent_sha256'] = intent['sha256']
        receipts['audit'] = core.encode(audit)
        return intent

    case = cases['monomer-0001']
    receipts, views, tensor = fixture_module.delivered(core, case)
    intent = reserve(case, receipts, views, None)
    bound = None
    if append:
        context, checkpoint = core.initialize(pool, cases, receipts, core.digest(receipts['audit']),
                                              views['delivered'], tensor)
        store.store_receipts(receipts)
        checkpoint_raw = core.encode(checkpoint)
        bound = dict(context_raw=context, context_sha256=core.digest(context),
            checkpoint_raw=checkpoint_raw, checkpoint_sha256=core.digest(checkpoint_raw),
            proof_blobs={core.digest(raw): raw for raw in receipts.values()})
        case = core.bind_next_case(cases['monomer-0002'], checkpoint_raw, core.digest(checkpoint_raw),
                                   context, core.digest(context), bound['proof_blobs'])
        receipts, views, tensor = fixture_module.delivered(core, case, checkpoint['snapshot'])
        intent = reserve(case, receipts, views, bound)
    data, output = tmp_path/'data', tmp_path/'output'
    data.mkdir()
    output.mkdir()
    case_path = tmp_path/'case.json'
    case_path.write_bytes(receipts['case'])
    for role, name in dict(outcome='pilot-outcome.json', envelope='server-native-envelope.json').items():
        (data/name).write_bytes(receipts[role])
    (data/'dispatch-intent.json').write_bytes(core.encode(intent))
    (data/'server-tensors.npz').write_bytes(tensor)
    if append:
        for role, name in dict(before='owned-before-entry.json', entry='owned-after-entry.json',
                              record='owned-after-record.json', delivered='owned-after-delivery.json').items():
            (data/name).write_bytes(core.encode(views[role]))
        audit = json.loads(receipts['audit'])
        audit['captured_preservation'] = module.validate_preservation(data, json.loads(receipts['outcome']),
            case, bound, core.normalize_specification(case['specification']))
        receipts['audit'] = core.encode(audit)
    else:
        (data/'bank-initial-snapshot.json').write_bytes(core.encode(views['delivered']))
    # A real audit allocation is distinct from the science allocation.
    monkeypatch.setenv('SLURM_JOB_ID', '9999')
    live, calls = [copy.deepcopy(views['delivered']) for _ in range(2)], []
    def snapshot(client, **kwargs):
        assert kwargs == dict(dataset_id=9999, max_entries=3)
        calls.append('readonly-snapshot')
        return live.pop(0)
    monkeypatch.setattr(base, 'snapshot_dataset', snapshot)
    kwargs = dict(pool=pool, prepared=cases, raw_audit=receipts['audit'], data=data,
        case_path=case_path, draft=case, append_inputs=bound, output=output)
    return module, kwargs, live, calls


@pytest.mark.parametrize('append', [False, True])
def test_only_ordinary_audited_live_agreement_publishes_context_and_checkpoint(
        monkeypatch, tmp_path, append):
    module, kwargs, _, calls = fixture(monkeypatch, tmp_path, append)
    result = module.publish(NS(), **kwargs)
    assert result['generation'] == (2 if append else 1)
    assert result['whole_bank_complete'] is False and len(result['refused_ids']) == 128
    assert calls == ['readonly-snapshot', 'readonly-snapshot']
    core = module.helper('audited_bank')
    context, checkpoint = (Path(result[name]).read_bytes() for name in ('context_path', 'checkpoint_path'))
    context_data, checkpoint_data = json.loads(context), json.loads(checkpoint)
    required = set(context_data['seed_receipts_sha256'].values())
    for link in checkpoint_data['ledger']:
        required.update(link['receipts_sha256'].values())
    proofs = module.helper('bank_store').read_proofs(required)
    core.validate_checkpoint(checkpoint, result['checkpoint_sha256'],
                              context, result['context_sha256'], proofs)


@pytest.mark.parametrize('append', [False, True])
def test_failed_report_refuses_before_live_queries_or_state_publication(monkeypatch, tmp_path, append):
    module, kwargs, _, calls = fixture(monkeypatch, tmp_path, append)
    audit = json.loads(kwargs['raw_audit'])
    audit['status'] = 'FAIL'
    kwargs['raw_audit'] = module.helper('audited_bank').encode(audit)
    with pytest.raises(ValueError, match='passing ordinary'):
        module.publish(NS(), **kwargs)
    assert not calls and not list(kwargs['output'].iterdir())


@pytest.mark.parametrize('append', [False, True])
def test_live_change_between_readonly_snapshots_never_publishes(monkeypatch, tmp_path, append):
    module, kwargs, live, calls = fixture(monkeypatch, tmp_path, append)
    live[1]['metadata']['name'] = 'changed during audit'
    with pytest.raises(ValueError, match='changed during'):
        module.publish(NS(), **kwargs)
    assert len(calls) == 2 and not list(kwargs['output'].iterdir())


@pytest.mark.parametrize('append', [False, True])
def test_live_signed_zero_change_is_not_hidden_by_python_numeric_equality(monkeypatch, tmp_path, append):
    module, kwargs, live, calls = fixture(monkeypatch, tmp_path, append)
    name = kwargs['draft']['entry']['name']
    geometry = live[1]['entries'][name]['molecule']['geometry']
    index = next(i for i, value in enumerate(geometry) if value == 0.0)
    geometry[index] = -0.0
    assert live[0] == live[1]
    with pytest.raises(ValueError, match='changed during'):
        module.publish(NS(), **kwargs)
    assert len(calls) == 2 and not list(kwargs['output'].iterdir())


def test_captured_initial_extras_or_changed_intent_cannot_bootstrap(monkeypatch, tmp_path):
    module, kwargs, _, _ = fixture(monkeypatch, tmp_path)
    core = module.helper('audited_bank')
    path = kwargs['data']/'bank-initial-snapshot.json'
    initial = json.loads(path.read_bytes())
    initial['associations']['extra'] = {}
    path.write_bytes(core.encode(initial))
    with pytest.raises(ValueError, match='only the fresh singleton'):
        module.validate_initial_capture(kwargs['data'],
            json.loads((kwargs['data']/'pilot-outcome.json').read_bytes()), kwargs['draft'])


@pytest.mark.parametrize('stage', ['owned-after-record.json', 'owned-after-delivery.json'])
def test_all_new_entry_metadata_is_preserved_in_captured_append_views(monkeypatch, tmp_path, stage):
    module, kwargs, _, _ = fixture(monkeypatch, tmp_path, True)
    core = module.helper('audited_bank')
    path = kwargs['data']/stage
    view = json.loads(path.read_bytes())
    view['entries'][kwargs['draft']['entry']['name']]['comment'] = 'unchecked metadata changed'
    path.write_bytes(core.encode(view))
    with pytest.raises(ValueError, match='new bank entry changed'):
        module.validate_preservation(kwargs['data'],
            json.loads((kwargs['data']/'pilot-outcome.json').read_bytes()), kwargs['draft'],
            kwargs['append_inputs'], core.normalize_specification(kwargs['draft']['specification']))


def test_changed_saved_intent_is_rejected_even_with_unchanged_tensor_payload(monkeypatch, tmp_path):
    module, kwargs, _, _ = fixture(monkeypatch, tmp_path, True)
    core = module.helper('audited_bank')
    path = kwargs['data']/'dispatch-intent.json'
    intent = json.loads(path.read_bytes())
    intent['sha256'] = 'f'*64
    path.write_bytes(core.encode(intent))
    with pytest.raises(ValueError, match='intent'):
        module.validate_preservation(kwargs['data'],
            json.loads((kwargs['data']/'pilot-outcome.json').read_bytes()), kwargs['draft'],
            kwargs['append_inputs'], core.normalize_specification(kwargs['draft']['specification']))


def test_intermediate_entry_changed_after_passing_report_cannot_advance(monkeypatch, tmp_path):
    module, kwargs, _, _ = fixture(monkeypatch, tmp_path, True)
    core = module.helper('audited_bank')
    path = kwargs['data']/'owned-after-entry.json'
    view = json.loads(path.read_bytes())
    view['entries'][kwargs['draft']['entry']['name']]['comment'] = 'changed after report'
    path.write_bytes(core.encode(view))
    with pytest.raises(ValueError):
        module.publish(NS(), **kwargs)
