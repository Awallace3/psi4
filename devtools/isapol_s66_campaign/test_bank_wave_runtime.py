"""Runtime contract seam tests; fake core callbacks are not integrated proofs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_wave_runtime_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**changes):
    return NS(**(dict(producer='grid400', campaign_scope='eligible-wave', remaining_case=True,
        wave_plan=Path('/unit/plan.json'), wave_plan_sha256='a'*64,
        wave_registration=Path('/unit/registration.json')) | changes))


def test_wave_metadata_publication_is_exclusive_and_no_follow(tmp_path):
    runtime = load('bank_wave_runtime')
    path = tmp_path/'evidence.json'
    runtime.write_json_once(path, {'unit': True})
    assert json.loads(path.read_bytes()) == {'unit': True}
    with pytest.raises(FileExistsError):
        runtime.write_json_once(path, {'unit': False})
    alias = tmp_path/'alias.json'
    alias.symlink_to(path)
    with pytest.raises(OSError):
        runtime.write_json_once(alias, {'unit': False})
    assert json.loads(path.read_bytes()) == {'unit': True}


@pytest.mark.parametrize('change', [
    {'wave_plan': None}, {'wave_plan_sha256': None}, {'remaining_case': False},
    {'producer': 'cb63305'}, {'bank_context': Path('/unit/old')},
    {'bank_checkpoint_sha256': 'b'*64}, {'accepted_envelope': Path('/unit/old')},
    {'advance_checkpoint': True}, {'advance_bank_checkpoint': True},
    {'owned_checkpoint': Path('/unit/old')},
])
def test_wave_arguments_fail_before_core_reads(monkeypatch, change):
    runtime = load('bank_wave_runtime')
    monkeypatch.setattr(runtime, 'helper', lambda name: pytest.fail('unexpected core read'))
    with pytest.raises(ValueError):
        runtime.prepare_draft(args(**change), Path('/unit/build-identity.json'))


def test_worker_reconstruction_never_loads_registration(monkeypatch):
    runtime = load('bank_wave_runtime')
    observed = []
    campaign = NS(verify_prepared_from_paths=lambda a, m: observed.append((a, m)) or {'unit': True})
    monkeypatch.setattr(runtime, 'helper', lambda name: campaign if name == 'bank_wave_campaign'
                        else pytest.fail(name))
    arguments, manifest = args(wave_registration=None), Path('/unit/build-identity.json')
    assert runtime.prepare_draft(arguments, manifest) == {'unit': True}
    assert observed == [(arguments, manifest)]


@pytest.mark.parametrize('audit', [False, True])
def test_load_requires_registration_and_audit_requires_whole_barrier(monkeypatch, audit):
    runtime = load('bank_wave_runtime')
    calls = []
    campaign = NS(load_binding=lambda a, **kw: calls.append(kw) or {'science_barrier': None},
                  revalidate_binding=lambda binding: calls.append('revalidated'))
    monkeypatch.setattr(runtime, 'helper', lambda name: campaign)
    if audit:
        arguments = args(case=Path('/unit/science'))
        with pytest.raises(ValueError, match='whole-science barrier'):
            runtime.load_binding(arguments)
    else:
        with pytest.raises(ValueError, match='registration and release permit'):
            runtime.load_binding(args(wave_registration=None))
    assert calls == []
    assert runtime.load_binding(args()) == {'science_barrier': None}
    assert calls == [{'require_registration': True}, 'revalidated']


def test_audit_cannot_trust_barrier_argument_without_verified_binding(monkeypatch):
    runtime = load('bank_wave_runtime')
    monkeypatch.setattr(runtime, 'helper', lambda name: NS(
        load_binding=lambda *a, **kw: {'science_barrier': None}))
    with pytest.raises(ValueError, match='verified whole-science'):
        runtime.load_binding(args(case=Path('/unit/science'),
            wave_science_barrier=Path('/unit/science-barrier.json'),
            wave_science_barrier_sha256='b'*64))


def fixture(monkeypatch, tmp_path):
    runtime = load('bank_wave_runtime')
    base, core = runtime.helper('dataset_guard'), runtime.helper('audited_bank')
    profile = load('grid400_bank_policy').execution_profile('s20')
    mid, tag = 'monomer-0026', 'ccasp-native-grid400-bank-monomer-0026-123'
    draft = dict(monomer_id=mid, dataset_name='SYNTHETIC_DATASET', campaign='SYNTHETIC_CAMPAIGN',
        specification_name='synthetic-spec', execution_profile=profile,
        specification=dict(program='psi4', driver='properties', method='pbe0', basis='x',
                           keywords={}, protocols={}),
        entry=dict(name=mid+'-entry', additional_keywords={'synthetic': True}))
    slot = dict(record_id=9000, molecule_id=8000, compute_tag=tag, science_job_id='123',
                case_sha256='c'*64)
    plan = dict(wave_id='wave-0025-0033', dataset_id=357,
                predecessor_snapshot_sha256='d'*64)
    binding = dict(plan_path=tmp_path/'plan.json', monomer_id=mid, plan=plan,
        plan_sha256='a'*64, context_sha256='b'*64, checkpoint_sha256='e'*64,
        checkpoint_raw=b'{"snapshot": {}}',
        registration=dict(dispatch_intent_sha256='f'*64), registration_sha256='1'*64,
        release_permit={}, release_permit_sha256='2'*64,
        science_barrier=None, registered_snapshot={})
    preflight = dict(status='PASS_SELECTED_WAVE_CASE_WORKER_PREFLIGHT_ONLY',
        pilot_sha256=slot['case_sha256'], wave_id=plan['wave_id'],
        plan_sha256=binding['plan_sha256'], execution_profile=profile)
    path = tmp_path/'members'/mid/'worker/payload/worker-preflight.json'
    path.parent.mkdir(parents=True)
    raw = json.dumps(preflight).encode()
    path.write_bytes(raw)
    slot['worker_preflight_sha256'] = hashlib.sha256(raw).hexdigest()
    effective = core.normalize_specification(draft['specification'])
    effective['keywords'].update(draft['entry']['additional_keywords'])
    own = dict(status='waiting', history=[], task={'compute_tag': tag},
               specification=effective)
    snapshot = dict(records={'9000': own})
    dataset = NS(id=357, name=draft['dataset_name'])
    record = NS(id=9000, molecule_id=8000, status=NS(value='waiting'),
                compute_history=[], task=NS(compute_tag=tag))
    dataset.get_record = lambda *a, **kw: record
    client = NS(get_dataset_by_id=lambda identifier: dataset,
        get_server_information=lambda: {'name': 'CDS QCFractal Server', 'version': '0.70'})
    dataset._client = client
    events, saved = [], {}
    def validate_slot(plan, registration, permit, case, *, allocation=None):
        events.append(('slot', allocation))
        if allocation is not None and allocation.get('JobId') != slot['science_job_id']:
            raise ValueError('wrong actual job')
        return slot
    fake_core = NS(validate_registered_slot=validate_slot,
        validate_capture=lambda *a, **kw: events.append(('capture', kw)) or
            dict(preserved_records=25, peer_statuses={'monomer-0027': 'running'}))
    def require(client_arg, binding_arg, *, monomer_id):
        events.append('readonly')
        if client_arg is not client or binding_arg is not binding or monomer_id != mid:
            raise ValueError('foreign client')
    transport = NS(require_readonly_client=require,
        snapshot_dataset=lambda client, *, binding: snapshot)
    campaign = NS(revalidate_binding=lambda binding: events.append('revalidate'))
    original = runtime.helper
    monkeypatch.setattr(runtime, 'helper', lambda name:
        {'bank_wave': fake_core, 'bank_wave_transport': transport,
         'bank_wave_campaign': campaign}.get(name) or original(name))
    def save(name, data):
        assert name not in saved
        saved[name] = copy.deepcopy(data)
    return NS(**locals())


def test_registered_record_reads_verified_slot_without_dataset_writes(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    result = f.runtime.registered_record(f.client, f.binding, f.draft,
        {'JobId': '123'}, f.preflight, f.save)
    assert result['dataset'] is f.dataset and result['record'] is f.record
    assert result['created'] == f.saved['record-created.json']
    assert result['created']['dispatch_intent_sha256'] == 'f'*64
    assert ('slot', {'JobId': '123'}) in f.events
    assert 'readonly' in f.events
    assert set(f.saved) == {'record-created.json'}


@pytest.mark.parametrize('change', ['job', 'member', 'record', 'molecule', 'tag', 'running', 'history',
                                  'preflight', 'preflight_hash', 'foreign_client', 'server'])
def test_registered_record_rejects_changed_authority_before_manager(monkeypatch, tmp_path, change):
    f = fixture(monkeypatch, tmp_path)
    allocation, client = {'JobId': '123'}, f.client
    if change == 'job':
        allocation['JobId'] = '456'
    elif change == 'member':
        f.draft['monomer_id'] = 'monomer-0027'
    elif change == 'record':
        f.record.id += 1
    elif change == 'molecule':
        f.record.molecule_id += 1
    elif change == 'tag':
        f.record.task.compute_tag = 'foreign'
    elif change == 'running':
        f.record.status.value = 'running'
    elif change == 'history':
        f.record.compute_history = [{'manager': 'foreign'}]
    elif change == 'preflight':
        f.preflight['execution_profile'] = {}
    elif change == 'preflight_hash':
        f.slot['worker_preflight_sha256'] = '0'*64
    elif change == 'foreign_client':
        client = NS()
    else:
        f.client.get_server_information = lambda: {'name': 'foreign', 'version': '0.70'}
    with pytest.raises(ValueError):
        f.runtime.registered_record(client, f.binding, f.draft, allocation, f.preflight, f.save)
    assert f.saved == {}


def completed(f):
    tensor = b'SYNTHETIC_NOT_A_TENSOR'
    envelope = dict(artifacts={'tensors.npz': dict(
        size=len(tensor), sha256=hashlib.sha256(tensor).hexdigest())})
    own = f.snapshot['records']['9000']
    own.update(status='complete', history=[{'manager_name': 'SYNTHETIC_MANAGER'}], task=None,
        native_envelope_sha256=f.base.digest(envelope), tensor_sha256=hashlib.sha256(tensor).hexdigest())
    return envelope, tensor


def test_delivery_captures_only_actual_own_delivery_not_registration_observations(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    envelope, tensor = completed(f)
    summary = f.runtime.capture_delivery(f.client, f.binding, f.draft, record_id=9000,
        manager='SYNTHETIC_MANAGER', envelope=envelope, tensor_raw=tensor, save=f.save)
    assert set(f.saved) == {'owned-after-delivery.json'}
    assert f.saved['owned-after-delivery.json'] == f.snapshot
    assert summary['preserved_records'] == 25
    assert f.events[-1][1]['member_id'] == f.mid
    assert f.events[-1][1]['require_complete'] is False


@pytest.mark.parametrize('change', ['record', 'manager', 'status', 'history', 'specification',
                                  'tensor', 'envelope', 'oversize'])
def test_delivery_preserves_exact_own_completion_gates(monkeypatch, tmp_path, change):
    f = fixture(monkeypatch, tmp_path)
    envelope, tensor = completed(f)
    record, manager = 9000, 'SYNTHETIC_MANAGER'
    own = f.snapshot['records']['9000']
    if change == 'record':
        record = 9999
    elif change == 'manager':
        manager = 'foreign'
    elif change == 'status':
        own['status'] = 'running'
    elif change == 'history':
        own['history'] *= 2
    elif change == 'specification':
        own['specification']['keywords']['synthetic'] = False
    elif change == 'tensor':
        tensor += b'changed'
    elif change == 'envelope':
        envelope['changed'] = True
    else:
        tensor = bytes(16*1024**2+1)
    with pytest.raises(ValueError):
        f.runtime.capture_delivery(f.client, f.binding, f.draft, record_id=record,
            manager=manager, envelope=envelope, tensor_raw=tensor, save=f.save)
    assert f.saved == {}


def test_audit_preservation_requires_barrier_and_reduces_actual_delivery(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    envelope, tensor = completed(f)
    outcome = dict(f.runtime.started_fields(f.binding), record_id=9000,
        dataset_id=357, monomer_id=f.mid, campaign=f.draft['campaign'],
        producer_profile='grid400', dataset_mode='append',
        compute_tag=f.tag, allocation={'JobId': '123'}, manager='SYNTHETIC_MANAGER',
        worker_preflight_sha256=f.slot['worker_preflight_sha256'], pilot_sha256='c'*64,
        tensors_sha256=hashlib.sha256(tensor).hexdigest(), tensor_bytes=len(tensor))
    data = tmp_path/'science/payload'
    data.mkdir(parents=True)
    for name, document in {'pilot-outcome.json': outcome,
        'owned-after-delivery.json': f.snapshot, 'server-native-envelope.json': envelope}.items():
        (data/name).write_bytes(json.dumps(document).encode())
    (data/'server-tensors.npz').write_bytes(tensor)
    spec = f.core.normalize_specification(f.draft['specification'])
    with pytest.raises(ValueError, match='whole-science barrier'):
        f.runtime.audit_preservation(data, outcome, f.draft, f.binding, spec)
    barrier_member = dict(delivery_snapshot_sha256=f.base.digest(f.snapshot),
        outcome_sha256=hashlib.sha256((data/'pilot-outcome.json').read_bytes()).hexdigest())
    f.binding['science_barrier'] = dict(members={f.mid: barrier_member})
    report = f.runtime.audit_preservation(data, outcome, f.draft, f.binding, spec)
    assert report['status'] == 'PASS_CAPTURED_WAVE_DATASET_PRESERVATION'
    assert report['registration_sha256'] == f.binding['registration_sha256']
    assert report['before_sha256'] == f.base.digest(f.binding['registered_snapshot'])
    barrier_member['delivery_snapshot_sha256'] = '0'*64
    with pytest.raises(ValueError, match='actual saved delivery'):
        f.runtime.audit_preservation(data, outcome, f.draft, f.binding, spec)
