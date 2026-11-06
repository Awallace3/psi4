"""Real wave/runtime/SDK integration; synthetic metadata and mock HTTP only.

Run explicitly with home p4qca (QCPortal 0.63). Production path/descriptor
reads are replaced by repeated pure validation of this local synthetic chain;
no plan/registration/permit/capture/slot guard is replaced.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import requests
from qcportal.record_models import ComputeHistory, RecordStatusEnum
from qcportal.singlepoint import SinglepointRecord


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_wave_runtime_integration_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TIME = '2026-01-01T00:00:00+00:00'
TENSOR = b'SYNTHETIC_TENSOR_TRANSPORT_ONLY'


def fixture(monkeypatch, tmp_path):
    source = load('test_bank_wave')
    f = source.fixture(tmp_path)
    runtime = load('bank_wave_runtime')
    core, transport = runtime.helper('bank_wave'), runtime.helper('bank_wave_transport')
    base = runtime.helper('dataset_guard')
    records, by_entry = {}, {}
    for name, associations in f.registered['associations'].items():
        identifier = next(iter(associations.values()))
        summary = f.registered['records'][str(identifier)]
        records[identifier] = SinglepointRecord(id=identifier, is_service=False,
            status=summary['status'], created_on=TIME, modified_on=TIME,
            specification=summary['specification'], molecule_id=summary['molecule_id'],
            task=(dict(summary['task'], id=identifier+100000, record_id=identifier,
                compute_priority=1, required_programs=['psi4']) if summary['task'] else None),
            compute_history=[], extras={'user:metadata': {'note': 'immutable fixture metadata'}})
        by_entry[name] = identifier
    prefix_id = f.previous['audited_records'][0]
    seed_envelope = next(json.loads(raw) for raw in f.binding['proof_blobs'].values()
                         if json.loads(raw).get('schema') == 'native-monomer-artifacts/2')

    def finish(mid, envelope=None):
        identifier = prefix_id if mid is None else f.reg['members'][mid]['record_id']
        model = records[identifier]
        envelope = seed_envelope if mid is None else json.loads(f.receipts[mid]['envelope'])
        model.status = RecordStatusEnum.complete
        model.manager_name = ('SYNTHETIC_MANAGER_1' if mid is None
                              else 'SYNTHETIC_MANAGER_'+str(int(mid[-4:])))
        model.task_ = None
        model.compute_history_ = [ComputeHistory(id=identifier+200000, record_id=identifier,
            status='complete', manager_name=model.manager_name, modified_on=TIME, provenance=None)]
        model.extras = dict(model.extras, **{'psi4:native_monomer': envelope})
        return envelope
    finish(None)

    class Dataset:
        id, name = 357, f.registered['metadata']['name']
        specifications = f.registered['specifications']
        entry_names = list(by_entry)

        def fetch_specifications(self, **kwargs):
            assert kwargs == {'force_refetch': True}

        def fetch_entry_names(self):
            pass

        def iterate_entries(self, **kwargs):
            for name, entry in f.registered['entries'].items():
                yield NS(name=name, dict=lambda by_alias, entry=entry: copy.deepcopy(entry))

        def iterate_records(self, **kwargs):
            assert kwargs == {'force_refetch': True, 'include': ['compute_history', 'task', 'molecule']}
            for name, identifier in by_entry.items():
                yield name, f.cases[f.plan['member_ids'][0]]['specification_name'], records[identifier]

        def get_record(self, name, spec, **kwargs):
            assert kwargs == {'force_refetch': True}
            return records[by_entry[name]]

    dataset = Dataset()
    # Initial observation uses exactly the same snapshot engine as registration.
    initial_client = NS(
        make_request=lambda *a: copy.deepcopy(f.registered['metadata']),
        get_dataset_by_id=lambda identifier: dataset,
        query_singlepoints=lambda **kwargs: list(records.values()))
    observed = transport._snapshot_dataset(initial_client, f.plan)
    previous = copy.deepcopy(f.previous)
    previous['snapshot']['records'][str(prefix_id)] = observed['records'][str(prefix_id)]
    previous['snapshot_sha256'] = base.digest(previous['snapshot'])
    # Build a new synthetic seed commitment for the actual SDK prefix body,
    # rather than changing an already accepted prefix under its old hash.
    context = json.loads(f.binding['context_raw'])
    context['seed_snapshot_sha256'] = previous['snapshot_sha256']
    context_raw = core.encode(context)
    f.binding.update(context_raw=context_raw, context_sha256=core.digest(context_raw))
    previous['bootstrap_context_sha256'] = core.digest(context_raw)
    previous_raw = core.encode(previous)
    core.validate_checkpoint(previous_raw, core.digest(previous_raw),
        f.binding['context_raw'], f.binding['context_sha256'], f.binding['proof_blobs'])
    f.plan.update(predecessor_sha256=core.digest(previous_raw),
                  bootstrap_context_sha256=core.digest(context_raw),
                  predecessor_snapshot_sha256=previous['snapshot_sha256'])
    f.cases = {mid: core.bind_member(f.prepared[mid], plan=f.plan, monomer_id=mid)
               for mid in f.plan['member_ids']}
    for mid, case in f.cases.items():
        f.plan['members'][mid]['case_sha256'] = core.digest(core.encode(case))
    plan_raw = core.encode(f.plan)
    before, captures = previous['snapshot'], {}
    for mid, case in f.cases.items():
        slot, name = f.reg['members'][mid], case['entry']['name']
        entry = copy.deepcopy(before)
        entry['entries'][name] = copy.deepcopy(observed['entries'][name])
        entry['associations'][name] = {case['specification_name']: None}
        record = copy.deepcopy(entry)
        record['records'][str(slot['record_id'])] = copy.deepcopy(observed['records'][str(slot['record_id'])])
        record['associations'][name][case['specification_name']] = slot['record_id']
        captures[mid] = dict(before=before, entry=entry, record=record)
        slot['case_sha256'] = f.plan['members'][mid]['case_sha256']
        for role in captures[mid]:
            slot['captures'][role]['snapshot_sha256'] = base.digest(captures[mid][role])
        before = record
    binding = f.binding
    binding.update(plan_raw=plan_raw, plan_sha256=core.digest(plan_raw), plan=f.plan,
        checkpoint_raw=previous_raw, checkpoint_sha256=core.digest(previous_raw),
        captures=captures, registered_snapshot=before)
    preflights = {}
    for mid, case in f.cases.items():
        profile = case['execution_profile']
        preflight = dict(status='PASS_SELECTED_WAVE_CASE_WORKER_PREFLIGHT_ONLY',
            producer_profile='grid400', dataset_mode='append', campaign_scope='eligible-wave',
            dispatch_protocol=core.PROTOCOL, monomer_id=mid, campaign=case['campaign'],
            wave_id=case['wave_id'], pilot_sha256=f.reg['members'][mid]['case_sha256'],
            plan_sha256=binding['plan_sha256'], execution_profile=profile,
            bootstrap_context_sha256=case['bootstrap_context_sha256'],
            checkpoint_sha256=case['checkpoint_sha256'],
            predecessor_snapshot_sha256=case['predecessor_snapshot_sha256'])
        path = f.directory/'members'/mid/'worker/payload/worker-preflight.json'
        path.parent.mkdir(parents=True)
        raw = core.encode(preflight)
        path.write_bytes(raw)
        f.reg['members'][mid]['worker_preflight_sha256'] = core.digest(raw)
        preflights[mid] = preflight
    f.reg.update(plan_sha256=binding['plan_sha256'], predecessor_sha256=core.digest(previous_raw),
        bootstrap_context_sha256=core.digest(context_raw),
        predecessor_snapshot_sha256=previous['snapshot_sha256'],
        registered_snapshot_sha256=base.digest(before))
    reg_raw = core.encode(f.reg)
    permit = dict(binding['release_permit'], plan_sha256=binding['plan_sha256'],
        registration_sha256=core.digest(reg_raw), predecessor_sha256=core.digest(previous_raw),
        members=core.permit_members(f.reg))
    permit_raw = core.encode(permit)
    binding.update(registration=f.reg, registration_raw=reg_raw, registration_sha256=core.digest(reg_raw),
        release_permit=permit, release_permit_raw=permit_raw, release_permit_sha256=core.digest(permit_raw))

    barrier_observations = {}
    def revalidate(binding):
        for raw_key, view_key, sha_key in (
                ('plan_raw', 'plan', 'plan_sha256'),
                ('registration_raw', 'registration', 'registration_sha256'),
                ('release_permit_raw', 'release_permit', 'release_permit_sha256')):
            if core.encode(core.decoded(binding[raw_key], binding[sha_key])) != core.encode(binding[view_key]):
                raise ValueError('synthetic metadata binding changed')
        core.validate_plan(binding['plan_raw'], binding['plan_sha256'], prepared_cases=f.prepared,
            checkpoint_raw=binding['checkpoint_raw'], checkpoint_sha256=binding['checkpoint_sha256'],
            context_raw=binding['context_raw'], context_sha256=binding['context_sha256'],
            proof_blobs=binding['proof_blobs'])
        core.validate_registration(binding['plan'], binding['registration_raw'],
            binding['registration_sha256'], predecessor_snapshot=previous['snapshot'],
            cases=f.cases, captures=binding['captures'], registered_snapshot=binding['registered_snapshot'])
        core.validate_release_permit(binding['plan'], binding['plan_sha256'], binding['registration'],
            binding['registration_sha256'], binding['release_permit_raw'])
        if binding['science_barrier'] is not None:
            core.validate_barrier(binding['plan'], binding['registration'],
                binding['release_permit_raw'], binding['science_barrier_raw'], **barrier_observations)
    # Synthetic local chain has generation1 and tmp paths, not production pins.
    # Replace only production authority reads, retaining all pure chain guards.
    monkeypatch.setattr(runtime.helper('bank_wave_campaign'), 'revalidate_binding', revalidate)
    monkeypatch.setattr(transport.helper('bank_wave_campaign'), 'revalidate_binding', revalidate)
    revalidate(binding)
    Client = transport.readonly_client_class(binding, monomer_id=binding['monomer_id'])
    client = object.__new__(Client)
    http_fixture = load('transport_cb63305_tests')
    client.__dict__.update(http_fixture.legacy_fixtures.client([], adapted=False).__dict__)
    def response(request, **kwargs):
        assert request.method == 'GET' and request.url.endswith('/api/v1/datasets/357')
        client._req_session.sent.append((request, kwargs))
        result = requests.Response()
        result.status_code, result.request = 200, request
        result.headers['Content-Type'] = 'application/json'
        result._content = json.dumps(f.registered['metadata']).encode()
        return result
    client._req_session.send = response
    client.get_dataset_by_id = lambda identifier: dataset
    client.query_singlepoints = lambda **kwargs: list(records.values())
    client.get_server_information = lambda: {'name': 'CDS QCFractal Server', 'version': '0.70'}
    dataset._client = client
    mid, case = binding['monomer_id'], f.cases[binding['monomer_id']]
    profile, slot = case['execution_profile'], f.reg['members'][mid]
    allocation = dict(JobId=slot['science_job_id'], Account='gts-cs207-chemx', QOS='embers',
        Partition='cpu-small', NumCPUs=str(profile['cores']), NumNodes='1',
        MinMemoryNode=str(profile['job_memory_GiB'])+'G', TimeLimit=profile['time_limit'])
    saved = {}
    def save(name, data):
        assert name not in saved
        saved[name] = copy.deepcopy(data)
    return NS(**(vars(f) | locals()))


def test_real_core_registered_record_uses_actual_snapshot_and_slot(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    result = f.runtime.registered_record(f.client, f.binding, f.case, f.allocation,
        f.preflights[f.mid], f.save)
    assert result['record'] is f.records[f.slot['record_id']]
    assert f.saved['record-created.json']['record_id'] == f.slot['record_id']
    assert f.client._req_session.sent and all(
        req.method == 'GET' for req, _ in f.client._req_session.sent)


def test_real_sdk_completion_is_valid_with_peers_still_waiting(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    envelope = f.finish(f.mid)
    summary = f.runtime.capture_delivery(f.client, f.binding, f.case,
        record_id=f.slot['record_id'], manager=f.records[f.slot['record_id']].manager_name,
        envelope=envelope, tensor_raw=TENSOR, save=f.save)
    assert summary['peer_statuses'][f.mid] == 'complete'
    assert set(summary['peer_statuses'].values()) == {'complete', 'waiting'}
    assert set(f.saved) == {'owned-after-delivery.json'}


def test_real_peer_completion_does_not_prevent_own_waiting_slot_consumption(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    peer = f.plan['member_ids'][1]
    f.finish(peer)
    result = f.runtime.registered_record(f.client, f.binding, f.case, f.allocation,
        f.preflights[f.mid], f.save)
    assert result['record'].status.value == 'waiting'
    assert f.records[f.reg['members'][peer]['record_id']].status.value == 'complete'


def test_actual_sdk_waiting_running_complete_identity_is_stable_and_body_changes(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    peer = f.plan['member_ids'][1]
    identifier = f.reg['members'][peer]['record_id']
    record = f.records[identifier]
    waiting = f.transport.snapshot_dataset(f.client, binding=f.binding)['records'][str(identifier)]
    record.status = RecordStatusEnum.running
    record.manager_name = 'SYNTHETIC_MANAGER_'+str(int(peer[-4:]))
    record.compute_history_ = [ComputeHistory(id=identifier+200000, record_id=identifier,
        status='running', manager_name=record.manager_name, modified_on=TIME, provenance=None)]
    running_snapshot = f.transport.snapshot_dataset(f.client, binding=f.binding)
    running = running_snapshot['records'][str(identifier)]
    assert f.core.validate_capture(f.plan, f.reg, predecessor_snapshot=f.previous['snapshot'],
        registered_snapshot=f.binding['registered_snapshot'], snapshot=running_snapshot
        )['peer_statuses'][peer] == 'running'
    # Real own-slot consumption must tolerate this declared running peer.
    f.runtime.registered_record(f.client, f.binding, f.case, f.allocation,
        f.preflights[f.mid], f.save)
    f.finish(peer)
    completed_snapshot = f.transport.snapshot_dataset(f.client, binding=f.binding)
    completed = completed_snapshot['records'][str(identifier)]
    assert f.core.validate_capture(f.plan, f.reg, predecessor_snapshot=f.previous['snapshot'],
        registered_snapshot=f.binding['registered_snapshot'], snapshot=completed_snapshot
        )['peer_statuses'][peer] == 'complete'
    assert len({view['body_sha256'] for view in (waiting, running, completed)}) == 3
    assert len({view['wave_identity_sha256'] for view in (waiting, running, completed)}) == 1
    assert 'compute_history' in f.base.plain(record) and 'task' in f.base.plain(record)


def barrier(f):
    outcomes, accounting, deliveries, members = {}, {}, {}, {}
    for mid in f.plan['member_ids']:
        f.finish(mid)
        slot, case = f.reg['members'][mid], f.cases[mid]
        snapshot = f.transport.snapshot_dataset(f.client, binding=f.binding)
        outcome = dict(f.runtime.started_fields(f.binding),
            status='DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION',
            dataset_id=357, record_id=slot['record_id'], monomer_id=mid, campaign=case['campaign'],
            producer_profile='grid400', dataset_mode='append', allocation={'JobId': slot['science_job_id']},
            compute_tag=slot['compute_tag'], pilot_sha256=slot['case_sha256'],
            worker_preflight_sha256=slot['worker_preflight_sha256'], runtime_unchanged=True,
            maximum_records=1, find_existing=False, qcengine_internal_retries=0,
            tensors_sha256=f.core.digest(TENSOR), tensor_bytes=len(TENSOR),
            manager=f.records[slot['record_id']].manager_name)
        run = f.directory/'members'/mid/'science'
        data = run/'payload'
        data.mkdir(parents=True)
        for name, value in {
            'pilot-outcome.json': outcome, 'owned-after-delivery.json': snapshot,
            'server-native-envelope.json': f.records[slot['record_id']].extras['psi4:native_monomer'],
        }.items():
            f.runtime.write_json_once(data/name, value)
        (data/'server-tensors.npz').write_bytes(TENSOR)
        outcomes[mid] = (data/'pilot-outcome.json').read_bytes()
        accounting[mid] = f.source.accounting(slot['science_job_id'])
        (run/'accounting.txt').write_bytes(accounting[mid])
        deliveries[mid] = snapshot
        members[mid] = dict(
            **{key: slot[key] for key in ('science_job_id', 'record_id', 'compute_tag', 'case_sha256')},
            outcome_path=str(data/'pilot-outcome.json'), outcome_sha256=f.core.digest(outcomes[mid]),
            delivery_snapshot_sha256=f.base.digest(snapshot), accounting_path=str(run/'accounting.txt'),
            accounting_sha256=f.core.digest(accounting[mid]))
    document = dict(schema='native-eligible-wave-science-barrier/1',
        status='ALL_SCIENCE_POSITIVELY_VERIFIED', wave_id=f.plan['wave_id'],
        plan_sha256=f.binding['plan_sha256'], registration_sha256=f.binding['registration_sha256'],
        release_permit_sha256=f.binding['release_permit_sha256'],
        member_ids=f.plan['member_ids'], members=members)
    raw = f.core.encode(document)
    f.barrier_observations.update(outcomes=outcomes, accounting=accounting, deliveries=deliveries)
    f.binding.update(science_barrier=document, science_barrier_raw=raw,
        science_barrier_path=f.directory/'science-barrier.json', science_barrier_sha256=f.core.digest(raw))
    f.revalidate(f.binding)
    return outcomes, accounting, deliveries


def test_real_whole_barrier_and_runtime_preservation_reduce_actual_own_capture(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    outcomes, _, deliveries = barrier(f)
    data = f.directory/'members'/f.mid/'science/payload'
    outcome = json.loads(outcomes[f.mid])
    report = f.runtime.audit_preservation(data, outcome, f.case, f.binding,
        f.runtime.helper('audited_bank').normalize_specification(f.case['specification']))
    assert report['status'] == 'PASS_CAPTURED_WAVE_DATASET_PRESERVATION'
    assert report['before_sha256'] == f.base.digest(f.binding['registered_snapshot'])
    assert report['after_sha256'] == f.base.digest(deliveries[f.mid])
    # Own actual capture predates peer completion. Auditing after the barrier
    # must not pretend it was an all-complete observation.
    assert set(report['peer_statuses'].values()) == {'complete', 'waiting'}
    assert report['preserved_records'] == 1
    with pytest.raises(ValueError, match='passing ordinary'):
        load('bank_audit').passing_report(f.core.encode(dict(
            status='PASS_SELECTED_WAVE_CASE_SINGLETON_EVIDENCE',
            campaign_scope='eligible-wave', producer_profile='grid400', dataset_mode='append')), {})


def test_runtime_audit_cannot_ignore_failed_peer_accounting_in_real_barrier(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    outcomes, accounting, _ = barrier(f)
    peer = f.plan['member_ids'][-1]
    accounting[peer] = accounting[peer].replace(b'COMPLETED|0:0', b'FAILED|1:0')
    with pytest.raises(ValueError):
        f.runtime.audit_preservation(f.directory/'members'/f.mid/'science/payload',
            json.loads(outcomes[f.mid]), f.case, f.binding,
            f.runtime.helper('audited_bank').normalize_specification(f.case['specification']))


@pytest.mark.parametrize('mutation', ['name', 'specification', 'history', 'prefix',
                                    'extras_add', 'extras_native_prefix', 'extras_change', 'extras_remove'])
def test_real_snapshot_rejects_immutable_peer_or_prefix_changes(monkeypatch, tmp_path, mutation):
    f = fixture(monkeypatch, tmp_path)
    peer = f.plan['member_ids'][1]
    f.finish(peer)
    record = f.records[f.reg['members'][peer]['record_id']]
    if mutation == 'name':
        record.name = 'unexpected new immutable name'
    elif mutation == 'specification':
        record.specification.method = 'hf'
    elif mutation == 'history':
        record.compute_history_ *= 2
    elif mutation == 'prefix':
        f.records[f.prefix_id].name = 'accepted prefix changed'
    elif mutation == 'extras_add':
        record.extras['unknown:metadata'] = {'new': 'not permitted'}
    elif mutation == 'extras_native_prefix':
        record.extras['psi4:native_monomer:unreviewed'] = {'new': 'not the exact result slot'}
    elif mutation == 'extras_change':
        record.extras['user:metadata']['note'] = 'changed after registration'
    else:
        del record.extras['user:metadata']
    with pytest.raises(ValueError):
        f.runtime.registered_record(f.client, f.binding, f.case, f.allocation,
            f.preflights[f.mid], f.save)
    assert not f.saved


@pytest.mark.parametrize('change', ['job', 'record', 'tag', 'member', 'worker_hash', 'permit'])
def test_real_core_rejects_wrong_runtime_authority(monkeypatch, tmp_path, change):
    f = fixture(monkeypatch, tmp_path)
    if change == 'job':
        f.allocation['JobId'] = '123456'
    elif change == 'record':
        f.records[f.slot['record_id']].id += 1
    elif change == 'tag':
        f.records[f.slot['record_id']].task_.compute_tag = 'foreign'
    elif change == 'member':
        f.case = f.cases[f.plan['member_ids'][1]]
    elif change == 'worker_hash':
        path = f.directory/'members'/f.mid/'worker/payload/worker-preflight.json'
        path.write_bytes(path.read_bytes()+b' ')
    else:
        f.binding['release_permit_raw'] += b' '
    with pytest.raises((ValueError, KeyError)):
        f.runtime.registered_record(f.client, f.binding, f.case, f.allocation,
            f.preflights[f.mid], f.save)
    assert not f.saved


def test_real_session_guard_binds_wave_allocation_record_tag_and_protocols(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    execution = load('grid400_bank_execution')
    original = execution.helper
    monkeypatch.setattr(execution, 'helper', lambda name:
        f.runtime if name == 'bank_wave_runtime' else original(name))
    runtime = execution.helper('phoenix_worker_probe').runtime_input(f.case)
    runtime['specification']['protocols'] = execution.helper(
        'phoenix_pilot').execution_specification({})['protocols']
    kwargs = dict(runtime=runtime, compute_tag=f.slot['compute_tag'],
        worker_shape=(f.profile['cores'], f.profile['science_memory_GiB']),
        record_id=f.slot['record_id'], campaign_scope='eligible-wave',
        wave_binding=f.binding, allocation=f.allocation)
    assert execution.validate_binding(f.case, **kwargs) == tuple(range(5234314, 5234328))
    runtime['specification']['protocols']['error_correction']['default_policy'] = True
    with pytest.raises(ValueError):
        execution.validate_binding(f.case, **kwargs)
