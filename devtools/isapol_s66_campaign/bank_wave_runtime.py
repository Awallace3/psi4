"""Registered-wave runtime adapter; never registers, submits, or advances.

Dataset reads use the wave-owned read-only client. The separate existing
single-record manager owns its sole claim/result lifecycle, not this module.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_MODULES = {}
_HELPERS = {
    'bank_wave', 'bank_wave_campaign', 'bank_wave_transport', 'bank_store', 'bank_campaign',
    'audited_bank', 'dataset_guard', 'phoenix_pilot',
}


def helper(name):
    if name not in _HELPERS:
        raise ValueError('closed registered-wave runtime helper required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_wave_runtime_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def _raw(path, maximum=32*1024**2):
    return helper('bank_store')._read_regular(Path(path), maximum=maximum)


def _read(path):
    return json.loads(_raw(path))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _same(left, right):
    return helper('audited_bank').encode(left) == helper('audited_bank').encode(right)


def write_json_once(path, value):
    """Exclusive no-follow/fsynced metadata publication for wave payloads."""
    helper('bank_store')._write_once(Path(path), helper('audited_bank').encode(value))


def validate_args(args, *, registered=False):
    if (getattr(args, 'producer', None) != 'grid400'
            or getattr(args, 'campaign_scope', None) != 'eligible-wave'
            or getattr(args, 'remaining_case', False) is not True
            or getattr(args, 'wave_plan', None) is None
            or getattr(args, 'wave_plan_sha256', None) is None):
        raise ValueError('explicit grid400 append wave plan required')
    forbidden = (
        'bank_context', 'bank_context_sha256', 'bank_checkpoint', 'bank_checkpoint_sha256',
        'owned_checkpoint', 'checkpoint_sha256', 'checkpoint_proofs',
        'accepted_audit', 'accepted_outcome', 'accepted_pilot', 'accepted_envelope',
        'advance_checkpoint', 'advance_bank_checkpoint',
    )
    if any(getattr(args, key, None) not in (None, False) for key in forbidden):
        raise ValueError('wave cannot consume serial inputs or advance singleton checkpoints')
    audit = getattr(args, 'case', None) is not None
    barrier = (getattr(args, 'wave_science_barrier', None),
               getattr(args, 'wave_science_barrier_sha256', None))
    if audit and not all(value is not None for value in barrier):
        raise ValueError('wave member audit requires the whole-science barrier')
    if not audit and any(value is not None for value in barrier):
        raise ValueError('only wave audit consumes the whole-science barrier')
    if registered and getattr(args, 'wave_registration', None) is None:
        raise ValueError('registered wave requires registration and release permit')


def prepare_draft(args, manifest_path):
    validate_args(args)
    return helper('bank_wave_campaign').verify_prepared_from_paths(args, manifest_path)


def load_binding(args):
    validate_args(args, registered=True)
    campaign = helper('bank_wave_campaign')
    binding = campaign.load_binding(args, require_registration=True)
    if getattr(args, 'case', None) is not None and binding['science_barrier'] is None:
        raise ValueError('verified whole-science barrier required for wave audit')
    campaign.revalidate_binding(binding)
    return binding


def pins():
    return helper('bank_wave_campaign').pins()


def source_names():
    return tuple(dict.fromkeys(
        helper('bank_campaign').ADDITIONAL_SOURCES+
        helper('bank_wave_campaign').source_names()+(
            'bank_wave_runtime.py', 'native_controller.py',
            'runtime_audit_cb63305.py', 'runtime_audit_cb63305.sbatch')))


def started_fields(binding):
    plan, registration = binding['plan'], binding['registration']
    if registration is None or binding['release_permit'] is None:
        raise ValueError('registered wave bindings required')
    return dict(campaign_scope='eligible-wave', dispatch_protocol='registered-wave/1',
        wave_id=plan['wave_id'], plan_sha256=binding['plan_sha256'],
        bootstrap_context_sha256=binding['context_sha256'],
        checkpoint_sha256=binding['checkpoint_sha256'],
        predecessor_snapshot_sha256=plan['predecessor_snapshot_sha256'],
        registration_sha256=binding['registration_sha256'],
        release_permit_sha256=binding['release_permit_sha256'],
        dispatch_intent_sha256=registration['dispatch_intent_sha256'])


def validate_paths(args, *, role):
    """Only the exact closed plan member path may use nested run names."""
    if role not in ('worker', 'science', 'audit'):
        raise ValueError('closed wave runtime role required')
    binding = helper('bank_wave_campaign').load_binding(args, require_registration=False)
    root, mid = binding['plan_path'].parent, binding['monomer_id']
    if (binding['plan']['source'] != str(Path(__file__).resolve().parents[2])
            or binding['plan']['operational_commit'] != args.commit):
        raise ValueError('wave plan must bind actual operational checkout and commit')
    expected = root/'members'/mid/role
    # Campaign validates the closed root, no-follow plan/case and member selection.
    # Refuse aliases here as well; resolve alone would admit a symlink spelling.
    if args.run != expected or args.run.resolve() != expected:
        raise ValueError('exact wave member run directory required')
    if role != 'worker':
        preflight = root/'members'/mid/'worker/payload/worker-preflight.json'
        if args.worker_preflight != preflight or args.worker_preflight.resolve() != preflight:
            raise ValueError('exact wave member worker proof path required')
    if role == 'audit':
        science = root/'members'/mid/'science'
        if args.case != science or args.case.resolve() != science:
            raise ValueError('exact wave member science directory required')


def _slot(binding, draft, allocation=None):
    if draft.get('monomer_id') != binding['monomer_id']:
        raise ValueError('runtime may consume only its own declared wave member')
    return helper('bank_wave').validate_registered_slot(binding['plan'],
        binding['registration'], binding['release_permit'], draft, allocation=allocation)


def _validate_capture(binding, snapshot, *, mid=None, complete=False):
    return helper('bank_wave').validate_capture(binding['plan'], binding['registration'],
        predecessor_snapshot=json.loads(binding['checkpoint_raw'])['snapshot'],
        registered_snapshot=binding['registered_snapshot'], snapshot=snapshot,
        member_id=mid, require_complete=complete)


def _snapshot(client, binding):
    transport = helper('bank_wave_transport')
    transport.require_readonly_client(client, binding, monomer_id=binding['monomer_id'])
    info = client.get_server_information()
    if info.get('name') != 'CDS QCFractal Server' or info.get('version') != '0.70':
        raise ValueError('approved wave read-only server identity changed')
    return transport.snapshot_dataset(client, binding=binding)


def registered_record(client, binding, draft, allocation, preflight, save):
    helper('bank_wave_campaign').revalidate_binding(binding)
    if type(allocation) is not dict:
        raise ValueError('actual science allocation required to consume registered wave slot')
    slot = _slot(binding, draft, allocation)
    path = (binding['plan_path'].parent/'members'/binding['monomer_id']/
            'worker/payload/worker-preflight.json')
    raw = _raw(path)
    if (_sha(raw) != slot['worker_preflight_sha256'] or not _same(json.loads(raw), preflight)
            or preflight.get('status') != 'PASS_SELECTED_WAVE_CASE_WORKER_PREFLIGHT_ONLY'
            or preflight.get('pilot_sha256') != slot['case_sha256']
            or preflight.get('plan_sha256') != binding['plan_sha256']
            or preflight.get('wave_id') != binding['plan']['wave_id']
            or preflight.get('execution_profile') != draft['execution_profile']):
        raise ValueError('registered slot worker proof differs')
    snapshot = _snapshot(client, binding)
    _validate_capture(binding, snapshot)
    own = snapshot['records'][str(slot['record_id'])]
    helper('dataset_guard').validate_fresh_record(own, tag=slot['compute_tag'],
        specification=helper('audited_bank').normalize_specification(draft['specification']),
        entry=draft['entry'])
    dataset = client.get_dataset_by_id(binding['plan']['dataset_id'])
    record = dataset.get_record(draft['entry']['name'], draft['specification_name'], force_refetch=True)
    if (dataset.id != binding['plan']['dataset_id'] or dataset.name != draft['dataset_name']
            or getattr(dataset, '_client', None) is not client
            or record.id != slot['record_id'] or record.molecule_id != slot['molecule_id']):
        raise ValueError('registered dataset/record identity changed')
    # Refreshed object must also still be waiting, not only the earlier snapshot.
    if (record.status.value != 'waiting' or record.compute_history
            or record.task is None or record.task.compute_tag != slot['compute_tag']):
        raise ValueError('own registered wave record must still be waiting')
    created = dict(dataset_id=dataset.id, record_id=record.id, molecule_id=record.molecule_id,
        compute_tag=slot['compute_tag'], n_inserted=1, n_existing=0,
        dispatch_intent_sha256=binding['registration']['dispatch_intent_sha256'])
    save('record-created.json', created)
    return dict(dataset=dataset, record=record, created=created)


def _completed(snapshot, binding, draft, *, record_id, manager, envelope, tensor_raw):
    slot = _slot(binding, draft)
    if (record_id != slot['record_id'] or type(tensor_raw) is not bytes
            or len(tensor_raw) > 16*1024**2):
        raise ValueError('exact own wave record and bounded tensor bytes required')
    # require_complete=True is the finalizer's whole-wave gate. At own delivery
    # peers may legitimately still run; own completion is checked below.
    preservation = _validate_capture(binding, snapshot, mid=draft['monomer_id'])
    finished = snapshot['records'][str(record_id)]
    effective = helper('audited_bank').normalize_specification(draft['specification'])
    effective['keywords'].update(draft['entry']['additional_keywords'])
    artifact = envelope['artifacts']['tensors.npz']
    sha = _sha(tensor_raw)
    if (finished.get('status') != 'complete' or len(finished.get('history', [])) != 1
            or finished['history'][0].get('manager_name') != manager
            or finished.get('specification') != effective
            or finished.get('native_envelope_sha256') != helper('dataset_guard').digest(envelope)
            or finished.get('tensor_sha256') != sha or artifact['sha256'] != sha
            or artifact['size'] != len(tensor_raw)):
        raise ValueError('own wave completion/history/envelope/tensor differs')
    return preservation


def capture_delivery(client, binding, draft, *, record_id, manager, envelope, tensor_raw, save):
    snapshot = _snapshot(client, binding)
    preservation = _completed(snapshot, binding, draft, record_id=record_id,
        manager=manager, envelope=envelope, tensor_raw=tensor_raw)
    save('owned-after-delivery.json', snapshot)
    return preservation


def audit_preservation(data, outcome, draft, binding, specification):
    helper('bank_wave_campaign').revalidate_binding(binding)
    barrier = binding['science_barrier']
    if barrier is None:
        raise ValueError('whole-science barrier required before member audit')
    slot = _slot(binding, draft)
    fields = started_fields(binding)
    if (any(outcome.get(key) != value for key, value in fields.items())
            or outcome.get('dataset_id') != binding['plan']['dataset_id']
            or outcome.get('monomer_id') != draft['monomer_id']
            or outcome.get('campaign') != draft['campaign']
            or outcome.get('producer_profile') != 'grid400'
            or outcome.get('dataset_mode') != 'append'
            or outcome.get('record_id') != slot['record_id']
            or outcome.get('compute_tag') != slot['compute_tag']
            or outcome.get('allocation', {}).get('JobId') != slot['science_job_id']
            or outcome.get('worker_preflight_sha256') != slot['worker_preflight_sha256']
            or outcome.get('pilot_sha256') != slot['case_sha256']
            or not _same(specification,
                         helper('audited_bank').normalize_specification(draft['specification']))):
        raise ValueError('wave outcome/draft/specification binding differs')
    delivered = _read(data/'owned-after-delivery.json')
    evidence = barrier['members'][draft['monomer_id']]
    if (evidence['delivery_snapshot_sha256'] != helper('dataset_guard').digest(delivered)
            or evidence['outcome_sha256'] != _sha(_raw(data/'pilot-outcome.json'))):
        raise ValueError('wave barrier differs from actual saved delivery')
    envelope = _read(data/'server-native-envelope.json')
    tensor = _raw(data/'server-tensors.npz', 16*1024**2)
    summary = _completed(delivered, binding, draft, record_id=outcome['record_id'],
        manager=outcome['manager'], envelope=envelope, tensor_raw=tensor)
    if outcome.get('tensors_sha256') != _sha(tensor) or outcome.get('tensor_bytes') != len(tensor):
        raise ValueError('wave outcome tensor binding differs')
    return dict(fields, status='PASS_CAPTURED_WAVE_DATASET_PRESERVATION',
        before_sha256=helper('dataset_guard').digest(binding['registered_snapshot']),
        after_sha256=helper('dataset_guard').digest(delivered),
        audited_native_envelope_sha256=helper('dataset_guard').digest(envelope),
        audited_tensor_sha256=_sha(tensor),
        preserved_records=summary['preserved_records'], peer_statuses=summary['peer_statuses'])
