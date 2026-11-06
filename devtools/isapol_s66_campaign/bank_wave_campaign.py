"""Closed measured-input reconstruction and bounded wave metadata loading."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import importlib.util
import sys

_MODULES = {}


def helper(name):
    if name not in ('bank_campaign', 'bank_wave', 'bank_store', 'dataset_guard'):
        raise ValueError('closed wave campaign helper required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_wave_campaign_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


WAVE_ROOT = Path('/storage/project/r-cs207-0/awallace43/runs/'
                 'ccasp-native-s66-grid400-1aaa9e0a30-waves002')
CONTEXT_SHA256 = 'a92028bc139edcffe5c6f03cc8dd1945c30183149a235b94852882dc2678fec9'
ANCHOR_SHA256 = 'd4ace10ad0823800f2b0a678feaa88539a30907c8a6a869e9363f7a91ff7ced8'
ADDITIONAL_SOURCES = ('bank_wave.py', 'bank_wave_campaign.py', 'bank_wave_transport.py',
                      'bank_wave_controller.py', 'bank_wave_ops.sbatch', 'native_controller.py')


def pins():
    return dict(helper('bank_campaign').pins(), campaign_scope='eligible-wave',
                dispatch_protocol='registered-wave/1')


def source_names():
    return tuple(dict.fromkeys(helper('bank_campaign').ADDITIONAL_SOURCES+ADDITIONAL_SOURCES))


def _raw(path):
    return helper('bank_store')._read_regular(path, maximum=64*1024**2)


def _closed(path, expected):
    if (not isinstance(path, Path) or not path.is_absolute() or path != expected
            or path.resolve() != expected):
        raise ValueError('exact closed wave metadata path required')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('wave metadata aliases are forbidden')
    return path


def _load_predecessor(plan):
    core, campaign, store = helper('bank_wave'), helper('bank_campaign'), helper('bank_store')
    context_path = Path(plan['bootstrap_context_path'])
    if (context_path.name != 'bank-context.json'
            or not context_path.is_relative_to(campaign.CAMPAIGN_ROOT)):
        raise ValueError('unchanged original bootstrap context path required')
    _closed(context_path, context_path)
    if plan['bootstrap_context_sha256'] != CONTEXT_SHA256:
        raise ValueError('original pinned bootstrap context required')
    context_raw = _raw(context_path)
    context = core.decoded(context_raw, CONTEXT_SHA256)
    previous_path = Path(plan['predecessor_path'])
    anchor_path = campaign.CAMPAIGN_ROOT/(
        'ccasp-native-s66-grid400-monomer-0025-audit/payload/bank-checkpoint.json')
    if plan['predecessor_sha256'] == ANCHOR_SHA256:
        _closed(previous_path, anchor_path)
    else:
        _closed(previous_path, WAVE_ROOT/(
            'wave-%04d-%04d' % (max(25, plan['generation_before']-8), plan['generation_before'])
            )/'finalizer/payload/bank-checkpoint.json')
    previous_raw = _raw(previous_path)
    previous = core.decoded(previous_raw, plan['predecessor_sha256'])
    if previous.get('schema') == 'native-audited-bank-checkpoint/1':
        if previous.get('generation') != 25 or core.digest(previous_raw) != ANCHOR_SHA256:
            raise ValueError('positively verified generation25 serial anchor required')
        required = core._serial_required(previous, context)
        proofs = store.read_proofs(required)
    else:
        if previous.get('serial_anchor_sha256') != ANCHOR_SHA256:
            raise ValueError('unchanged actual serial anchor required')
        proofs = store._read_wave_proofs(set(previous['proof_sha256']))
    core.validate_checkpoint(previous_raw, plan['predecessor_sha256'],
                             context_raw, CONTEXT_SHA256, proofs)
    return context_raw, previous_raw, proofs


def _base_binding(args):
    core, campaign = helper('bank_wave'), helper('bank_campaign')
    if (getattr(args, 'producer', None) != 'grid400'
            or getattr(args, 'campaign_scope', None) != 'eligible-wave'
            or getattr(args, 'remaining_case', None) is not True):
        raise ValueError('explicit eligible-wave grid400 append mode required')
    forbidden = ('bank_context', 'bank_context_sha256', 'bank_checkpoint', 'bank_checkpoint_sha256',
                 'owned_checkpoint', 'checkpoint_sha256', 'checkpoint_proofs',
                 'accepted_audit', 'accepted_outcome', 'accepted_pilot', 'accepted_envelope')
    if any(getattr(args, key, None) is not None for key in forbidden):
        raise ValueError('wave plan is the sole predecessor authority')
    campaign.authority()
    plan_path = getattr(args, 'wave_plan', None)
    if not isinstance(plan_path, Path):
        raise ValueError('recorded wave plan path required')
    plan_raw = _raw(plan_path)
    plan_sha = core.sha(args.wave_plan_sha256)
    plan = core.decoded(plan_raw, plan_sha)
    _closed(plan_path, WAVE_ROOT/plan['wave_id']/'plan.json')
    context_raw, previous_raw, proofs = _load_predecessor(plan)
    pool, inputs = campaign.load_pool(args.inputs, args.build/'build-identity.json',
                                      args.qualification, args.descriptors)
    prepared = campaign.prepared_cases(pool, inputs)
    core.validate_plan(plan_raw, plan_sha, prepared_cases=prepared,
        checkpoint_raw=previous_raw, checkpoint_sha256=plan['predecessor_sha256'],
        context_raw=context_raw, context_sha256=plan['bootstrap_context_sha256'], proof_blobs=proofs)
    cases = {mid: core.bind_member(prepared[mid], plan=plan, monomer_id=mid)
             for mid in plan['member_ids']}
    for mid, case in cases.items():
        path = _closed(Path(plan['members'][mid]['case_path']), plan_path.parent/'cases'/(mid+'.json'))
        raw = _raw(path)
        if (core.digest(raw) != plan['members'][mid]['case_sha256']
                or core.encode(core.decoded(raw)) != core.encode(case)):
            raise ValueError('actual wave case bytes differ')
    mid = args.pilot.stem
    if mid not in cases:
        raise ValueError('undeclared wave case path')
    _closed(args.pilot, plan_path.parent/'cases'/(mid+'.json'))
    return dict(plan_path=plan_path, plan_raw=plan_raw, plan_sha256=plan_sha, plan=plan,
        context_raw=context_raw, context_sha256=plan['bootstrap_context_sha256'],
        checkpoint_raw=previous_raw, checkpoint_sha256=plan['predecessor_sha256'],
        proof_blobs=proofs, monomer_id=mid), cases


def verify_prepared_from_paths(args, manifest_path):
    if manifest_path != args.build/'build-identity.json':
        raise ValueError('actual build manifest path required')
    binding, cases = _base_binding(args)
    return cases[binding['monomer_id']]


def load_binding(args, *, require_registration=False):
    binding, cases = _base_binding(args)
    core, base = helper('bank_wave'), helper('dataset_guard')
    none_fields = ('registration_path', 'registration_raw', 'registration_sha256', 'registration',
        'release_permit_path', 'release_permit_raw', 'release_permit_sha256', 'release_permit',
        'captures', 'registered_snapshot', 'science_barrier_path', 'science_barrier_raw',
        'science_barrier_sha256', 'science_barrier')
    binding.update({key: None for key in none_fields})
    if not require_registration:
        return binding
    directory, plan = binding['plan_path'].parent, binding['plan']
    reg_path = _closed(getattr(args, 'wave_registration', None), directory/'registration.json')
    permit_path = _closed(directory/'release-permit.json', directory/'release-permit.json')
    permit_raw, reg_raw = _raw(permit_path), _raw(reg_path)
    permit = core.decoded(permit_raw)
    reg_sha = permit['registration_sha256']
    reg = core.decoded(reg_raw, reg_sha)
    explicit = getattr(args, 'wave_registration_sha256', None)
    if explicit is not None and explicit != reg_sha:
        raise ValueError('explicit audit registration hash differs from permit')
    captures = {}
    for mid in plan['member_ids']:
        captures[mid] = {}
        for role, filename in (('before', 'owned-before-entry.json'),
                               ('entry', 'owned-after-entry.json'),
                               ('record', 'owned-after-record.json')):
            path = _closed(Path(reg['members'][mid]['captures'][role]['path']),
                           directory/'registration'/mid/filename)
            captures[mid][role] = core.decoded(_raw(path))
    registered_path = _closed(Path(reg['registered_snapshot_path']),
                              directory/'registration/registered-snapshot.json')
    registered = core.decoded(_raw(registered_path))
    previous = core.decoded(binding['checkpoint_raw'])['snapshot']
    core.validate_registration(plan, reg_raw, reg_sha, predecessor_snapshot=previous,
                               cases=cases, captures=captures, registered_snapshot=registered)
    core.validate_release_permit(plan, binding['plan_sha256'], reg, reg_sha, permit_raw)
    intent = core.decoded(_raw(directory/'registration/dispatch-intent.json'))
    fields = intent_fields(binding, reg['members'], reg['registration_role'])
    intent_sha = helper('bank_store')._verify_wave_intent(intent, **fields)
    if intent_sha != reg['dispatch_intent_sha256']:
        raise ValueError('registered durable wave intent differs')
    binding.update(registration_path=reg_path, registration_raw=reg_raw,
        registration_sha256=reg_sha, registration=reg,
        release_permit_path=permit_path, release_permit_raw=permit_raw,
        release_permit_sha256=core.digest(permit_raw), release_permit=permit,
        captures=captures, registered_snapshot=registered)
    barrier_path = getattr(args, 'wave_science_barrier', None)
    barrier_sha = getattr(args, 'wave_science_barrier_sha256', None)
    if getattr(args, 'case', None) is not None and (barrier_path is None or barrier_sha is None):
        raise ValueError('audit requires all-SCI-positive barrier')
    if barrier_path is not None or barrier_sha is not None:
        _closed(barrier_path, directory/'science-barrier.json')
        barrier_raw = _raw(barrier_path)
        barrier = core.decoded(barrier_raw, barrier_sha)
        outcomes, accounting, deliveries = {}, {}, {}
        for mid in plan['member_ids']:
            member = barrier['members'][mid]
            run = directory/'members'/mid/'science'
            outcome_path = _closed(Path(member['outcome_path']), run/'payload/pilot-outcome.json')
            accounting_path = _closed(Path(member['accounting_path']), run/'accounting.txt')
            outcomes[mid], accounting[mid] = _raw(outcome_path), _raw(accounting_path)
            deliveries[mid] = core.decoded(_raw(run/'payload/owned-after-delivery.json'))
            core.validate_capture(plan, reg, predecessor_snapshot=previous,
                registered_snapshot=registered, snapshot=deliveries[mid], member_id=mid)
        core.validate_barrier(plan, reg, permit_raw, barrier_raw, outcomes=outcomes,
                               accounting=accounting, deliveries=deliveries)
        binding.update(science_barrier_path=barrier_path, science_barrier_raw=barrier_raw,
                       science_barrier_sha256=barrier_sha, science_barrier=barrier)
    return binding


def intent_fields(binding, members, registration_role):
    plan = binding['plan']
    return dict(dataset_id=357, checkpoint_sha256=binding['checkpoint_sha256'],
        snapshot_sha256=plan['predecessor_snapshot_sha256'], context_sha256=binding['context_sha256'],
        plan_sha256=binding['plan_sha256'], eligibility_sha256=plan['eligibility_sha256'],
        wave_id=plan['wave_id'], registration_role=registration_role,
        members={mid: {key: members[mid][key] for key in (
            'science_job_id', 'compute_tag', 'case_sha256', 'worker_preflight_sha256')}
            for mid in plan['member_ids']})


def revalidate_binding(binding):
    campaign = helper('bank_campaign')
    args = NS(producer='grid400', campaign_scope='eligible-wave', remaining_case=True,
        wave_plan=binding['plan_path'], wave_plan_sha256=binding['plan_sha256'],
        pilot=binding['plan_path'].parent/'cases'/(binding['monomer_id']+'.json'),
        build=campaign.BUILD_IDENTITY.parent, inputs=campaign.INPUTS,
        qualification=campaign.QUALIFICATION, descriptors=campaign.DESCRIPTORS,
        wave_registration=binding['registration_path'],
        wave_registration_sha256=binding['registration_sha256'],
        wave_science_barrier=binding['science_barrier_path'],
        wave_science_barrier_sha256=binding['science_barrier_sha256'])
    current = load_binding(args, require_registration=binding['registration'] is not None)
    # Raw byte comparisons bind everything recursively; JSON views must also
    # remain unmodified in memory, including distinctions such as signed zero.
    def equal(a, b):
        if type(a) is dict and type(b) is dict:
            return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
        if type(a) is list and type(b) is list:
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, (str, int, float, bool)) or a is None:
            return helper('bank_wave').encode(a) == helper('bank_wave').encode(b)
        return type(a) is type(b) and a == b
    if not equal(binding, current):
        raise ValueError('wave binding changed during execution')
