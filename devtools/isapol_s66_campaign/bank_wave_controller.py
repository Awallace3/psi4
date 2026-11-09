"""One bounded held/registered/released wave. Production activation is disabled.

Only new jobs are controlled. Partial or ambiguous operations stop permanently;
there is no retry, clearing, cancellation, scientific promotion or resume mode.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace as NS

PRODUCTION_ENABLED = True  # Coordinator review + a new immutable commit required.
PREFIX = 'CCASP_WAVE_CONTROLLER_RESULT='
SLURM = '/opt/slurm/current/bin/'
_MODULES = {}


def helper(name):
    if name not in ('bank_wave', 'bank_wave_campaign', 'bank_wave_transport', 'bank_store',
                    'bank_campaign', 'native_controller', 'dataset_guard', 'bank_wave_runtime',
                    'phoenix_installed_checks', 'phoenix_pilot_cb63305', 'phoenix_worker_cb63305',
                    'runtime_audit_cb63305', 'descriptor_scratch_cb63305','qomp_wave_policy',
                    'qomp_execution'):
        raise ValueError('closed bounded-wave controller helper required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_wave_controller_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def raw(path):
    return helper('bank_store')._read_regular(Path(path), maximum=64*1024**2)


def read(path):
    return helper('bank_wave').decoded(raw(path))


def write(path, value):
    helper('bank_store')._directory(Path(path).parent, private=True)
    helper('bank_store')._write_once(Path(path), helper('bank_wave').encode(value))


def mkdir(path):
    helper('native_controller').mkdir_once(path)
    helper('bank_store')._directory(Path(path), private=True)


def admit_headroom(observed_used, new_jobs):
    if (type(observed_used) is not int or observed_used < 0
            or type(new_jobs) is not int or not 0 <= new_jobs <= 8
            or observed_used+new_jobs > 45):
        raise ValueError('shared embers cap50 minus five reserved slots exceeded')


def embers_used(text):
    lines = text.splitlines()
    if any(not line.strip() or len(line.split()) != 1 for line in lines):
        raise ValueError('unambiguous all-user-job QoS observation required')
    return sum(line.strip() == 'embers' for line in lines)


def admit_scratch(cases, observation=None,*,producer='grid400',observe=None):
    """Conservative whole-stage sum plus one additional largest-job headroom."""
    if producer=='qomp':
        if observation is not None or not callable(observe):
            raise ValueError('fresh trusted QOMP whole-wave quota callback required')
        return helper('qomp_wave_policy').helper('qomp_bank_policy').admit_scratch(
            [case['execution_profile'] for case in cases.values()],observe)
    if producer!='grid400' or observe is not None:
        raise ValueError('closed wave scratch producer required')
    profiles = [helper('bank_wave').helper('grid400_bank_policy').validate_profile(
        case['execution_profile']) for case in cases.values()]
    if not 1 <= len(profiles) <= 8:
        raise ValueError('one bounded wave scratch admission required')
    guard = helper('descriptor_scratch_cb63305')
    for profile in profiles:
        guard.admit_scratch(**observation, required_bytes=profile['scratch_bytes'],
                            required_files=profile['scratch_files'])
    return guard.admit_scratch(**observation,
        required_bytes=sum(p['scratch_bytes'] for p in profiles)+max(p['scratch_bytes'] for p in profiles),
        required_files=sum(p['scratch_files'] for p in profiles)+max(p['scratch_files'] for p in profiles))


def phase_command(binding, mid, phase):
    """Exact published wrapper argv; no SCI command contains a future hash."""
    plan, directory = binding['plan'], binding['plan_path'].parent
    p=helper('bank_wave').settings(plan)
    if phase not in ('worker', 'science', 'audit', 'registration', 'finalizer'):
        raise ValueError('closed wave phase required')
    member = phase in ('worker', 'science', 'audit')
    if member and mid not in plan['member_ids']:
        raise ValueError('declared wave member required')
    if not member and mid is not None:
        raise ValueError('wave operations are not singleton science jobs')
    source, commit = Path(plan['source']), plan['operational_commit']
    run = directory/'members'/mid/phase if member else directory/phase
    profile = plan['members'][mid]['execution_profile'] if member else None
    cores, memory, duration = (4, 24, '00:45:00')
    if phase in ('worker', 'science'):
        p.profile.validate_profile(profile)
        cores, memory = profile['cores'], profile['job_memory_GiB']
        if phase == 'science':
            duration = profile['time_limit']
    command = [SLURM+'sbatch', '--parsable', '--account=gts-cs207-chemx',
        '--partition=cpu-small', '--qos=embers', '--no-requeue', '--nodes=1', '--ntasks=1',
        '--cpus-per-task='+str(cores), '--mem='+str(memory)+'G', '--time='+duration,
        '--job-name=ccasp-wave-'+(mid[-4:]+'-' if member else '')+phase,
        '--output='+str(run/'logs/job-%j.out'), '--error='+str(run/'logs/job-%j.err')]
    if phase == 'science':
        command.append('--hold')
    if p.producer=='qomp':
        if source!=helper('qomp_wave_policy').SOURCE:
            raise ValueError('closed new QOMP wave source required')
        command.insert(2,'--export=USER,LOGNAME,HOME,PATH,LANG,LC_ALL')
        if not member:
            return list(map(str,command+[source/'devtools/isapol_s66_campaign/bank_wave_ops.sbatch',
                source,commit,phase,binding['plan_path'],binding['plan_sha256'],run,'qomp',
                plan['adoption_sha256']]))
        worker=directory/'members'/mid/'worker'
        wp,wh='-','-'
        if phase!='worker':
            proof=read(worker/'verified.json')
            wp=worker/'payload/worker-preflight.json'
            wh=helper('bank_wave').digest(raw(wp))
            if (proof.get('phase')!='worker' or proof.get('plan_sha256')!=binding['plan_sha256']
                    or proof.get('worker_preflight_sha256')!=wh):
                raise ValueError('actual positively verified QOMP member worker SHA required')
        values=[source,commit,phase,run,directory/'cases'/(mid+'.json'),wp,wh,
            directory/'members'/mid/'science' if phase=='audit' else '-',
            binding['plan_path'],binding['plan_sha256'],
            directory/'registration.json' if phase!='worker' else '-',
            binding.get('registration_sha256') if phase=='audit' else '-',
            directory/'science-barrier.json' if phase=='audit' else '-',
            binding.get('science_barrier_sha256') if phase=='audit' else '-']
        if any(value is None for value in values):
            raise ValueError('whole-SCI barrier precedes QOMP audit command')
        return list(map(str,command+[source/'devtools/isapol_s66_campaign/qomp148.sbatch',*values]))
    scripts = dict(worker='phoenix_worker_cb63305.sbatch', science='phoenix_pilot_cb63305.sbatch',
                   audit='runtime_audit_cb63305.sbatch')
    if member:
        case_path = directory/'cases'/(mid+'.json')
        command += [source/'devtools/isapol_s66_campaign'/scripts[phase], source, commit, case_path]
        if phase != 'worker':
            command += [directory/'members'/mid/'worker/payload/worker-preflight.json']
        if phase == 'audit':
            command += [directory/'members'/mid/'science']
        command += [run, '--grid400-bank-wave', binding['plan_path'], binding['plan_sha256'],
                    '-' if phase == 'worker' else directory/'registration.json']
        if phase == 'audit':
            if not binding.get('science_barrier_sha256') or not binding.get('registration_sha256'):
                raise ValueError('whole-SCI barrier precedes every audit command')
            command += ['--wave-registration-sha256', binding['registration_sha256'],
                        '--wave-science-barrier', directory/'science-barrier.json',
                        '--wave-science-barrier-sha256', binding['science_barrier_sha256']]
    else:
        command += [source/'devtools/isapol_s66_campaign/bank_wave_ops.sbatch',
                    source, commit, phase, binding['plan_path'], binding['plan_sha256'], run]
    return list(map(str, command))


def verify_held(allocation, case, actual_job):
    # SLURM displays an exact single-node pending request as either 1 or 1-1.
    # Normalize only that literal singleton display, without changing captures.
    if allocation.get('NumNodes') == '1-1':
        allocation = dict(allocation, NumNodes='1')
    qomp=case.get('producer_profile')=='qomp'
    profile=(helper('qomp_wave_policy').helper('qomp_bank_policy') if qomp
             else helper('bank_wave').helper('grid400_bank_policy')).validate_profile(case['execution_profile'])
    if qomp:
        policy=helper('qomp_wave_policy')
        if (actual_job in policy.PROTECTED_JOBS or case.get('campaign_scope')!='qomp-wave'
                or allocation.get('Command')!=str(policy.SOURCE/'devtools/isapol_s66_campaign/qomp148.sbatch')):
            raise ValueError('new own-source QOMP held job required; historical jobs forbidden')
    expected = dict(JobId=actual_job, JobState='PENDING', Reason='JobHeldUser',
        Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small', NumNodes='1',
        NumCPUs=str(profile['cores']), MinMemoryNode=str(profile['job_memory_GiB'])+'G',
        TimeLimit=profile['time_limit'], Requeue='0', Priority='0')
    if (any(allocation.get(key) != value for key, value in expected.items())
            or allocation.get('UserId', '').split('(')[0] != 'awallace43'):
        raise ValueError('actual new SCI job must remain held on the exact finite route')


def register_once(client, scope, *, save,before_publish=None):
    """One writer; actual sequential captures, no manager and no release."""
    transport, core, store = helper('bank_wave_transport'), helper('bank_wave'), helper('bank_store')
    transport.require_registration_client(client, scope)
    binding, plan = scope.binding, scope.binding['plan']
    p=core.settings(plan)
    helper('bank_wave_campaign').revalidate_binding(binding)
    base = helper('dataset_guard')
    previous = core.decoded(binding['checkpoint_raw'])['snapshot']
    snapshot = lambda: transport._snapshot_dataset(client, plan, previous)
    before = snapshot()
    if base.digest(before) != plan['predecessor_snapshot_sha256']:
        raise ValueError('live cutover predecessor differs before registration')
    info = client.get_server_information()
    if info.get('name') != 'CDS QCFractal Server' or info.get('version') != '0.70':
        raise ValueError('approved deployment changed')
    managers = list(client.query_managers(status=['active'], limit=200))
    tags = {member['compute_tag'] for member in scope.members.values()}
    if len(managers) == 200 or any('*' in m.compute_tags or tags.intersection(m.compute_tags)
                                   for m in managers):
        raise ValueError('held wave tags are not isolated from active managers')
    if p.producer=='qomp':
        helper('qomp_execution').manager_isolation(managers,next(iter(tags)))
        helper('qomp_execution').live_originals(client)
    intent = store._reserve_wave_intent(**helper('bank_wave_campaign').intent_fields(
        binding, scope.members, scope.registration_role))
    save('registration/dispatch-intent.json', intent)
    from qcportal.singlepoint import SinglepointDatasetNewEntry
    from qcportal.dataset_models import DatasetSubmitBody
    from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata
    members, captures = {}, {}
    for mid in plan['member_ids']:
        scope.select(mid)
        case, slot = scope.cases[mid], scope.members[mid]
        name, spec = case['entry']['name'], case['specification_name']
        directory = binding['plan_path'].parent/'registration'/mid
        filenames = dict(before='owned-before-entry.json', entry='owned-after-entry.json',
                         record='owned-after-record.json')
        views = {'before': before}
        save('registration/'+mid+'/'+filenames['before'], before)
        save('registration/'+mid+'/entry-insert-attempt.json',
             dict(plan_sha256=binding['plan_sha256'], case_sha256=slot['case_sha256'],
                  entry_name=name, transport_retries=False))
        if base.digest(snapshot()) != base.digest(before):
            raise ValueError('live state changed immediately before wave insertion')
        meta = client.make_request('post', f'api/v1/datasets/singlepoint/{p.dataset}/entries/bulkCreate',
            InsertMetadata, body=[SinglepointDatasetNewEntry(**case['entry'])], allow_retries=False)
        if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
            raise RuntimeError('wave entry was not exactly 1new/0existing; no retry')
        views['entry'] = snapshot()
        base.assert_append_delta(before, views['entry'], entry_name=name, specification_name=spec)
        base.validate_stored_entry(views['entry']['entries'][name], case['entry'])
        save('registration/'+mid+'/'+filenames['entry'], views['entry'])
        save('registration/'+mid+'/record-submit-attempt.json',
             dict(plan_sha256=binding['plan_sha256'], entry_name=name, specification_name=spec,
                  compute_tag=slot['compute_tag'], find_existing=False, transport_retries=False))
        if base.digest(snapshot()) != base.digest(views['entry']):
            raise ValueError('live state changed immediately before wave submit')
        meta = client.make_request('post', f'api/v1/datasets/singlepoint/{p.dataset}/submit',
            InsertCountsMetadata, body=DatasetSubmitBody(entry_names=[name],
                specification_names=[spec], compute_tag=slot['compute_tag'], find_existing=False),
            allow_retries=False)
        if not meta.success or meta.n_inserted != 1 or meta.n_existing != 0:
            raise RuntimeError('wave record was not exactly 1new/0existing; no retry')
        views['record'] = snapshot()
        identifier = core.positive(views['record']['associations'][name][spec])
        base.assert_append_delta(before, views['record'], entry_name=name, record_id=identifier,
                                 specification_name=spec)
        fresh = views['record']['records'][str(identifier)]
        base.validate_fresh_record(fresh, tag=slot['compute_tag'],
            specification=core.helper('audited_bank').normalize_specification(case['specification']),
            entry=case['entry'])
        save('registration/'+mid+'/'+filenames['record'], views['record'])
        members[mid] = dict(slot, entry_name=name, specification_name=spec,
            record_id=identifier, molecule_id=fresh['molecule_id'],
            captures={role: dict(path=str(directory/filename), snapshot_sha256=base.digest(views[role]))
                      for role, filename in filenames.items()})
        captures[mid], before = views, views['record']
        scope.finish(mid)
    registered = snapshot()
    if base.digest(registered) != base.digest(before):
        raise ValueError('all slots must remain waiting before registration seal')
    save('registration/registered-snapshot.json', registered)
    reg = dict(schema='native-eligible-wave-registration/1', campaign_scope=p.scope,
        dispatch_protocol=core.PROTOCOL, wave_id=plan['wave_id'], plan_sha256=binding['plan_sha256'],
        dataset_id=p.dataset, bootstrap_context_sha256=binding['context_sha256'],
        predecessor_sha256=binding['checkpoint_sha256'],
        predecessor_snapshot_sha256=plan['predecessor_snapshot_sha256'],
        registration_role=scope.registration_role, dispatch_intent_sha256=intent['sha256'],
        member_ids=plan['member_ids'], members=members,
        registered_snapshot_path=str(binding['plan_path'].parent/'registration/registered-snapshot.json'),
        registered_snapshot_sha256=base.digest(registered))
    reg_raw = core.encode(reg)
    core.validate_registration(plan, reg_raw, core.digest(reg_raw), predecessor_snapshot=previous,
        cases=scope.cases, captures=captures, registered_snapshot=registered)
    if p.producer=='qomp' and before_publish is None:
        raise ValueError('QOMP registration postseal required before publication')
    if before_publish is not None:
        before_publish()
    save('registration.json', reg)
    return reg


def _registration(binding):
    core, campaign = helper('bank_wave'), helper('bank_wave_campaign')
    directory, plan = binding['plan_path'].parent, binding['plan']
    reg_raw = raw(directory/'registration.json')
    reg = core.decoded(reg_raw)
    cases, captures = {}, {}
    for mid in plan['member_ids']:
        cases[mid] = read(directory/'cases'/(mid+'.json'))
        captures[mid] = {}
        for role, name in (('before', 'owned-before-entry.json'),
                           ('entry', 'owned-after-entry.json'), ('record', 'owned-after-record.json')):
            path = campaign._closed(Path(reg['members'][mid]['captures'][role]['path']),
                                    directory/'registration'/mid/name)
            captures[mid][role] = read(path)
    campaign._closed(Path(reg['registered_snapshot_path']), directory/'registration/registered-snapshot.json')
    registered = read(Path(reg['registered_snapshot_path']))
    core.validate_registration(plan, reg_raw, core.digest(reg_raw),
        predecessor_snapshot=read_checkpoint(binding)['snapshot'], cases=cases,
        captures=captures, registered_snapshot=registered)
    if reg['plan_sha256'] != binding['plan_sha256']:
        raise ValueError('registration actual plan hash differs')
    saved = read(directory/'registration/dispatch-intent.json')
    intent_sha = helper('bank_store')._verify_wave_intent(saved,
        **campaign.intent_fields(binding, reg['members'], reg['registration_role']))
    if intent_sha != reg['dispatch_intent_sha256']:
        raise ValueError('actual wave reservation differs')
    return reg_raw, reg, captures, registered


def read_checkpoint(binding):
    return helper('bank_wave').decoded(binding['checkpoint_raw'])


def publish_wave(client, binding, *, allocation, save,before_publish=None):
    """Read-only independent finalizer, with proofs durable before publication."""
    core, base = helper('bank_wave'), helper('dataset_guard')
    helper('native_controller').admit_audit_allocation(allocation, allocation['JobId'])
    helper('bank_wave_campaign').revalidate_binding(binding)
    helper('bank_wave_transport').require_readonly_client(
        client, binding, monomer_id=binding['monomer_id'])
    if binding['science_barrier'] is None:
        raise ValueError('finalizer requires complete all-SCI barrier')
    directory, plan = binding['plan_path'].parent, binding['plan']
    p=core.settings(plan)
    blobs = dict(binding['proof_blobs'])
    def add(value):
        key = core.digest(value)
        blobs[key] = value
        return key
    link = dict(predecessor_sha256=add(binding['checkpoint_raw']),
        plan_sha256=add(binding['plan_raw']), registration_sha256=add(binding['registration_raw']),
        permit_sha256=add(binding['release_permit_raw']), barrier_sha256=add(binding['science_barrier_raw']),
        registered_snapshot_sha256=add(raw(directory/'registration/registered-snapshot.json')),
        receipts_sha256={}, captures_sha256={}, deliveries_sha256={},
        science_accounting_sha256={}, audit_accounting_sha256={})
    for mid in plan['member_ids']:
        run, audit_run = directory/'members'/mid/'science', directory/'members'/mid/'audit'
        report, submission = read(audit_run/'payload/audit-report.json'), read(audit_run/'submission.json')
        if report['allocation']['JobId'] != submission['job_id']:
            raise ValueError('member audit must match actual independent submitted job')
        if submission['command'] != phase_command(binding, mid, 'audit'):
            raise ValueError('member audit command/source/barrier changed')
        core.completed_accounting(raw(audit_run/'accounting.txt'), submission['job_id'])
        link['receipts_sha256'][mid] = {
            'case': add(raw(directory/'cases'/(mid+'.json'))),
            'audit': add(raw(audit_run/'payload/audit-report.json')),
            'outcome': add(raw(run/'payload'/('qomp-outcome.json' if p.producer=='qomp' else 'pilot-outcome.json'))),
            'envelope': add(raw(run/'payload/server-native-envelope.json'))}
        link['captures_sha256'][mid] = {role: add(raw(
            binding['registration']['members'][mid]['captures'][role]['path']))
            for role in ('before', 'entry', 'record')}
        link['deliveries_sha256'][mid] = add(raw(run/'payload/owned-after-delivery.json'))
        link['science_accounting_sha256'][mid] = add(raw(run/'accounting.txt'))
        link['audit_accounting_sha256'][mid] = add(raw(audit_run/'accounting.txt'))
    snapshot = lambda: helper('bank_wave_transport').snapshot_dataset(client, binding=binding)
    final = snapshot()
    save('final-snapshot.json', final)
    link['final_snapshot_sha256'] = add(core.encode(final))
    checkpoint = core.reduce_wave(binding['checkpoint_raw'], binding['checkpoint_sha256'],
        binding['context_raw'], binding['context_sha256'], blobs, link=link)
    if base.digest(snapshot()) != checkpoint['snapshot_sha256']:
        raise ValueError('double-live wave finalization disagreement')
    helper('bank_wave_campaign').revalidate_binding(binding)
    if p.producer=='qomp' and before_publish is None:
        raise ValueError('QOMP finalizer postseal required before accepted publication')
    if before_publish is not None:
        before_publish()
    # Store even metadata/accounting as content-addressed bytes, never fake
    # serial receipt roles. The original proof namespace is shared.
    helper('bank_store')._store_wave_proofs({key: blobs[key] for key in checkpoint['proof_sha256']},
        **(dict(producer='qomp') if p.producer=='qomp' else {}))
    raw_checkpoint = core.encode(checkpoint)
    core.validate_checkpoint(raw_checkpoint, core.digest(raw_checkpoint), binding['context_raw'],
                             binding['context_sha256'], blobs)
    save('bank-checkpoint.json', checkpoint)
    result = dict(status='PASS_AUDITED_BANK_WAVE_ADVANCE', allocation=allocation,
        wave_id=plan['wave_id'], plan_sha256=binding['plan_sha256'],
        registration_sha256=binding['registration_sha256'],
        science_barrier_sha256=binding['science_barrier_sha256'], dataset_id=p.dataset,
        generation=checkpoint['generation'], context_sha256=binding['context_sha256'],
        checkpoint_path=str(directory/'finalizer/payload/bank-checkpoint.json'),
        checkpoint_sha256=core.digest(raw_checkpoint), audited_records=checkpoint['audited_records'],
        qualification='Ordinary-OS metadata/audit/live agreement only; no scientific accuracy claim.')
    save('wave-advance.json', result)
    return result


def run_wave(remote, wait):
    """Finite barriers only. remote callbacks are injected in metadata tests."""
    selected = remote('initialize', {})
    state = dict(wave_plan=selected['wave_plan'], wave_plan_sha256=selected['wave_plan_sha256'])
    for phase in ('worker', 'science', 'registration'):
        result = remote('submit', dict(state, phase=phase))
        for actual_job in result['jobs'].values():
            wait(actual_job)
        # Held science jobs cannot be waited for completion before release.
        # Production submit(science) returns an empty wait map (held IDs are
        # separately durably recorded and verified).
        remote('verify', dict(state, phase=phase))
    remote('release', state)
    jobs = remote('science-jobs', state)['jobs']
    for actual_job in jobs.values():
        wait(actual_job)
    remote('verify', dict(state, phase='delivery'))
    remote('barrier', state)
    for phase in ('audit', 'finalizer'):
        result = remote('submit', dict(state, phase=phase))
        for actual_job in result['jobs'].values():
            wait(actual_job)
        remote('verify', dict(state, phase=phase))
    return remote('accept', state)


class Production:
    """Closed production adapters; never constructed by local unit tests."""
    qomp=False
    def __init__(self, args):
        self.args = args
        if getattr(args,'producer','grid400') not in ('grid400','qomp'):
            raise ValueError('closed bounded-wave producer required')
        self.core = helper('bank_wave')
        self.campaign = helper('bank_wave_campaign')
        self.qomp=getattr(args,'producer','grid400')=='qomp'
        self.policy=helper('qomp_wave_policy') if self.qomp else None
        self.bank=(self.policy.helper('qomp_expansion').helper('qomp_campaign')
                   if self.qomp else helper('bank_campaign'))

    def validate_source(self):
        if self.qomp:
            self.policy.validate_source(self.args.source,self.args.commit)

    def require_enabled(self):
        if self.qomp and not self.policy.PRODUCTION_ENABLED:
            raise RuntimeError('QOMP wave activation disabled pending actual failure-lifecycle evidence/review')

    def scratch_admission(self,binding):
        if not self.qomp:
            return admit_scratch({mid:read(binding['plan_path'].parent/'cases'/(mid+'.json'))
                                  for mid in binding['plan']['member_ids']},self.scratch_observation())
        cases={mid:read(binding['plan_path'].parent/'cases'/(mid+'.json'))
               for mid in binding['plan']['member_ids']}
        return admit_scratch(cases,producer='qomp',
            observe=lambda:helper('qomp_execution').observe(self.policy.SCRATCH_ROOT))

    def queue(self):
        return embers_used(subprocess.check_output(
            [SLURM+'squeue', '-u', 'awallace43', '-r', '-h', '-o', '%q'], text=True))

    def binding(self, *, registered=False, audit=False):
        plan = read(self.args.wave_plan)
        if (plan['source'] != str(self.args.source)
                or plan['operational_commit'] != self.args.commit
                or self.qomp and plan.get('adoption_sha256')!=self.args.adoption_sha256):
            raise ValueError('wave plan must bind this actual operational checkout/commit')
        args = NS(**dict(vars(self.args),producer='qomp' if self.qomp else 'grid400',
                  campaign_scope='qomp-wave' if self.qomp else 'eligible-wave',
                  remaining_case=True, pilot=self.args.wave_plan.parent/'cases'/(plan['member_ids'][0]+'.json'),
                  build=self.bank.BUILD if self.qomp else self.bank.BUILD_IDENTITY.parent, inputs=self.bank.INPUTS,
                  qualification=self.bank.QUALIFICATION, descriptors=self.bank.DESCRIPTORS))
        args.wave_registration = self.args.wave_plan.parent/'registration.json'
        if audit:
            args.case = self.args.wave_plan.parent/'members'/plan['member_ids'][0]/'science'
            args.wave_science_barrier = self.args.wave_plan.parent/'science-barrier.json'
            args.wave_science_barrier_sha256 = self.core.digest(raw(args.wave_science_barrier))
        return self.campaign.load_binding(args, require_registration=registered)

    def client(self, binding, *, registration_scope=None):
        # Credentials are only read in this production adapter, never during
        # imports/tests. Never persist the server dictionary or configuration.
        import yaml
        server = yaml.safe_load((self.bank.ROOT/'qca/resources_losii_ac.yml').read_text())['server']
        helper('phoenix_pilot_cb63305').validate_server_config(server)
        transport = helper('bank_wave_transport')
        if registration_scope is not None:
            cls = transport.client_class(registration_scope)
        elif binding['registration'] is not None:
            cls = transport.readonly_client_class(binding, monomer_id=binding['monomer_id'])
        else:
            # Internal read-only cutover client; no serial constructor or fake
            # registered authority. The predecessor was fully validated above.
            def check(client, method, endpoint, kwargs):
                if transport._endpoint(method, endpoint,
                    self.core.settings(binding['plan']).dataset if self.qomp else 357,
                    **(dict(producer='qomp') if self.qomp else {})) != 'read':
                    raise ValueError('cutover client is read-only')
            cls = transport._engine(check)
        client = cls(server['fractal_uri'], username=server['username'], password=server['password'],
                     verify=server['verify'], show_motd=False)
        info = client.get_server_information()
        if info.get('name') != 'CDS QCFractal Server' or info.get('version') != '0.70':
            raise ValueError('approved wave deployment differs')
        return client

    def held(self, binding,*,members=None):
        result = {}
        mids=binding['plan']['member_ids'] if members is None else members
        if (not 1<=len(mids)<=8 or len(set(mids))!=len(mids)
                or not set(mids)<=set(binding['plan']['member_ids'])):
            raise ValueError('declared own held member subset required')
        for mid in mids:
            submission = read(binding['plan_path'].parent/'members'/mid/'science/submission.json')
            if submission['command'] != phase_command(binding, mid, 'science'):
                raise ValueError('held SCI command differs')
            text = subprocess.check_output([SLURM+'scontrol', 'show', 'job', '-o', submission['job_id']],
                                           text=True)
            allocation = dict(item.split('=', 1) for item in text.split() if '=' in item)
            verify_held(allocation, read(binding['plan_path'].parent/'cases'/(mid+'.json')),
                        submission['job_id'])
            result[mid] = dict(science_job_id=submission['job_id'],
                compute_tag=self.core.settings(binding['plan']).namespace+mid+'-'+submission['job_id'],
                case_sha256=binding['plan']['members'][mid]['case_sha256'],
                worker_preflight_sha256=self.core.digest(raw(
                    binding['plan_path'].parent/'members'/mid/'worker/payload/worker-preflight.json')))
        if len({member['science_job_id'] for member in result.values()}) != len(result):
            raise ValueError('duplicate actual held SCI job')
        return result

    def _accounting(self, run, actual_job):
        response = subprocess.check_output(
            [SLURM+'sacct', '-n', '-P', '-j', actual_job, '-o', 'JobIDRaw,State,ExitCode'])
        self.core.completed_accounting(response, actual_job)
        helper('bank_store')._write_once(run/'accounting.txt', response)

    def submit(self, phase):
        self.require_enabled()
        binding = self.binding(registered=phase in ('audit', 'finalizer'),
                               audit=phase in ('audit', 'finalizer'))
        directory, mids = binding['plan_path'].parent, binding['plan']['member_ids']
        if phase == 'science':
            self._require_verified(directory, mids, 'worker')
        elif phase == 'registration':
            self.held(binding)
            if self.queue() < len(mids):
                raise ValueError('all held SCI jobs must appear in shared queue observation')
        elif phase == 'audit':
            if binding['science_barrier'] is None:
                raise ValueError('all SCI positive barrier required')
        elif phase == 'finalizer':
            self._require_verified(directory, mids, 'audit')
        elif phase != 'worker':
            raise ValueError('closed submission phase')
        identifiers = mids if phase in ('worker', 'science', 'audit') else [None]
        admit_headroom(self.queue(), len(identifiers))
        if self.qomp:
            self.scratch_admission(binding)
        jobs = {}
        for mid in identifiers:
            admit_headroom(self.queue(), 1)
            run = directory/'members'/mid/phase if mid else directory/phase
            if phase != 'registration':
                mkdir(run)
            else:
                # Registration capture directory was prepared at initialization.
                if (run/'intent.json').exists() or (run/'submission.json').exists():
                    raise ValueError('existing registration attempt; no resume')
            mkdir(run/'logs')
            command = phase_command(binding, mid, phase)
            subprocess.run(command[:1]+['--test-only']+command[1:],
                           check=True, capture_output=True, text=True)
            write(run/'intent.json', dict(plan_sha256=binding['plan_sha256'], command=command))
            if self.qomp:
                write(run/'pre-submit-quota.json',self.scratch_admission(binding))
                self.validate_source()
            response = subprocess.run(command, capture_output=True, text=True, check=False)
            write(run/'submission-response.json', dict(returncode=response.returncode,
                stdout=response.stdout, stderr=response.stderr))
            actual_job = response.stdout.strip()
            if response.returncode or re.fullmatch('[1-9][0-9]*', actual_job) is None:
                raise RuntimeError('ambiguous new-wave job submission; no replay')
            write(run/'submission.json', dict(plan_sha256=binding['plan_sha256'],
                  command=command, job_id=actual_job))
            jobs[mid or phase] = actual_job
        if phase == 'science':
            self.held(binding)
        return dict(jobs={} if phase == 'science' else jobs, held_jobs=jobs if phase == 'science' else {})

    def _require_verified(self, directory, mids, phase):
        for mid in mids:
            run = directory/'members'/mid/phase
            verified, submission = read(run/'verified.json'), read(run/'submission.json')
            if (verified.get('job_id') != submission['job_id']
                    or verified.get('phase') != phase
                    or verified.get('plan_sha256') != submission.get('plan_sha256')
                    or verified.get('accounting_sha256') != self.core.digest(raw(run/'accounting.txt'))):
                raise ValueError('complete positive phase verification required')
            self.core.completed_accounting(raw(run/'accounting.txt'), submission['job_id'])

    def verify(self, phase):
        binding = self.binding(registered=phase in ('delivery', 'audit', 'finalizer'),
                               audit=phase in ('audit', 'finalizer'))
        directory, mids = binding['plan_path'].parent, binding['plan']['member_ids']
        if phase == 'science':
            self.held(binding)
            return dict(held_verified=True)
        actual_phase = 'science' if phase == 'delivery' else phase
        identifiers = mids if actual_phase in ('worker', 'science', 'audit') else [None]
        for mid in identifiers:
            run = directory/'members'/mid/actual_phase if mid else directory/actual_phase
            submission = read(run/'submission.json')
            if submission['command'] != phase_command(binding, mid, actual_phase):
                raise ValueError('submitted phase command/source changed')
            self._accounting(run, submission['job_id'])
            if phase == 'registration':
                _, reg, _, _ = _registration(binding)
                if reg['registration_role']['job_id'] != submission['job_id']:
                    raise ValueError('registration authority differs from actual ops job')
                self.held(binding)
            elif phase == 'worker':
                report_path = run/'payload/worker-preflight.json'
                report = read(report_path)
                case = read(directory/'cases'/(mid+'.json'))
                worker, science = helper('phoenix_worker_cb63305'), helper('phoenix_pilot_cb63305')
                p=self.core.settings(binding['plan'])
                worker.admit_allocation(report['allocation'], producer=p.producer,
                    campaign_scope=p.scope, execution_profile=case['execution_profile'])
                if report['allocation']['JobId'] != submission['job_id']:
                    raise ValueError('actual worker job differs')
                science.validate_preflight(report, case,
                    manifest_sha=self.core.digest(raw(self.bank.QUALIFICATION if self.qomp else self.bank.BUILD_IDENTITY)),
                    qualification_sha=self.core.digest(raw(self.bank.QUALIFICATION)),
                    pilot_sha=binding['plan']['members'][mid]['case_sha256'],
                    worker_sources=worker.source_hashes(p.producer, remaining_case=True,
                                                      campaign_scope=p.scope),
                    transport=helper('bank_wave_transport').transport_binding(**(
                        dict(producer='qomp') if self.qomp else {})),
                    producer=p.producer, campaign_scope=p.scope, remaining_case=True,
                    **(dict(operational_commit=self.args.commit,wave_binding=binding) if self.qomp else {}))
            elif phase == 'delivery':
                self._delivery(binding, mid, submission['job_id'])
            elif phase == 'audit':
                report = read(run/'payload/audit-report.json')
                helper('native_controller').admit_audit_allocation(report['allocation'], submission['job_id'])
                sci = directory/'members'/mid/'science/payload'
                receipts = dict(audit=raw(run/'payload/audit-report.json'),
                    case=raw(directory/'cases'/(mid+'.json')), outcome=raw(sci/(
                        'qomp-outcome.json' if self.qomp else 'pilot-outcome.json')),
                    envelope=raw(sci/'server-native-envelope.json'))
                delivered = read(sci/'owned-after-delivery.json')
                self.core._member_receipts(binding['plan'], binding['registration'],
                    binding['release_permit'], binding['science_barrier_sha256'], mid,
                    receipts, delivered, delivered)
            elif phase == 'finalizer':
                result = read(run/'payload/wave-advance.json')
                helper('native_controller').admit_audit_allocation(result['allocation'], submission['job_id'])
                self._published(binding, result)
            else:
                raise ValueError('closed verification phase')
            self.validate_source()
            fields=dict(phase=actual_phase, job_id=submission['job_id'],
                plan_sha256=binding['plan_sha256'],
                accounting_sha256=self.core.digest(raw(run/'accounting.txt')))
            if self.qomp and phase=='worker':
                fields['worker_preflight_sha256']=self.core.digest(raw(report_path))
            write(run/'verified.json',fields)
        return dict(verified=phase)

    def _delivery(self, binding, mid, actual_job):
        directory = binding['plan_path'].parent
        data = directory/'members'/mid/'science/payload'
        outcome, case = read(data/('qomp-outcome.json' if self.qomp else 'pilot-outcome.json')), read(directory/'cases'/(mid+'.json'))
        slot = self.core.validate_registered_slot(binding['plan'], binding['registration'],
            binding['release_permit'], case, allocation=outcome['allocation'])
        if (outcome['allocation']['JobId'] != actual_job
                or outcome['operational_commit'] != binding['plan']['operational_commit']
                or outcome.get('plan_sha256') != binding['plan_sha256']
                or outcome.get('registration_sha256') != binding['registration_sha256']
                or outcome.get('release_permit_sha256') != binding['release_permit_sha256']
                or outcome.get('worker_preflight_sha256') != slot['worker_preflight_sha256']
                or outcome.get('pilot_sha256') != slot['case_sha256']
                or outcome.get('qcengine_internal_retries') != 0):
            raise ValueError('actual SCI delivery binding differs')
        helper('runtime_audit_cb63305').validate_delivery(outcome, read(data/'manager-delivery.json'),
            read(data/'record-created.json'), read(data/'claim.json'), read(data/'server-record-outcome.json'),
            owned354=True, producer='qomp' if self.qomp else 'grid400',
            campaign_scope='qomp-wave' if self.qomp else 'eligible-wave')
        self.core.validate_capture(binding['plan'], binding['registration'],
            predecessor_snapshot=read_checkpoint(binding)['snapshot'],
            registered_snapshot=binding['registered_snapshot'],
            snapshot=read(data/'owned-after-delivery.json'), member_id=mid)

    def barrier(self):
        binding = self.binding(registered=True)
        directory, plan = binding['plan_path'].parent, binding['plan']
        self._require_verified(directory, plan['member_ids'], 'science')
        members, outcomes, accounting, deliveries = {}, {}, {}, {}
        for mid in plan['member_ids']:
            run = directory/'members'/mid/'science'
            outcome_path=run/'payload'/('qomp-outcome.json' if self.qomp else 'pilot-outcome.json')
            outcomes[mid], accounting[mid] = raw(outcome_path), raw(run/'accounting.txt')
            deliveries[mid] = read(run/'payload/owned-after-delivery.json')
            members[mid] = {key: binding['registration']['members'][mid][key] for key in (
                'science_job_id', 'record_id', 'compute_tag', 'case_sha256')}
            members[mid].update(outcome_path=str(outcome_path),
                outcome_sha256=self.core.digest(outcomes[mid]), accounting_path=str(run/'accounting.txt'),
                accounting_sha256=self.core.digest(accounting[mid]),
                delivery_snapshot_sha256=helper('dataset_guard').digest(deliveries[mid]))
        barrier = dict(schema='native-eligible-wave-science-barrier/1',
            status='ALL_SCIENCE_POSITIVELY_VERIFIED', wave_id=plan['wave_id'],
            plan_sha256=binding['plan_sha256'], registration_sha256=binding['registration_sha256'],
            release_permit_sha256=binding['release_permit_sha256'], member_ids=plan['member_ids'],
            members=members)
        self.core.validate_barrier(plan, binding['registration'], binding['release_permit_raw'],
            self.core.encode(barrier), outcomes=outcomes, accounting=accounting, deliveries=deliveries)
        write(directory/'science-barrier.json', barrier)
        return dict(barrier_sha256=self.core.digest(self.core.encode(barrier)))

    def release(self):
        self.require_enabled()
        binding = self.binding()
        directory = binding['plan_path'].parent
        verified, submission = read(directory/'registration/verified.json'), read(
            directory/'registration/submission.json')
        if verified['job_id'] != submission['job_id']:
            raise ValueError('positive independent registration verification required')
        self.core.completed_accounting(raw(directory/'registration/accounting.txt'), submission['job_id'])
        reg_raw, reg, _, _ = _registration(binding)
        held = self.held(binding)
        for mid, slot in held.items():
            if any(reg['members'][mid][key] != value for key, value in slot.items()):
                raise ValueError('held identity changed after registration')
        used = self.queue()
        if used < len(held):
            raise ValueError('all held SCI jobs must appear in shared queue observation')
        admit_headroom(used, 0)
        cases = {mid: read(directory/'cases'/(mid+'.json')) for mid in binding['plan']['member_ids']}
        admission = self.scratch_admission(binding) if self.qomp else admit_scratch(cases, self.scratch_observation())
        write(directory/'scratch-release-admission.json', admission)
        # Immediately recheck live registration and all held routes before permit.
        client = self.client(binding)
        live = helper('bank_wave_transport')._snapshot_dataset(
            client, binding['plan'], read_checkpoint(binding)['snapshot'])
        if helper('dataset_guard').digest(live) != reg['registered_snapshot_sha256']:
            raise ValueError('live waiting registration changed before release')
        self.held(binding)
        permit = dict(schema='native-eligible-wave-release-permit/1',
            dispatch_protocol=self.core.PROTOCOL, wave_id=binding['plan']['wave_id'],
            plan_sha256=binding['plan_sha256'], registration_sha256=self.core.digest(reg_raw),
            dataset_id=self.core.settings(binding['plan']).dataset, predecessor_sha256=binding['checkpoint_sha256'],
            dispatch_intent_sha256=reg['dispatch_intent_sha256'], members=self.core.permit_members(reg))
        self.core.validate_release_permit(binding['plan'], binding['plan_sha256'], reg,
                                          self.core.digest(reg_raw), self.core.encode(permit))
        if self.qomp:
            write(directory/'pre-permit-quota.json',self.scratch_admission(binding))
            self.validate_source()
        write(directory/'release-permit.json', permit)
        for mid in binding['plan']['member_ids']:
            run = directory/'members'/mid/'science'
            actual_job = reg['members'][mid]['science_job_id']
            if self.qomp:
                write(run/'pre-release-quota.json',self.scratch_admission(binding))
                self.held(binding,members=(mid,))
                self.validate_source()
            write(run/'release-attempt.json', dict(job_id=actual_job, plan_sha256=binding['plan_sha256'],
                release_permit_sha256=self.core.digest(self.core.encode(permit))))
            response = subprocess.run([SLURM+'scontrol', 'release', actual_job],
                                      check=False, capture_output=True, text=True)
            write(run/'release-response.json', dict(returncode=response.returncode,
                stdout=response.stdout, stderr=response.stderr))
            if response.returncode or response.stdout.strip() or response.stderr.strip():
                raise RuntimeError('failed/ambiguous release; never retry or cancel')
        return dict(released=list(held))

    def scratch_observation(self):
        """Production-only free-space/quota reads; tests inject this observation."""
        scratch = Path('/storage/scratch1/3/awallace43')
        fs = os.statvfs(scratch)
        quotas = {}
        guard = helper('descriptor_scratch_cb63305')
        for role, flag, owner in (('user', '-u', os.getuid()), ('group', '-g', scratch.stat().st_gid)):
            result = subprocess.run(['/usr/bin/lfs', 'quota', flag, str(owner), str(scratch)],
                                    check=True, capture_output=True, text=True, timeout=45)
            if len(result.stdout) > 65536:
                raise ValueError('bounded scratch quota observation required')
            quotas[role+'_quota'] = guard.parse_quota(result.stdout, str(scratch))
        return dict(free_bytes=fs.f_bavail*fs.f_frsize, free_files=fs.f_favail, **quotas)

    def _published(self, binding, result):
        directory = binding['plan_path'].parent
        checkpoint_path = self.campaign._closed(Path(result['checkpoint_path']),
                                               directory/'finalizer/payload/bank-checkpoint.json')
        checkpoint_raw = raw(checkpoint_path)
        checkpoint = self.core.decoded(checkpoint_raw, result['checkpoint_sha256'])
        proofs = helper('bank_store')._read_wave_proofs(set(checkpoint['proof_sha256']),
            **(dict(producer='qomp') if self.qomp else {}))
        self.core.validate_checkpoint(checkpoint_raw, result['checkpoint_sha256'],
                                     binding['context_raw'], binding['context_sha256'], proofs)
        expected_link = dict(plan_sha256=binding['plan_sha256'],
            registration_sha256=binding['registration_sha256'],
            permit_sha256=binding['release_permit_sha256'],
            barrier_sha256=binding['science_barrier_sha256'],
            predecessor_sha256=binding['checkpoint_sha256'])
        if any(checkpoint['wave_link'].get(key) != value for key, value in expected_link.items()):
            raise ValueError('published checkpoint must bind this actual wave plan/registration/permit/barrier')
        if (checkpoint['previous_checkpoint_sha256'] != binding['checkpoint_sha256']
                or checkpoint['generation'] != binding['plan']['generation_after']
                or result['generation'] != checkpoint['generation']
                or result['status'] != 'PASS_AUDITED_BANK_WAVE_ADVANCE'
                or result['plan_sha256'] != binding['plan_sha256']
                or result['registration_sha256'] != binding['registration_sha256']
                or result['science_barrier_sha256'] != binding['science_barrier_sha256']):
            raise ValueError('finalizer whole-wave successor differs')
        return checkpoint

    def accept(self):
        self.require_enabled()
        binding = self.binding(registered=True, audit=True)
        directory = binding['plan_path'].parent
        self._require_verified(directory, binding['plan']['member_ids'], 'audit')
        run = directory/'finalizer'
        submitted, verified = read(run/'submission.json'), read(run/'verified.json')
        if (verified['job_id'] != submitted['job_id']
                or verified['phase'] != 'finalizer'
                or verified['accounting_sha256'] != self.core.digest(raw(run/'accounting.txt'))):
            raise ValueError('positive finalizer verification required before acceptance')
        self.core.completed_accounting(raw(run/'accounting.txt'), submitted['job_id'])
        result = read(run/'payload/wave-advance.json')
        helper('native_controller').admit_audit_allocation(result['allocation'], submitted['job_id'])
        checkpoint = self._published(binding, result)
        live = helper('bank_wave_transport').snapshot_dataset(self.client(binding), binding=binding)
        if helper('dataset_guard').digest(live) != checkpoint['snapshot_sha256']:
            raise ValueError('live disagreement after positive finalizer; no state advancement')
        state = dict(status='ORDINARY_WAVE_AUDIT_ACCEPTED', generation=checkpoint['generation'],
            wave_id=binding['plan']['wave_id'], plan_sha256=binding['plan_sha256'],
            bank_context=binding['plan']['bootstrap_context_path'],
            bank_context_sha256=binding['context_sha256'], bank_checkpoint=result['checkpoint_path'],
            bank_checkpoint_sha256=result['checkpoint_sha256'],
            finalizer_job_id=submitted['job_id'], finalizer_accounting_sha256=verified['accounting_sha256'])
        self.validate_source()
        write(directory/'accepted.json', state)
        return state

    def initialize(self):
        """Only after positive serial cutover or a verified accepted wave."""
        if self.qomp:
            return self._initialize_qomp()
        campaign, core = self.bank, self.core
        state = read(self.args.predecessor_state)
        plan_stub = dict(bootstrap_context_path=state['bank_context'],
            bootstrap_context_sha256=state['bank_context_sha256'],
            predecessor_path=state['bank_checkpoint'],
            predecessor_sha256=state['bank_checkpoint_sha256'], generation_before=state['generation'])
        context_raw, previous_raw, proofs = self.campaign._load_predecessor(plan_stub)
        previous, context = core.decoded(previous_raw), core.decoded(context_raw)
        if state['generation'] == 25:
            expected_state = campaign.CAMPAIGN_ROOT/'states/025.json'
            self.campaign._closed(self.args.predecessor_state, expected_state)
            for mid, phase, actual_job in (('monomer-0025', 'science', '15216900'),
                                            ('monomer-0025', 'audit', '15220617')):
                run = campaign.CAMPAIGN_ROOT/('ccasp-native-s66-grid400-'+mid+'-'+phase)
                verified = read(run/'verified.json')
                if verified['job_id'] != actual_job or verified['phase'] != phase:
                    raise ValueError('original positive serial cutover verification required')
                response = subprocess.check_output([SLURM+'sacct', '-n', '-P', '-j', actual_job,
                                                    '-o', 'JobIDRaw,State,ExitCode'])
                core.completed_accounting(response, actual_job)
                if phase == 'audit':
                    if verified['state'] != state or verified['audit_sha256'] != core.digest(
                            raw(run/'payload/audit-report.json')):
                        raise ValueError('serial accepted state/proof differs')
        else:
            path = Path(state['bank_checkpoint']).parents[2]/'accepted.json'
            self.campaign._closed(self.args.predecessor_state, path)
            run = Path(state['bank_checkpoint']).parents[1]
            verified = read(run/'verified.json')
            if verified['job_id'] != state['finalizer_job_id']:
                raise ValueError('previous wave finalizer verification differs')
            core.completed_accounting(raw(run/'accounting.txt'), state['finalizer_job_id'])
        pool, inputs = campaign.load_pool(campaign.INPUTS, campaign.BUILD_IDENTITY,
                                          campaign.QUALIFICATION, campaign.DESCRIPTORS)
        prepared = campaign.prepared_cases(pool, inputs)
        g, mids = previous['generation'], context['eligible_ids'][previous['generation']:previous['generation']+8]
        if not mids:
            raise ValueError('eligible bank already complete')
        wave_id = 'wave-%04d-%04d' % (g, g+len(mids))
        directory = self.campaign.WAVE_ROOT/wave_id
        plan = dict(plan_stub, schema='native-eligible-wave-plan/1', campaign_scope='eligible-wave',
            dispatch_protocol=core.PROTOCOL, wave_id=wave_id, dataset_id=357, source=str(self.args.source),
            operational_commit=self.args.commit, predecessor_snapshot_sha256=previous['snapshot_sha256'],
            generation_after=g+len(mids), eligible_ids=pool['eligible_ids'], refused_ids=pool['refused_ids'],
            eligibility_sha256=context['eligibility_sha256'], member_ids=mids, members={})
        cases = {mid: core.bind_member(prepared[mid], plan=plan, monomer_id=mid) for mid in mids}
        plan['members'] = {mid: dict(case_path=str(directory/'cases'/(mid+'.json')),
            prepared_case_sha256=core.digest(core.encode(prepared[mid])),
            case_sha256=core.digest(core.encode(cases[mid])),
            execution_profile=cases[mid]['execution_profile']) for mid in mids}
        plan_raw = core.encode(plan)
        core.validate_plan(plan_raw, core.digest(plan_raw), prepared_cases=prepared,
            checkpoint_raw=previous_raw, checkpoint_sha256=plan['predecessor_sha256'],
            context_raw=context_raw, context_sha256=plan['bootstrap_context_sha256'], proof_blobs=proofs)
        # Read live before ANY new filesystem state or job submission.
        temporary = dict(registration=None, checkpoint_raw=previous_raw,plan=plan)
        client = self.client(temporary)
        live = helper('bank_wave_transport')._snapshot_dataset(client, plan, previous['snapshot'])
        if helper('dataset_guard').digest(live) != previous['snapshot_sha256']:
            raise ValueError('live cutover differs from positively verified predecessor')
        root = self.campaign.WAVE_ROOT
        if not root.exists():
            mkdir(root)
        helper('bank_store')._directory(root, private=True)
        mkdir(directory)
        for child in ('cases', 'members', 'registration'):
            mkdir(directory/child)
        for mid in mids:
            write(directory/'cases'/(mid+'.json'), cases[mid])
            mkdir(directory/'members'/mid)
            mkdir(directory/'registration'/mid)
        write(directory/'plan.json', plan)
        write(directory/'cutover.json', dict(predecessor_state=str(self.args.predecessor_state),
            predecessor_state_sha256=core.digest(raw(self.args.predecessor_state)),
            predecessor_sha256=core.digest(previous_raw), live_snapshot_sha256=previous['snapshot_sha256']))
        return dict(wave_plan=str(directory/'plan.json'), wave_plan_sha256=core.digest(plan_raw))

    def _initialize_qomp(self):
        """Explicit actual37aa cutover; original603/25 provenance is not rebased."""
        policy,core=self.policy,self.core
        adopted=policy.load_adoption(self.args.adoption_sha256)
        predecessor=self.args.predecessor_state
        if predecessor==policy.WAVE_ROOT/'inputs/adopted/state.json':
            state=adopted['state']
            previous_path=policy.WAVE_ROOT/'inputs/adopted/bank-checkpoint.json'
        else:
            state=read(predecessor)
            previous_path=Path(state['bank_checkpoint'])
            directory=previous_path.parents[2]
            self.campaign._closed(predecessor,directory/'accepted.json')
            if (state.get('status')!='ORDINARY_WAVE_AUDIT_ACCEPTED'
                    or directory.parent!=policy.WAVE_ROOT):
                raise ValueError('actual own accepted predecessor or copied37aa cutover required')
            run=directory/'finalizer'
            verified=read(run/'verified.json')
            accounting=raw(run/'accounting.txt')
            if (verified.get('job_id')!=state['finalizer_job_id']
                    or verified.get('phase')!='finalizer'
                    or verified.get('accounting_sha256')!=core.digest(accounting)
                    or state.get('finalizer_accounting_sha256')!=core.digest(accounting)):
                raise ValueError('actual positive own finalizer accounting required')
            helper('qomp_execution').helper('qomp_expansion').completed_rows(
                accounting.decode(),state['finalizer_job_id'])
        context_path=policy.WAVE_ROOT/'inputs/adopted/bank-context.json'
        plan_stub=dict(producer_profile='qomp',
            producer_authority=helper('bank_wave').decoded(
                policy.bootstrap_binding()['manifest_raw']),
            adoption_sha256=self.args.adoption_sha256,
            bootstrap_context_path=str(context_path),bootstrap_context_sha256=policy.CONTEXT_SHA256,
            predecessor_path=str(previous_path),predecessor_sha256=state['bank_checkpoint_sha256'],
            generation_before=state['generation'])
        context_raw,previous_raw,proofs=self.campaign._load_predecessor(plan_stub)
        context,previous=map(core.decoded,(context_raw,previous_raw))
        _,prepared,_=policy.prepared()
        width=self.args.member_limit
        if type(width) is not int or not 1<=width<=8:
            raise ValueError('bounded1..8 QOMP wave width required')
        g=previous['generation']
        mids=context['eligible_ids'][g:g+width]
        if not mids:
            raise ValueError('new106 cohort already fully audited')
        wave_id='wave-%04d-%04d'%(g,g+len(mids))
        directory=policy.WAVE_ROOT/wave_id
        plan=dict(plan_stub,schema='native-eligible-wave-plan/1',campaign_scope='qomp-wave',
            dispatch_protocol=core.PROTOCOL,wave_id=wave_id,dataset_id=context['dataset_id'],
            source=str(self.args.source),operational_commit=self.args.commit,
            predecessor_snapshot_sha256=previous['snapshot_sha256'],
            generation_after=g+len(mids),eligible_ids=context['eligible_ids'],refused_ids=[],
            eligibility_sha256=context['eligibility_sha256'],member_limit=width,member_ids=mids,members={})
        cases={mid:core.bind_member(prepared[mid],plan=plan,monomer_id=mid) for mid in mids}
        plan['members']={mid:dict(case_path=str(directory/'cases'/(mid+'.json')),
            prepared_case_sha256=core.digest(core.encode(prepared[mid])),
            case_sha256=core.digest(core.encode(case)),execution_profile=case['execution_profile'])
            for mid,case in cases.items()}
        plan_raw=core.encode(plan)
        core.validate_plan(plan_raw,core.digest(plan_raw),prepared_cases=prepared,
            checkpoint_raw=previous_raw,checkpoint_sha256=plan['predecessor_sha256'],
            context_raw=context_raw,context_sha256=policy.CONTEXT_SHA256,proof_blobs=proofs)
        client=self.client(dict(registration=None,checkpoint_raw=previous_raw,plan=plan))
        q=helper('qomp_execution')
        q.manager_isolation(list(client.query_managers(status=['active'],limit=200)),None)
        q.live_originals(client)
        live=helper('bank_wave_transport')._snapshot_dataset(client,plan,previous['snapshot'])
        if helper('dataset_guard').digest(live)!=previous['snapshot_sha256']:
            raise ValueError('current QOMP dataset differs from independently audited cutover')
        admission=admit_scratch(cases,producer='qomp',observe=lambda:q.observe(policy.SCRATCH_ROOT))
        self.validate_source()
        helper('bank_store')._directory(policy.WAVE_ROOT,private=True)
        mkdir(directory)
        for child in ('cases','members','registration'):
            mkdir(directory/child)
        for mid in mids:
            write(directory/'cases'/(mid+'.json'),cases[mid])
            mkdir(directory/'members'/mid)
            mkdir(directory/'registration'/mid)
        write(directory/'plan.json',plan)
        write(directory/'cutover.json',dict(adoption_sha256=self.args.adoption_sha256,
            producing_serial_operations=policy.SERIAL_OPS,source=self.args.commit,
            predecessor_state=str(predecessor),predecessor_state_sha256=core.digest(raw(predecessor)),
            predecessor_sha256=core.digest(previous_raw),live_snapshot_sha256=previous['snapshot_sha256'],
            scratch_admission=admission))
        return dict(wave_plan=str(directory/'plan.json'),wave_plan_sha256=core.digest(plan_raw))

    def operations(self, phase):
        if self.qomp:
            return self._qomp_operations(phase)
        binding = self.binding(registered=phase == 'finalizer', audit=phase == 'finalizer')
        allocation = helper('phoenix_installed_checks').allocation()
        helper('native_controller').admit_audit_allocation(allocation, allocation['JobId'])
        directory = binding['plan_path'].parent
        expected_run = directory/phase
        self.campaign._closed(self.args.run, expected_run)
        # sbatch may start this job BEFORE returning its ID to the controller.
        # Only the command intent is available at that instant; it was fsynced
        # before submission. Bind actual allocation in produced metadata, then
        # require response-ID equality in controller verify before permit or
        # acceptance. This introduces no additional hold/release permission.
        intent = read(expected_run/'intent.json')
        if (intent.get('plan_sha256') != binding['plan_sha256']
                or intent.get('command') != phase_command(binding, None, phase)
                or allocation['JobId'] != os.environ.get('SLURM_JOB_ID')):
            raise ValueError('actual ops allocation/command differs from prewritten launch intent')
        mkdir(expected_run/'payload')
        if phase == 'registration':
            members = self.held(binding)
            cases = {mid: read(directory/'cases'/(mid+'.json')) for mid in binding['plan']['member_ids']}
            role = dict(kind='allocation', job_id=allocation['JobId'], allocation=allocation)
            scope = helper('bank_wave_transport').WaveScope(binding, cases, members, role)
            return register_once(self.client(binding, registration_scope=scope), scope,
                                 save=lambda name, value: write(directory/name, value))
        if phase == 'finalizer':
            return publish_wave(self.client(binding), binding, allocation=allocation,
                                save=lambda name, value: write(expected_run/'payload'/name, value))
        raise ValueError('closed registration/finalizer operation')

    def _qomp_operations(self,phase):
        self.require_enabled()
        if phase not in ('registration','finalizer'):
            raise ValueError('closed QOMP wave operations role required')
        q=helper('qomp_execution')
        q.helper('phoenix_installed_checks').qomp_engine().qomp_enter()
        allocation=helper('phoenix_installed_checks').allocation()
        helper('native_controller').admit_audit_allocation(allocation,allocation['JobId'])
        self.validate_source()
        binding=self.binding(registered=phase=='finalizer',audit=phase=='finalizer')
        directory=binding['plan_path'].parent
        run=directory/phase
        self.campaign._closed(self.args.run,run)
        intent=read(run/'intent.json')
        if (intent.get('plan_sha256')!=binding['plan_sha256']
                or intent.get('command')!=phase_command(binding,None,phase)
                or allocation['JobId']!=os.environ.get('SLURM_JOB_ID')):
            raise ValueError('actual QOMP ops launch intent/allocation required')
        error=None
        sealed=False
        def post():
            nonlocal sealed
            q.runtime_check()
            self.validate_source()
            self.campaign.revalidate_binding(binding)
            sealed=True
        try:
            q.runtime_check()
            mkdir(run/'payload')
            if phase=='registration':
                members=self.held(binding)
                cases={mid:read(directory/'cases'/(mid+'.json')) for mid in binding['plan']['member_ids']}
                role=dict(kind='allocation',job_id=allocation['JobId'],allocation=allocation)
                scope=helper('bank_wave_transport').WaveScope(binding,cases,members,role)
                return register_once(self.client(binding,registration_scope=scope),scope,
                    save=lambda name,value:write(directory/name,value),before_publish=post)
            return publish_wave(self.client(binding),binding,allocation=allocation,
                save=lambda name,value:write(run/'payload'/name,value),before_publish=post)
        except BaseException as exc:
            error=exc
            raise
        finally:
            if not sealed:
                try:
                    post()
                except BaseException as final:
                    if error is None:
                        raise
                    error.add_note('QOMP wave postseal: '+str(final)[:1024])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'remote', 'ops'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--predecessor-state', type=Path)
    parser.add_argument('--wave-plan', type=Path)
    parser.add_argument('--wave-plan-sha256')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--phase', choices=('worker', 'science', 'delivery', 'audit',
                                          'registration', 'finalizer'))
    parser.add_argument('--action', choices=('initialize', 'submit', 'verify', 'release',
                                           'science-jobs', 'barrier', 'accept'))
    parser.add_argument('--adapter-seal')
    parser.add_argument('--producer',choices=('grid400','qomp'),default='grid400')
    parser.add_argument('--adoption-sha256')
    parser.add_argument('--member-limit',type=int,default=8)
    args = parser.parse_args()
    if not PRODUCTION_ENABLED:
        raise RuntimeError('production wave activation disabled pending coordinator review')
    native = helper('native_controller')
    qomp=args.producer=='qomp'
    if qomp:
        if (not helper('qomp_wave_policy').PRODUCTION_ENABLED
                and not (args.mode=='remote' and args.action=='initialize')):
            raise RuntimeError('QOMP wave activation disabled pending actual failure-lifecycle evidence/review')
        helper('qomp_execution').helper('phoenix_installed_checks').qomp_engine().qomp_enter()
        helper('bank_wave').sha(args.adoption_sha256)
        if not 1<=args.member_limit<=8:
            raise ValueError('QOMP wave width1..8 required')
    elif args.adoption_sha256 is not None or args.member_limit!=8:
        raise ValueError('QOMP authority/width cannot cross legacy wave mode')
    if args.mode in ('remote', 'ops'):
        (helper('qomp_wave_policy').validate_source if qomp else native.validate_source)(
            args.source,args.commit)
        production = Production(args)
        if args.mode == 'ops':
            result = production.operations(args.phase)
        elif args.action in ('submit', 'verify'):
            result = getattr(production, args.action)(args.phase)
        elif args.action == 'science-jobs':
            binding = production.binding(registered=True)
            result = dict(jobs={mid: slot['science_job_id']
                               for mid, slot in binding['registration']['members'].items()})
        else:
            result = getattr(production, args.action)()
        (helper('qomp_wave_policy').validate_source if qomp else native.validate_source)(
            args.source,args.commit)
        print(PREFIX+helper('bank_wave').encode(result).decode(), flush=True)
        return
    class Remote(native.Remote):
        def call(self, action, state):
            if self.seal() != self.adapter_seal:
                raise ValueError('reviewed SSH/watcher adapters changed')
            command = [str(native.MANAGER/'bin/python'), '-I', '-B',
                str(args.source/'devtools/isapol_s66_campaign/bank_wave_controller.py'),
                'remote', '--source', str(args.source), '--commit', args.commit,
                '--action', action, '--adapter-seal', self.adapter_seal]
            if action == 'initialize':
                command += ['--predecessor-state', str(args.predecessor_state)]
            if qomp:
                command += ['--producer','qomp','--adoption-sha256',args.adoption_sha256,
                            '--member-limit',str(args.member_limit)]
            for key in ('wave_plan', 'wave_plan_sha256', 'phase'):
                if key in state:
                    command += ['--'+key.replace('_', '-'), str(state[key])]
            clean = ['/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', '-u', 'LD_PRELOAD',
                'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
                'PATH='+str(native.MANAGER/'bin')+':/usr/bin:/bin',
                'LD_LIBRARY_PATH='+str(native.MANAGER/'lib')]
            if qomp:
                clean=['/usr/bin/env','-i','USER=awallace43','LOGNAME=awallace43',
                    'HOME=/storage/home/hcoda1/3/awallace43','LANG=C.UTF-8','LC_ALL=C.UTF-8',
                    'GIT_OPTIONAL_LOCKS=0','PATH='+str(native.MANAGER/'bin')+':/usr/bin:/bin',
                    'LD_LIBRARY_PATH='+str(native.MANAGER/'lib')]
            response = subprocess.run([str(self.phx), shlex.join(clean+command)],
                env=dict(os.environ, PHX_ALIAS='phoenix2'), text=True, capture_output=True, timeout=300)
            lines = [line[len(PREFIX):] for line in response.stdout.splitlines() if line.startswith(PREFIX)]
            if response.returncode or len(lines) != 1:
                raise RuntimeError('ambiguous remote wave operation; no automatic retry')
            return helper('bank_wave').decoded(lines[0].encode())
    remote = Remote(args)
    print(helper('bank_wave').encode(run_wave(remote.call, remote.wait)).decode(), flush=True)


if __name__ == '__main__':
    main()
