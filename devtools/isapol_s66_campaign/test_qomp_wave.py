"""Offline QOMP wave selection; no native runtime or server."""
import importlib.util
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest


def load(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('_qomp_wave_test_'+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def test_qomp_wave_requires_explicit_pinned_authority_not_grid400_fallback():
    core=load('bank_wave')
    with pytest.raises(ValueError,match='authority'):
        core.settings(dict(producer_profile='qomp',dataset_id=359))


def fixture(monkeypatch,tmp_path,width=8):
    serial,exp,pool,prepared,context,previous,proofs,binding,delivered=load(
        'test_qomp_expansion').synthetic(monkeypatch)
    wave=load('bank_wave')
    policy=wave.helper('qomp_wave_policy')
    policy._MODULES['qomp_expansion']=exp
    wave._MODULES['audited_bank']=serial
    monkeypatch.setattr(policy,'AUTHORITY_SHA256',binding['manifest_sha256'])
    monkeypatch.setattr(policy,'CONTEXT_SHA256',wave.digest(context))
    proofs.update(policy.extra_proofs(binding))
    ctx=json.loads(context)
    mids=ctx['eligible_ids'][1:1+width]
    directory=tmp_path/('wave-%04d-%04d'%(1,1+len(mids)))
    plan=dict(schema='native-eligible-wave-plan/1',producer_profile='qomp',
        producer_authority=json.loads(binding['manifest_raw']),member_limit=width,
        campaign_scope='qomp-wave',dispatch_protocol=wave.PROTOCOL,
        wave_id=directory.name,dataset_id=ctx['dataset_id'],source=str(policy.SOURCE),
        adoption_sha256='a'*64, # SYNTHETIC external cutover digest; never actual evidence.
        operational_commit='e'*40,bootstrap_context_path=str(tmp_path/'bank-context.json'),
        bootstrap_context_sha256=wave.digest(context),
        predecessor_path=str(tmp_path/'bank-checkpoint.json'),
        predecessor_sha256=wave.digest(wave.encode(previous)),
        predecessor_snapshot_sha256=previous['snapshot_sha256'],generation_before=1,
        generation_after=1+len(mids),eligible_ids=ctx['eligible_ids'],refused_ids=[],
        eligibility_sha256=ctx['eligibility_sha256'],member_ids=mids,members={})
    cases={mid:wave.bind_member(prepared[mid],plan=plan,monomer_id=mid) for mid in mids}
    plan['members']={mid:dict(case_path=str(directory/'cases'/(mid+'.json')),
        prepared_case_sha256=wave.digest(wave.encode(prepared[mid])),
        case_sha256=wave.digest(wave.encode(case)),execution_profile=case['execution_profile'])
        for mid,case in cases.items()}
    args=dict(prepared_cases=prepared,checkpoint_raw=wave.encode(previous),
        checkpoint_sha256=plan['predecessor_sha256'],context_raw=context,
        context_sha256=wave.digest(context),proof_blobs=proofs)
    return wave,policy,plan,cases,args,previous,delivered


@pytest.mark.parametrize('width',range(1,9))
def test_qomp_wave_selects_exact_eligible_prefix_with_unchanged_science(
        monkeypatch,tmp_path,width):
    wave,policy,plan,cases,args,previous,_=fixture(monkeypatch,tmp_path,width)
    raw=wave.encode(plan)
    wave.validate_plan(raw,wave.digest(raw),**args)
    assert len(cases)==width
    assert plan['dataset_id']==9999 # Synthetic dataset, NEVER a production fixture.
    for mid,case in cases.items():
        prepared=args['prepared_cases'][mid]
        assert case['entry']==prepared['entry']
        assert case['execution_profile']==prepared['execution_profile']
        assert case['campaign_scope']=='qomp-wave' and case['dataset_mode']=='append'
        policy.settings(plan).profile.validate_profile(case['execution_profile'])
    assert previous['generation']==1
    with pytest.raises(ValueError):
        wave.validate_plan(raw,wave.digest(raw),**(args|{'proof_blobs':{}}))


@pytest.mark.parametrize('bad',['manifest','dataset','grid_profile','width','membership','source_context'])
def test_qomp_plan_crossing_or_unbound_members_refuse(monkeypatch,tmp_path,bad):
    wave,_,plan,_,args,_,_=fixture(monkeypatch,tmp_path,2)
    if bad=='manifest': plan['producer_authority']['dataset_id']=357
    elif bad=='dataset': plan['dataset_id']=357
    elif bad=='width': plan['member_limit']=True
    elif bad=='membership': plan['member_ids'].reverse()
    elif bad=='source_context': args['context_sha256']='f'*64
    else:
        mid=plan['member_ids'][0]
        plan['members'][mid]['execution_profile']=wave.helper('grid400_bank_policy').execution_profile('s20')
    raw=wave.encode(plan)
    with pytest.raises((ValueError,KeyError)):
        wave.validate_plan(raw,wave.digest(raw),**args)


def completed_fixture(monkeypatch,tmp_path,width=2):
    """Clearly synthetic actual-observation shapes; no file/native/server IO."""
    wave,policy,plan,cases,args,previous,delivered=fixture(monkeypatch,tmp_path,width)
    base=wave.helper('dataset_guard')
    plan_raw=wave.encode(plan)
    before=previous['snapshot'];captures={};members={};completions={};receipts={}
    directory=Path(next(iter(plan['members'].values()))['case_path']).parent.parent
    role=dict(kind='allocation',job_id='8000',allocation=dict(JobId='8000',
        Account='gts-cs207-chemx',QOS='embers',Partition='cpu-small',NumCPUs='4',
        NumNodes='1',MinMemoryNode='24G',TimeLimit='00:45:00'))
    for mid in plan['member_ids']:
        record_receipts,views,_=delivered(cases[mid],before)
        out=json.loads(record_receipts['outcome'])
        identifier=out['record_id']
        for key in ('record','delivered'):
            views[key]['records'][str(identifier)]['wave_identity_sha256']=wave.digest(mid.encode())
        captures[mid]={key:views[key] for key in ('before','entry','record')}
        completions[mid]=views['delivered']['records'][str(identifier)]
        members[mid]=dict(case_sha256=plan['members'][mid]['case_sha256'],
            worker_preflight_sha256='f'*64,science_job_id=out['allocation']['JobId'],
            compute_tag=out['compute_tag'],entry_name=cases[mid]['entry']['name'],
            specification_name=cases[mid]['specification_name'],record_id=identifier,
            molecule_id=completions[mid]['molecule_id'],captures={key:dict(
                path=str(directory/'registration'/mid/(key+'.json')),
                snapshot_sha256=base.digest(views[key])) for key in captures[mid]})
        receipts[mid]=record_receipts
        before=views['record']
    registered=before
    reg=dict(schema='native-eligible-wave-registration/1',campaign_scope='qomp-wave',
        dispatch_protocol=wave.PROTOCOL,wave_id=plan['wave_id'],plan_sha256=wave.digest(plan_raw),
        dataset_id=plan['dataset_id'],bootstrap_context_sha256=args['context_sha256'],
        predecessor_sha256=plan['predecessor_sha256'],
        predecessor_snapshot_sha256=previous['snapshot_sha256'],registration_role=role,
        dispatch_intent_sha256='d'*64,member_ids=plan['member_ids'],members=members,
        registered_snapshot_path=str(directory/'registration/registered-snapshot.json'),
        registered_snapshot_sha256=base.digest(registered))
    reg_raw=wave.encode(reg)
    wave.validate_registration(plan,reg_raw,wave.digest(reg_raw),predecessor_snapshot=previous['snapshot'],
        cases=cases,captures=captures,registered_snapshot=registered)
    permit=dict(schema='native-eligible-wave-release-permit/1',dispatch_protocol=wave.PROTOCOL,
        wave_id=plan['wave_id'],plan_sha256=wave.digest(plan_raw),registration_sha256=wave.digest(reg_raw),
        dataset_id=plan['dataset_id'],predecessor_sha256=plan['predecessor_sha256'],
        dispatch_intent_sha256='d'*64,members=wave.permit_members(reg))
    permit_raw=wave.encode(permit)
    wave.validate_release_permit(plan,wave.digest(plan_raw),reg,wave.digest(reg_raw),permit_raw)
    runtime=load('bank_wave_runtime');runtime._MODULES['bank_wave']=wave
    binding=dict(plan=plan,plan_path=directory/'plan.json',plan_raw=plan_raw,
        plan_sha256=wave.digest(plan_raw),context_raw=args['context_raw'],
        context_sha256=args['context_sha256'],checkpoint_raw=args['checkpoint_raw'],
        checkpoint_sha256=args['checkpoint_sha256'],proof_blobs=args['proof_blobs'],
        registration=reg,registration_raw=reg_raw,registration_sha256=wave.digest(reg_raw),
        release_permit=permit,release_permit_raw=permit_raw,release_permit_sha256=wave.digest(permit_raw))
    fields=runtime.started_fields(binding)
    final=copy.deepcopy(registered)
    for mid in plan['member_ids']:
        final['records'][str(members[mid]['record_id'])]=completions[mid]
    outcomes={};deliveries={};sci={};audits={};audit_accounting={}
    accounting=load('test_bank_wave').accounting
    for mid in plan['member_ids']:
        slot,case=members[mid],cases[mid];profile=case['execution_profile']
        out=wave.decoded(receipts[mid]['outcome'])
        out.update(fields,dataset_id=plan['dataset_id'],campaign=case['campaign'],
            worker_preflight_sha256=slot['worker_preflight_sha256'],qcengine_internal_retries=0,
            execution_profile=profile,allocation=dict(JobId=slot['science_job_id'],
                Account='gts-cs207-chemx',QOS='embers',Partition='cpu-small',
                NumCPUs=str(profile['cores']),NumNodes='1',
                MinMemoryNode=str(profile['job_memory_GiB'])+'G',TimeLimit=profile['time_limit']))
        outcomes[mid]=wave.encode(out);sci[mid]=accounting(slot['science_job_id'])
        deliveries[mid]=copy.deepcopy(registered)
        deliveries[mid]['records'][str(slot['record_id'])]=completions[mid]
    barrier=dict(schema='native-eligible-wave-science-barrier/1',
        status='ALL_SCIENCE_POSITIVELY_VERIFIED',wave_id=plan['wave_id'],
        plan_sha256=binding['plan_sha256'],registration_sha256=binding['registration_sha256'],
        release_permit_sha256=binding['release_permit_sha256'],member_ids=plan['member_ids'],members={})
    for mid in plan['member_ids']:
        slot=members[mid];run=directory/'members'/mid/'science'
        barrier['members'][mid]={k:slot[k] for k in ('science_job_id','record_id','compute_tag','case_sha256')}
        barrier['members'][mid].update(outcome_path=str(run/'payload/qomp-outcome.json'),
            outcome_sha256=wave.digest(outcomes[mid]),accounting_path=str(run/'accounting.txt'),
            accounting_sha256=wave.digest(sci[mid]),delivery_snapshot_sha256=base.digest(deliveries[mid]))
    barrier_raw=wave.encode(barrier)
    wave.validate_barrier(plan,reg,permit_raw,barrier_raw,outcomes=outcomes,accounting=sci,deliveries=deliveries)
    for i,mid in enumerate(plan['member_ids']):
        audit=wave.decoded(receipts[mid]['audit'])
        job=str(9000+i)
        audit.update(fields,status='PASS_QOMP_WAVE_CASE_EVIDENCE',dataset_id=plan['dataset_id'],
            campaign=cases[mid]['campaign'],compute_tag=members[mid]['compute_tag'],
            worker_preflight_sha256=members[mid]['worker_preflight_sha256'],
            allocation=dict(role['allocation'],JobId=job),audit_commit=plan['operational_commit'],
            case_outcome_sha256=wave.digest(outcomes[mid]),science_barrier_sha256=wave.digest(barrier_raw),
            execution_profile=cases[mid]['execution_profile'])
        audit['captured_preservation'].update(fields,status='PASS_CAPTURED_WAVE_DATASET_PRESERVATION',
            before_sha256=base.digest(registered),after_sha256=base.digest(deliveries[mid]),
            peer_statuses={peer:'complete' if peer==mid else 'waiting' for peer in plan['member_ids']})
        audits[mid]=wave.encode(audit);audit_accounting[mid]=accounting(job)
    blobs=dict(args['proof_blobs'])
    def add(raw):
        digest=wave.digest(raw);blobs[digest]=raw;return digest
    link=dict(predecessor_sha256=add(args['checkpoint_raw']),plan_sha256=add(plan_raw),
        registration_sha256=add(reg_raw),permit_sha256=add(permit_raw),barrier_sha256=add(barrier_raw),
        registered_snapshot_sha256=add(wave.encode(registered)),final_snapshot_sha256=add(wave.encode(final)),
        receipts_sha256={},captures_sha256={},deliveries_sha256={},
        science_accounting_sha256={},audit_accounting_sha256={})
    for mid in plan['member_ids']:
        link['receipts_sha256'][mid]=dict(audit=add(audits[mid]),outcome=add(outcomes[mid]),
            case=add(wave.encode(cases[mid])),envelope=add(receipts[mid]['envelope']))
        link['captures_sha256'][mid]={key:add(wave.encode(value)) for key,value in captures[mid].items()}
        link['deliveries_sha256'][mid]=add(wave.encode(deliveries[mid]))
        link['science_accounting_sha256'][mid]=add(sci[mid])
        link['audit_accounting_sha256'][mid]=add(audit_accounting[mid])
    return NS(wave=wave,policy=policy,plan=plan,cases=cases,args=args,previous=previous,
        binding=binding,reg=reg,permit=permit,barrier=barrier,blobs=blobs,link=link,final=final,
        outcomes=outcomes,deliveries=deliveries,science_accounting=sci,audits=audits,
        audit_accounting=audit_accounting,captures=captures,registered=registered)


@pytest.mark.parametrize('width',[1,2,8])
def test_only_whole_qomp_wave_advances_preserving_actual_serial_provenance(
        monkeypatch,tmp_path,width):
    f=completed_fixture(monkeypatch,tmp_path,width)
    cp=f.wave.reduce_wave(f.args['checkpoint_raw'],f.args['checkpoint_sha256'],
        f.args['context_raw'],f.args['context_sha256'],f.blobs,link=f.link)
    assert cp['generation']==1+width
    assert cp['serial_anchor_sha256']==f.args['checkpoint_sha256']
    assert cp['dataset_id']==9999 and cp['schema']=='qomp-audited-bank-wave-checkpoint/1'
    assert cp['audited_records'][:1]==f.previous['audited_records']
    raw=f.wave.encode(cp)
    assert f.wave.validate_checkpoint(raw,f.wave.digest(raw),f.args['context_raw'],
        f.args['context_sha256'],f.blobs)==cp
    assert f.blobs[cp['serial_anchor_sha256']]==f.args['checkpoint_raw']


@pytest.mark.parametrize('bad',['missing_audit','failed_peer','foreign_tag','failed_accounting',
                                'changed_prefix','wrong_audit_source','bad_numeric','partial_barrier'])
def test_partial_failed_or_crossed_wave_never_advances(monkeypatch,tmp_path,bad):
    f=completed_fixture(monkeypatch,tmp_path,2)
    mid=f.plan['member_ids'][-1]
    if bad=='missing_audit': f.blobs.pop(f.link['receipts_sha256'][mid]['audit'])
    else:
        if bad in ('failed_peer','changed_prefix'):
            key=f.link['final_snapshot_sha256'];x=f.wave.decoded(f.blobs[key])
            if bad=='failed_peer': x['records'][str(f.reg['members'][mid]['record_id'])]['status']='error'
            else: x['metadata']['name']='foreign'
            field='final_snapshot_sha256'
        elif bad=='partial_barrier':
            key=f.link['barrier_sha256'];x=f.wave.decoded(f.blobs[key]);x['member_ids'].pop()
            field='barrier_sha256'
        elif bad=='failed_accounting':
            key=f.link['science_accounting_sha256'][mid]
            f.blobs[key]=f.blobs[key].replace(b'COMPLETED',b'FAILED')
            x=None
        elif bad=='foreign_tag':
            key=f.link['registration_sha256'];x=f.wave.decoded(f.blobs[key])
            x['members'][mid]['compute_tag']='ccasp-native-grid400-bank-'+mid+'-123'
            field='registration_sha256'
        else:
            key=f.link['receipts_sha256'][mid]['audit'];x=f.wave.decoded(f.blobs[key])
            if bad=='wrong_audit_source': x['audit_commit']='f'*40
            else: x['numeric']['status']='FAIL'
            field=None
        if x is not None:
            raw=f.wave.encode(x);digest=f.wave.digest(raw);f.blobs[digest]=raw
            if field: f.link[field]=digest
            else: f.link['receipts_sha256'][mid]['audit']=digest
    with pytest.raises((ValueError,KeyError)):
        f.wave.reduce_wave(f.args['checkpoint_raw'],f.args['checkpoint_sha256'],
            f.args['context_raw'],f.args['context_sha256'],f.blobs,link=f.link)


@pytest.mark.parametrize('first',['serial','wave'])
def test_canonical_member_reservation_blocks_cross_route_replay(monkeypatch,tmp_path,first):
    f=completed_fixture(monkeypatch,tmp_path,2)
    store=f.wave.helper('bank_store')
    monkeypatch.setattr(store,'QOMP_INTENTS',tmp_path/'intents')
    mid=f.plan['member_ids'][0];slot=f.reg['members'][mid]
    fields=dict(producer='qomp',dataset_id=f.plan['dataset_id'],
        checkpoint_sha256=f.args['checkpoint_sha256'],snapshot_sha256=f.previous['snapshot_sha256'],
        context_sha256=f.args['context_sha256'],eligibility_sha256=f.plan['eligibility_sha256'])
    serial=dict(fields,monomer_id=mid,compute_tag=slot['compute_tag'],
                case_sha256=slot['case_sha256'])
    wave=dict(fields,plan_sha256=f.binding['plan_sha256'],wave_id=f.plan['wave_id'],
        members={m:{key:s[key] for key in ('science_job_id','compute_tag','case_sha256',
            'worker_preflight_sha256')} for m,s in f.reg['members'].items()},
        registration_role=f.reg['registration_role'])
    if first=='serial':
        monkeypatch.setenv('SLURM_JOB_ID',slot['science_job_id'])
        store.reserve_intent(**serial)
        monkeypatch.setenv('SLURM_JOB_ID','8000')
        with pytest.raises(FileExistsError): store._reserve_wave_intent(**wave)
    else:
        monkeypatch.setenv('SLURM_JOB_ID','8000')
        saved=store._reserve_wave_intent(**wave)
        assert store._verify_wave_intent(saved,**wave)==saved['sha256']
        monkeypatch.setenv('SLURM_JOB_ID',slot['science_job_id'])
        with pytest.raises(FileExistsError): store.reserve_intent(**serial)
        monkeypatch.setenv('SLURM_JOB_ID','8000')
        with pytest.raises(FileExistsError): store._reserve_wave_intent(**wave)


@pytest.mark.parametrize('count',range(1,9))
def test_stage_admission_uses_live_callback_and_sum_largest_for_heterogeneous_profiles(count):
    controller=load('bank_wave_controller')
    policy=controller.helper('qomp_wave_policy').helper('qomp_bank_policy')
    fixtures=load('test_qomp_bank_policy')
    cases={str(i):dict(execution_profile=policy.execution_profile(policy.NAMES[i%5]))
           for i in range(count)}
    calls=[]
    def observe():
        calls.append('raw')
        return fixtures.observation(policy,count)
    for _ in range(2): # preadmission and release are independent fresh calls
        out=controller.admit_scratch(cases,producer='qomp',observe=observe)
        assert out['required_files']==(count+1)*65536
        assert out['required_bytes']==(count+1)*256*1024**3
    assert calls==['raw','raw']
    with pytest.raises(ValueError):
        controller.admit_scratch(cases,fixtures.observation(policy,count),producer='qomp')
    with pytest.raises(ValueError):
        controller.admit_scratch(cases,producer='qomp',observe=lambda:
            fixtures.observation(policy,count,quota_device=999))


def test_activation_is_closed_and_historical_held_jobs_cannot_be_consumed():
    policy=load('qomp_wave_policy')
    assert policy.PRODUCTION_ENABLED is False
    controller=load('bank_wave_controller')
    case=dict(producer_profile='qomp',campaign_scope='qomp-wave',
              execution_profile=policy.helper('qomp_bank_policy').execution_profile('s20'))
    for job in policy.PROTECTED_JOBS:
        with pytest.raises(ValueError,match='historical jobs'):
            controller.verify_held(dict(Command=str(policy.SOURCE/'devtools/isapol_s66_campaign/qomp148.sbatch')),
                                   case,job)


def adoption_fixture(monkeypatch,tmp_path,advance=False):
    serial,exp,_,prepared,context,checkpoint,proofs,binding,delivered=load(
        'test_qomp_expansion').synthetic(monkeypatch)
    exp._MODULES['audited_bank']=serial
    policy=load('qomp_wave_policy')
    policy._MODULES['qomp_expansion']=exp
    monkeypatch.setattr(policy,'AUTHORITY_SHA256',binding['manifest_sha256'])
    monkeypatch.setattr(policy,'CONTEXT_SHA256',serial.digest(context))
    monkeypatch.setattr(policy,'SERIAL_OPS','e'*40) # Explicit SYNTHETIC old serial source.
    monkeypatch.setattr(policy,'WAVE_ROOT',tmp_path/'waves')
    wave=policy.helper('bank_wave')
    wave._MODULES.update(audited_bank=serial,qomp_wave_policy=policy)
    if advance:
        prior=serial.encode(checkpoint)
        case=serial.bind_next_case(prepared[__import__('json').loads(context)['eligible_ids'][1]],
            prior,serial.digest(prior),context,serial.digest(context),proofs,
            **exp.reducer_options(binding))
        receipts,views,tensor=delivered(case,checkpoint['snapshot'])
        checkpoint,proofs=serial.advance(prior,serial.digest(prior),context,serial.digest(context),
            proofs,receipts=receipts,expected_audit_sha256=serial.digest(receipts['audit']),
            captured=views,tensor_raw=tensor,**exp.reducer_options(binding))
    ctx=json.loads(context);generation=checkpoint['generation']
    original=policy.ROOT/'runs/ccasp-native-s66-qomp148-bank003'
    mid=ctx['eligible_ids'][generation-1]
    state=dict(generation=generation,
        status='ORDINARY_AUDIT_ACCEPTED' if advance else 'EXTERNAL_603_ORDINARY_PILOT_ACCEPTED',
        bank_context=str(original/'runs'/mid/'audit/payload/bank-context.json') if advance
            else str(original/'context/bank-context.json'),bank_context_sha256=serial.digest(context),
        bank_checkpoint=str(original/'runs'/mid/'audit/payload/bank-checkpoint.json') if advance
            else str(original/'seed/bank-checkpoint.json'),
        bank_checkpoint_sha256=serial.digest(serial.encode(checkpoint)))
    links=checkpoint['ledger'][-1]['receipts_sha256'] if advance else ctx['seed_receipts_sha256']
    audit,outcome=(serial.helper('dataset_guard').plain(json.loads(proofs[links[k]]))
                   for k in ('audit','outcome'))
    verified=dict(phase='audit',job_id=audit['allocation']['JobId'],
                  audit_sha256=links['audit'],state=state)
    accounting=load('test_bank_wave').accounting
    science_raw=accounting(outcome['allocation']['JobId'])
    audit_raw=accounting(audit['allocation']['JobId'])
    manifest=dict(schema='qomp-wave-serial-cutover/1',producing_operations=policy.SERIAL_OPS,
        bootstrap_authority_sha256=binding['manifest_sha256'],state_sha256=serial.digest(serial.encode(state)),
        context_sha256=serial.digest(context),checkpoint_sha256=state['bank_checkpoint_sha256'],
        science_accounting_sha256=serial.digest(science_raw),
        audit_accounting_sha256=serial.digest(audit_raw),
        verification_sha256=serial.digest(serial.encode(verified)) if advance else None)
    root=policy.WAVE_ROOT/'inputs'
    def write(path,raw):
        path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        path.write_bytes(raw)
    for role,path in exp.helper('qomp_campaign').ORIGINAL_PATHS.items():
        write(root/'original25'/path,binding['original_anchors'][role])
    write(root/'bootstrap/authority.json',binding['manifest_raw'])
    for role,name in dict(worker='worker-preflight.json',science_accounting='science-accounting.txt',
                          audit_accounting='audit-accounting.txt').items():
        write(root/'bootstrap'/name,binding['extras'][role])
    for name,raw in {'adoption.json':serial.encode(manifest),'state.json':serial.encode(state),
                     'bank-context.json':context,'bank-checkpoint.json':serial.encode(checkpoint),
                     'science-accounting.txt':science_raw,'audit-accounting.txt':audit_raw,
                     'verified.json':serial.encode(verified)}.items():
        write(root/'adopted'/name,raw)
    for digest,raw in proofs.items(): write(root/'adopted/proofs'/(digest+'.json'),raw)
    return policy,manifest,root,state


@pytest.mark.parametrize('advance',[False,True])
def test_actual_shaped_external_seed_and_serial_successor_adopt_without_relabelling(
        monkeypatch,tmp_path,advance):
    policy,manifest,root,state=adoption_fixture(monkeypatch,tmp_path,advance)
    result=policy.load_adoption(policy.sha(policy.encode(manifest)))
    assert result['state']==state
    cp=json.loads(result['checkpoint_raw'])
    assert cp['generation']==(2 if advance else 1)
    assert state['bank_checkpoint'].startswith(str(policy.ROOT/'runs/ccasp-native-s66-qomp148-bank003'))
    assert policy.WAVE_ROOT==tmp_path/'waves' # SYNTHETIC metadata; no deployed paths read/written.


@pytest.mark.parametrize('bad',['state','accounting','proof','source','digest'])
def test_adoption_missing_changed_or_unsuccessful_evidence_refuses(monkeypatch,tmp_path,bad):
    policy,manifest,root,_=adoption_fixture(monkeypatch,tmp_path,True)
    digest=policy.sha(policy.encode(manifest))
    if bad=='state': (root/'adopted/state.json').write_bytes(b'{}')
    elif bad=='accounting':
        p=root/'adopted/audit-accounting.txt';p.write_bytes(p.read_bytes().replace(b'COMPLETED',b'FAILED'))
    elif bad=='proof': next((root/'adopted/proofs').iterdir()).unlink()
    elif bad=='source':
        manifest['producing_operations']='f'*40
        (root/'adopted/adoption.json').write_bytes(policy.encode(manifest))
        digest=policy.sha(policy.encode(manifest)) # Synthetic parent seal cannot authorize foreign source.
    else: digest='f'*64
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        policy.load_adoption(digest)


@pytest.mark.parametrize('quota_failure',[None,3,4])
def test_release_reobserves_aggregate_quota_for_each_own_held_member_and_stops(
        monkeypatch,tmp_path,quota_failure):
    f=completed_fixture(monkeypatch,tmp_path,2)
    controller=load('bank_wave_controller')
    controller._MODULES['bank_wave']=f.wave
    obj=object.__new__(controller.Production)
    obj.qomp=True;obj.policy=NS(PRODUCTION_ENABLED=True);obj.core=f.wave
    obj.campaign=controller.helper('bank_wave_campaign')
    obj.binding=lambda **kw:f.binding
    obj.queue=lambda:len(f.plan['member_ids'])
    obj.client=lambda binding:NS()
    events=[];written={};released=set();observed=[]
    def held(binding,*,members=None):
        mids=f.plan['member_ids'] if members is None else members
        # Once one SCI is running, it must NOT be rechecked as still HELD.
        assert not set(mids)&released
        return {mid:{k:f.reg['members'][mid][k] for k in (
            'science_job_id','compute_tag','case_sha256','worker_preflight_sha256')} for mid in mids}
    obj.held=held
    obj.validate_source=lambda:events.append('SOURCE')
    def quota(binding):
        observed.append('fresh-raw-callback');events.append('QUOTA')
        if len(observed)==quota_failure: raise ValueError('synthetic fresh quota refusal')
        return dict(SYNTHETIC_FRESH=True)
    obj.scratch_admission=quota
    original_read=controller.read
    monkeypatch.setattr(controller,'read',lambda path:
        dict(job_id='8000') if Path(path).name in ('verified.json','submission.json')
        else copy.deepcopy(f.cases[Path(path).stem]) if Path(path).parent.name=='cases'
        else original_read(path))
    monkeypatch.setattr(controller,'raw',lambda path:load('test_bank_wave').accounting('8000'))
    monkeypatch.setattr(controller,'_registration',lambda binding:
        (f.binding['registration_raw'],f.reg,f.captures,f.registered))
    monkeypatch.setattr(controller,'write',lambda path,value:written.setdefault(str(path),value))
    monkeypatch.setattr(controller.helper('bank_wave_transport'),'_snapshot_dataset',
                        lambda *a:copy.deepcopy(f.registered))
    def release(command,**kwargs):
        assert command[:2]==[controller.SLURM+'scontrol','release']
        mid=next(mid for mid,s in f.reg['members'].items() if s['science_job_id']==command[-1])
        assert mid not in released and events[-2:]==['QUOTA','SOURCE']
        released.add(mid);events.append('RELEASE')
        return NS(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(controller.subprocess,'run',release)
    if quota_failure is None:
        obj.release()
        assert released==set(f.plan['member_ids']) and len(observed)==4
    else:
        with pytest.raises(ValueError,match='fresh quota'):
            obj.release()
        assert len(released)==quota_failure-3
    assert any(path.endswith('release-permit.json') for path in written)
    assert not any(path.endswith('accepted.json') for path in written)


def test_commands_bind_positive_member_worker_hash_and_closed_wave_wrapper(monkeypatch,tmp_path):
    f=completed_fixture(monkeypatch,tmp_path,1)
    controller=load('bank_wave_controller')
    controller._MODULES.update(bank_wave=f.wave,qomp_wave_policy=f.policy)
    mid=f.plan['member_ids'][0]
    directory=f.binding['plan_path'].parent
    command=controller.phase_command(f.binding,mid,'worker')
    script=next(i for i,value in enumerate(command) if value.endswith('/qomp148.sbatch'))
    assert len(command[script+1:])==14
    assert command[script+1:script+6]==[
        str(f.policy.SOURCE),'e'*40,'worker',str(directory/'members'/mid/'worker'),
        str(directory/'cases'/(mid+'.json'))]
    assert '--qos=embers' in command and '--no-requeue' in command and '--hold' not in command
    with pytest.raises(ValueError,match='bounded regular'):
        controller.phase_command(f.binding,mid,'science')
    worker=directory/'members'/mid/'worker'
    (worker/'payload').mkdir(parents=True,mode=0o700)
    raw=b'SYNTHETIC_POSITIVELY_VERIFIED_WORKER_BYTES'
    (worker/'payload/worker-preflight.json').write_bytes(raw)
    (worker/'verified.json').write_bytes(f.wave.encode(dict(phase='worker',
        plan_sha256=f.binding['plan_sha256'],worker_preflight_sha256=f.wave.digest(raw))))
    command=controller.phase_command(f.binding,mid,'science')
    assert '--hold' in command and f.wave.digest(raw) in command
    f.binding.update(science_barrier_sha256=f.wave.digest(f.wave.encode(f.barrier)))
    command=controller.phase_command(f.binding,mid,'audit')
    assert f.binding['registration_sha256'] in command
    assert f.binding['science_barrier_sha256'] in command
    ops=controller.phase_command(f.binding,None,'finalizer')
    assert ops[-2:]==['qomp',f.plan['adoption_sha256']]
    (worker/'payload/worker-preflight.json').write_bytes(raw+b'TAMPER')
    with pytest.raises(ValueError,match='worker SHA'):
        controller.phase_command(f.binding,mid,'science')


def test_v4_cannot_consume_or_write_the_frozen_v3_serial_run_paths():
    expansion=load('qomp_expansion')
    with pytest.raises(ValueError,match='frozenv3'):
        expansion.validate_paths(NS())


def test_disabled_qomp_dispatch_stops_before_control_or_server_access(monkeypatch):
    controller=load('bank_wave_controller')
    original=controller.helper;calls=[]
    def helper(name):
        calls.append(name)
        assert name in ('native_controller','qomp_wave_policy')
        return original(name)
    monkeypatch.setattr(controller,'helper',helper)
    monkeypatch.setattr(sys,'argv',['controller','run','--producer','qomp',
        '--source','/unused','--commit','e'*40,'--adoption-sha256','a'*64])
    with pytest.raises(RuntimeError,match='activation disabled'):
        controller.main()
    assert calls==['native_controller','qomp_wave_policy']


@pytest.mark.parametrize('signal_primary',[False,True])
def test_ops_finalization_preserves_primary_error_and_never_publishes(
        monkeypatch,tmp_path,signal_primary):
    controller=load('bank_wave_controller')
    obj=object.__new__(controller.Production)
    obj.qomp=True;obj.policy=NS(PRODUCTION_ENABLED=True)
    directory=tmp_path/'wave';run=directory/'finalizer';run.mkdir(parents=True,mode=0o700)
    obj.args=NS(run=run)
    obj.campaign=controller.helper('bank_wave_campaign')
    obj.validate_source=lambda:None
    binding=dict(plan_path=directory/'plan.json',plan_sha256='a'*64)
    obj.binding=lambda **kw:binding
    obj.client=lambda binding:NS()
    original=controller.helper
    entered=[];seals=[]
    allocation=dict(JobId='12000',Account='gts-cs207-chemx',QOS='embers',Partition='cpu-small',
        NumCPUs='4',NumNodes='1',MinMemoryNode='24G',TimeLimit='00:45:00')
    def seal():
        seals.append('RUNTIME')
        if len(seals)==2: raise ValueError('secondary postseal failure')
    q=NS(helper=lambda name:NS(qomp_engine=lambda:NS(qomp_enter=lambda:entered.append('CONTROL'))),
         runtime_check=seal)
    monkeypatch.setattr(controller,'helper',lambda name:
        q if name=='qomp_execution' else NS(allocation=lambda:allocation)
        if name=='phoenix_installed_checks' else original(name))
    monkeypatch.setattr(controller,'phase_command',lambda *a:['SYNTHETIC_OPS'])
    monkeypatch.setattr(controller,'read',lambda path:
        dict(plan_sha256='a'*64,command=['SYNTHETIC_OPS']))
    monkeypatch.setenv('SLURM_JOB_ID','12000')
    primary=KeyboardInterrupt('synthetic TERM') if signal_primary else OSError('synthetic child failure')
    def fail(*a,**kw): raise primary
    monkeypatch.setattr(controller,'publish_wave',fail)
    with pytest.raises(type(primary)) as caught:
        obj._qomp_operations('finalizer')
    assert caught.value is primary
    assert any('secondary postseal failure' in note for note in primary.__notes__)
    assert entered==['CONTROL'] and seals==['RUNTIME','RUNTIME']
    assert list((run/'payload').iterdir())==[]
