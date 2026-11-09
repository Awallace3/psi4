"""Synthetic serial expansion guards only; no native code or server traffic."""
import importlib.util
import copy
import json
from pathlib import Path
import sys

import pytest


def load(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('_qexp_test_'+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def test_qomp_reducer_requires_explicit_parent_bootstrap_authority():
    core=load('audited_bank')
    with pytest.raises(ValueError,match='QOMP bootstrap'):
        core.initialize({}, {}, {}, 'a'*64, {}, b'', producer='qomp', qomp_binding=None)


def test_seed_science_wrapper_projection_is_frozen603_git_bytes():
    import ast
    import subprocess
    exp=load('qomp_expansion')
    root=Path(__file__).resolve().parents[2]
    def blob(name):
        return subprocess.check_output(['/usr/bin/git','-C',str(root),'show',
            exp.PILOT_OPS+':devtools/isapol_s66_campaign/'+name])
    def literal(name,key):
        return next(ast.literal_eval(node.value) for node in ast.parse(blob(name)).body
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key
                                                   for t in node.targets))
    worker=set(literal('phoenix_worker_cb63305.py','SOURCES'))
    worker.update(literal('qomp_execution.py','SOURCE_NAMES'))
    added={'phoenix_pilot_cb63305.py','phoenix_pilot_cb63305.sbatch','single_record_cb63305.py'}
    assert added-worker=={'phoenix_pilot_cb63305.sbatch'}
    assert exp.sha(blob('phoenix_pilot_cb63305.sbatch'))==exp.PILOT_SCI_WRAPPER_SHA


def synthetic(monkeypatch):
    """Explicitly SYNTHETIC authorities, never production evidence."""
    c,inputs,pool,anchors,cases,carry=load('test_qomp148').synthetic(monkeypatch)
    for mid,case in cases.items():
        case['entry']['attributes']['geometry_sha256']=c.sha(('SYNTHETIC distinct '+mid).encode())
    exp=load('qomp_expansion')
    exp._MODULES['qomp_campaign']=c
    core=load('audited_bank')
    original=core.helper
    monkeypatch.setattr(core,'helper',lambda n:exp if n=='qomp_expansion' else original(n))
    fixture=load('test_audited_bank')
    def delivered(case,before=None):
        receipts,captured,tensor=fixture.delivered(core,case,before)
        outcome,audit=(json.loads(receipts[k]) for k in ('outcome','audit'))
        fresh=case['dataset_mode']=='fresh'
        op=exp.PILOT_OPS if fresh else 'e'*40
        job=outcome['allocation']['JobId']
        tag=c.NAMESPACE+case['monomer_id']+'-'+job
        for obj in (outcome,audit):
            obj.update(producer_profile='qomp',
                campaign_scope='qomp-pilot' if fresh else 'qomp-serial',
                scientific_build_commit=exp.helper('qualified_qomp').SCIENTIFIC_COMMIT)
        outcome.update(operational_commit=op,compute_tag=tag,
            worker_preflight_sha256=exp.WORKER_SHA,
            operational_sources_sha256={'SYNTHETIC':'NOT_PRODUCTION',
                'phoenix_pilot_cb63305.sbatch':exp.PILOT_SCI_WRAPPER_SHA},
            installed_qualification_sha256=exp.helper('qualified_qomp').RECEIPT_SHA256['qualification'],
            producer_binding_sha256=exp.helper('qualified_qomp').RECEIPT_SHA256['qualification'])
        captured['record']['records'][str(outcome['record_id'])]['task']['compute_tag']=tag
        receipts['outcome']=core.encode(outcome)
        audit.update(status='PASS_QOMP_ORDINARY_PILOT_EVIDENCE' if fresh else 'PASS_QOMP_SERIAL_CASE_EVIDENCE',
            case_operational_commit=op,audit_commit=op,
            allocation=dict(JobId='888'),case_outcome_sha256=core.digest(receipts['outcome']))
        receipts['audit']=core.encode(audit)
        return receipts,captured,tensor
    first=cases[carry['missing_ids'][0]]
    # Seed receipt uses known source roles but synthetic records/accounting.
    worker=dict(status='PASS_QOMP_FRESH_WORKER_PREFLIGHT_ONLY',
        allocation=dict(JobId=exp.WORKER_JOB),operational_commit=exp.PILOT_OPS,
        scientific_build_commit=exp.helper('qualified_qomp').SCIENTIFIC_COMMIT,
        runtime_unchanged=True,operational_sources_sha256={'SYNTHETIC':'NOT_PRODUCTION'},
        harness_path='/SYNTHETIC/harness.py',harness_sha256='a'*64,
        qcengine_version='SYNTHETIC',qcfractalcompute_version='SYNTHETIC',
        manager_sources_sha256={'SYNTHETIC':'NOT_PRODUCTION'},
        transport_binding=load('bank_dataset').transport_binding(producer='qomp'))
    worker_raw=core.encode(worker)
    monkeypatch.setattr(exp,'WORKER_SHA',core.digest(worker_raw))
    receipts,captured,tensor=delivered(first)
    monkeypatch.setattr(exp,'PILOT_AUDIT_SHA',core.digest(receipts['audit']))
    monkeypatch.setattr(exp,'PILOT_OUTCOME_SHA',core.digest(receipts['outcome']))
    monkeypatch.setattr(exp,'PILOT_CASE_SHA',core.digest(receipts['case']))
    monkeypatch.setattr(exp,'PILOT_TENSOR_SHA',core.digest(tensor))
    seed_outcome=json.loads(receipts['outcome'])
    monkeypatch.setattr(exp,'PILOT_JOB',seed_outcome['allocation']['JobId'])
    accounting=lambda job:''.join(job+s+'|COMPLETED|0:0\n' for s in ('','.batch','.0')).encode()
    extras=dict(worker=worker_raw,science_accounting=accounting(exp.PILOT_JOB),
                audit_accounting=accounting('888'))
    q=exp.helper('qualified_qomp')
    manifest=dict(schema='qomp-ordinary-bootstrap-authority/1',
        compiled_science=q.SCIENTIFIC_COMMIT,installed_tests=q.TEST_COMMIT,
        qualifying_operations=q.QUALIFYING_OPS,descriptor_operations=c.DESCRIPTOR_OPS,
        science_operations=exp.PILOT_OPS,audit_operations=exp.PILOT_OPS,
        science_job=exp.PILOT_JOB,worker_job=exp.WORKER_JOB,worker_sha256=exp.WORKER_SHA,
        audit_job='888',dataset_id=9999,record_id=seed_outcome['record_id'],
        seed_receipts_sha256={k:core.digest(v) for k,v in receipts.items()},
        snapshot_sha256=core.helper('dataset_guard').digest(captured['delivered']),
        tensor_sha256=core.digest(tensor),
        science_accounting_sha256=core.digest(extras['science_accounting']),
        audit_accounting_sha256=core.digest(extras['audit_accounting']))
    raw=core.encode(manifest)
    binding=dict(manifest_raw=raw,manifest_sha256=core.digest(raw),
                 original_anchors=anchors,extras=extras)
    options=exp.reducer_options(binding)
    context,checkpoint=core.initialize(pool,cases,receipts,core.digest(receipts['audit']),
        captured['delivered'],tensor,**options)
    proofs={core.digest(raw):raw for raw in receipts.values()}
    return core,exp,pool,cases,context,checkpoint,proofs,binding,delivered


def test_actual_seed_required_then_derived105_next_append_checkpoint(monkeypatch):
    core,exp,pool,cases,context,checkpoint,proofs,binding,delivered=synthetic(monkeypatch)
    options=exp.reducer_options(binding)
    ctx=json.loads(context)
    assert ctx['schema']=='qomp-audited-bank-context/1'
    assert len(ctx['eligible_ids'])-checkpoint['generation']==105
    assert ctx['eligible_ids'][0]=='monomer-0002' # Unit data is noncontiguous, not real0026.
    assert checkpoint['generation']==1 and len(checkpoint['audited_records'])==1
    prior=core.encode(checkpoint)
    case=core.bind_next_case(cases[ctx['eligible_ids'][1]],prior,core.digest(prior),
        context,core.digest(context),proofs,**options)
    assert case['dataset_mode']=='append' and case['campaign_scope']=='qomp-serial'
    assert case['dataset_id']==9999
    receipts,captured,tensor=delivered(case,checkpoint['snapshot'])
    after,new_proofs=core.advance(prior,core.digest(prior),context,core.digest(context),proofs,
        receipts=receipts,expected_audit_sha256=core.digest(receipts['audit']),
        captured=captured,tensor_raw=tensor,**options)
    assert after['generation']==2 and len(after['audited_records'])==2
    core.validate_checkpoint(core.encode(after),core.digest(core.encode(after)),
                             context,core.digest(context),new_proofs,**options)
    with pytest.raises(ValueError):
        core.validate_checkpoint(prior,core.digest(prior),context,core.digest(context),proofs)


@pytest.mark.parametrize('change',[
    'audit_pin','science_failed','audit_failed','science_source','old_dataset',
    'old_record','worker_bytes','partial_proofs','changed_predecessor'])
def test_partial_stale_foreign_or_unqualified_seed_is_not_authority(monkeypatch,change):
    core,exp,_,_,context,checkpoint,proofs,binding,_=synthetic(monkeypatch)
    bad=copy.deepcopy(binding)
    manifest=json.loads(bad['manifest_raw'])
    if change=='audit_pin': manifest['seed_receipts_sha256']['audit']='f'*64
    elif change=='science_source': manifest['science_operations']='f'*40
    elif change=='old_dataset': manifest['dataset_id']=357
    elif change=='old_record': manifest['record_id']=5234328
    elif change=='worker_bytes': bad['extras']['worker']+=b' '
    elif change in ('science_failed','audit_failed'):
        role='science_accounting' if change=='science_failed' else 'audit_accounting'
        bad['extras'][role]=bad['extras'][role].replace(b'COMPLETED|0:0',b'FAILED|1:0')
        manifest[role+'_sha256']=core.digest(bad['extras'][role])
    elif change=='partial_proofs':
        proofs.pop(next(iter(proofs)))
    else:
        checkpoint['snapshot']['metadata']['name']='changed'
        checkpoint['snapshot_sha256']=core.helper('dataset_guard').digest(checkpoint['snapshot'])
    bad['manifest_raw']=core.encode(manifest)
    bad['manifest_sha256']=core.digest(bad['manifest_raw']) # Synthetic parent pin; semantic guards must still reject.
    with pytest.raises(ValueError):
        core.validate_checkpoint(core.encode(checkpoint),core.digest(core.encode(checkpoint)),
            context,core.digest(context),proofs,producer='qomp',qomp_binding=bad)


def controller_fixture(monkeypatch,tmp_path):
    import base64
    core,exp,pool,cases,context,checkpoint,proofs,binding,delivered=synthetic(monkeypatch)
    root=tmp_path/'bank';root.mkdir(mode=0o700)
    bootstrap=root/'inputs/bootstrap';bootstrap.mkdir(mode=0o700,parents=True)
    monkeypatch.setattr(exp,'CAMPAIGN_ROOT',root)
    monkeypatch.setattr(exp,'CASE_ROOT',root/'cases')
    monkeypatch.setattr(exp,'BOOTSTRAP',bootstrap)
    monkeypatch.setattr(exp,'SOURCE',Path('/unit/source'))
    monkeypatch.setattr(exp,'ADMISSION_SCRATCH',tmp_path/'admission')
    monkeypatch.setattr(exp,'validate_source',lambda *args:None)
    c=exp.helper('qomp_campaign')
    monkeypatch.setattr(c,'anchors',lambda:binding['original_anchors'])
    carry=c.original25(binding['original_anchors'])
    monkeypatch.setattr(exp,'actual_pool',lambda:(pool,{},copy.deepcopy(cases),carry))
    store=exp.helper('bank_store')
    monkeypatch.setattr(store,'QOMP_INTENTS',tmp_path/'intents')
    monkeypatch.setattr(store,'QOMP_PROOFS',tmp_path/'proofs')
    files=dict(authority='authority.json',worker='worker-preflight.json',
               science_accounting='science-accounting.txt',audit_accounting='audit-accounting.txt')
    (bootstrap/files['authority']).write_bytes(binding['manifest_raw'])
    for key,raw in binding['extras'].items(): (bootstrap/files[key]).write_bytes(raw)
    ctx=json.loads(context)
    for role,name in dict(audit='audit-report.json',outcome='qomp-outcome.json',
                          case='case.json',envelope='server-native-envelope.json').items():
        (bootstrap/name).write_bytes(proofs[ctx['seed_receipts_sha256'][role]])
    (bootstrap/'bank-initial-snapshot.json').write_bytes(core.encode(checkpoint['snapshot']))
    envelope=json.loads(proofs[ctx['seed_receipts_sha256']['envelope']])
    (bootstrap/'server-tensors.npz').write_bytes(base64.b64decode(envelope['artifacts']['tensors.npz']['data']))
    controller=load('bank_controller')
    original=controller.helper
    audit=original('bank_audit')
    audit._MODULES.update(audited_bank=core,bank_store=store,qomp_expansion=exp)
    exp._MODULES['audited_bank']=core
    execution=load('qomp_execution')
    execution._MODULES.update(qomp_expansion=exp,qomp_campaign=c)
    science=original('phoenix_pilot_cb63305')
    old_science_helper=science.helper
    monkeypatch.setattr(science,'helper',lambda name:
        execution if name=='qomp_execution' else old_science_helper(name))
    monkeypatch.setattr(controller,'helper',lambda n:
        exp if n=='qomp_expansion' else core if n=='audited_bank'
        else store if n=='bank_store' else original(n))
    args=__import__('types').SimpleNamespace(source=exp.SOURCE,commit='e'*40,adapter_seal='d'*64,
        producer='qomp',bootstrap_sha256=binding['manifest_sha256'],phase='worker',monomer=None)
    return controller,exp,core,args,cases,delivered


def test_existing_controller_qomp_seed_selection_and_no_second_creation(monkeypatch,tmp_path):
    controller,exp,core,args,cases,_=controller_fixture(monkeypatch,tmp_path)
    state=controller.initialize(args)
    assert state['generation']==1 and state['status']=='EXTERNAL_603_ORDINARY_PILOT_ACCEPTED'
    selection=controller.select(args)
    args.monomer=selection['monomer']
    assert selection['monomer']==json.loads(Path(state['bank_context']).read_bytes())['eligible_ids'][1]
    _,_,case,bound,_=controller.selected(args)
    assert case['dataset_mode']=='append' and case['dataset_id']==9999
    assert case['checkpoint_sha256']==state['bank_checkpoint_sha256']
    assert len(controller.context(args)[0]['eligible_ids'])==106
    with pytest.raises(FileExistsError): controller.select(args)
    with pytest.raises(FileExistsError): controller.initialize(args)
    cmd=controller.phase_command(args.source,args.commit,case,state,'worker',
        producer='qomp',bootstrap_sha256=args.bootstrap_sha256)
    assert '--qos=embers' in cmd and '--no-requeue' in cmd
    assert str(exp.CASE_ROOT/(case['monomer_id']+'.json')) in cmd
    assert cmd[-1]==args.bootstrap_sha256
    assert not any('fresh' in value for value in cmd)


def test_qomp_controller_allocation_observes_new_quota_twice_and_never_replays(monkeypatch,tmp_path):
    from types import SimpleNamespace as NS
    controller,exp,_,args,_,_=controller_fixture(monkeypatch,tmp_path)
    controller.initialize(args)
    selected=controller.select(args);args.monomer=selected['monomer']
    observed=[];submissions=[]
    monkeypatch.setattr(exp,'controller_quota',
        lambda case:observed.append(case['monomer_id']) or dict(SYNTHETIC=True))
    monkeypatch.setattr(controller.subprocess,'check_output',lambda *a,**kw:'')
    monkeypatch.setattr(controller.subprocess,'run',lambda cmd,**kw:
        submissions.append(cmd) or NS(returncode=0,stdout='' if '--test-only' in cmd else '777\n',stderr=''))
    assert controller.submit(args)['job_id']=='777'
    assert observed==[args.monomer,args.monomer]
    assert len(submissions)==2
    with pytest.raises(FileExistsError): controller.submit(args)
    assert len(submissions)==2
    args.phase='science'
    with pytest.raises(ValueError): controller.submit(args)
    assert len(submissions)==2


@pytest.mark.parametrize('controller_seal_failure',[False,True])
def test_serial_controller_positive_worker_science_audit_advance(
        monkeypatch,tmp_path,controller_seal_failure):
    from types import SimpleNamespace as NS
    controller,exp,core,args,_,delivered=controller_fixture(monkeypatch,tmp_path)
    first=controller.initialize(args)
    selection=controller.select(args);args.monomer=selection['monomer']
    _,_,case,bound,_=controller.selected(args)
    profile=case['execution_profile']
    receipts,views,tensor=delivered(case,json.loads(bound['checkpoint_raw'])['snapshot'])
    outcome=json.loads(receipts['outcome'])
    jobs=dict(worker='777',science=outcome['allocation']['JobId'],audit='888')
    accounting=lambda job:''.join(job+s+'|COMPLETED|0:0\n' for s in ('','.batch','.0'))
    monkeypatch.setattr(exp,'controller_quota',lambda case:dict(SYNTHETIC=True))
    monkeypatch.setattr(controller.subprocess,'check_output',lambda cmd,**kw:
        accounting(cmd[cmd.index('-j')+1]) if cmd[0].endswith('/sacct') else '')
    def run(cmd,**kw):
        if '--test-only' in cmd: return NS(returncode=0,stdout='',stderr='')
        phase=next(p for p in ('worker','science','audit') if p in cmd)
        return NS(returncode=0,stdout=jobs[phase]+'\n',stderr='')
    monkeypatch.setattr(controller.subprocess,'run',run)
    q=exp.helper('qualified_qomp')
    c=exp.helper('qomp_campaign')
    qualification=tmp_path/'installed-validation.json';qualification.write_bytes(b'SYNTHETIC_QUAL')
    monkeypatch.setattr(c,'QUALIFICATION',qualification)
    def alloc(phase):
        return dict(JobId=jobs[phase],Account='gts-cs207-chemx',QOS='embers',Partition='cpu-small',
            NumCPUs='4' if phase=='audit' else str(profile['cores']),NumNodes='1',
            MinMemoryNode='24G' if phase=='audit' else str(profile['job_memory_GiB'])+'G',
            TimeLimit=profile['time_limit'] if phase=='science' else '00:45:00')
    store=exp.helper('bank_store')
    def save(path,value): path.write_bytes(core.encode(value))
    args.phase='worker';controller.submit(args)
    w=exp.phase_path(args.monomer,'worker')/'payload';w.mkdir(mode=0o700)
    worker=controller.helper('phoenix_worker_cb63305')
    worker_path=w/'worker-preflight.json'
    scope=dict(producer='qomp',campaign_scope='qomp-serial')
    preflight=dict(status='PASS_QOMP_SERIAL_WORKER_PREFLIGHT_ONLY',producer_profile='qomp',
        campaign_scope='qomp-serial',dataset_mode='append',runtime_unchanged=True,
        scientific_build_commit=q.SCIENTIFIC_COMMIT,operational_commit=args.commit,
        allocation=alloc('worker'),producer_binding_sha256=core.digest(qualification.read_bytes()),
        qualification_sha256=core.digest(qualification.read_bytes()),pilot_sha256=selection['case_sha256'],
        runtime_input_sha256=case['runtime_input_sha256'],monomer_id=case['monomer_id'],
        execution_profile=profile,operational_sources_sha256=worker.source_hashes(remaining_case=True,**scope),
        transport_binding=controller.helper('bank_dataset').transport_binding(producer='qomp'),
        task_config=dict(ncores=profile['cores'],memory_GiB=profile['science_memory_GiB'],internal_retries=0),
        checkpoint_sha256=case['checkpoint_sha256'],bootstrap_context_sha256=case['bootstrap_context_sha256'])
    seed=exp.audit_policy(bound['qomp_binding']).worker
    preflight.update({k:seed[k] for k in ('harness_path','harness_sha256','qcengine_version',
                                        'qcfractalcompute_version','manager_sources_sha256')})
    save(worker_path,preflight)
    controller.verify(args)
    args.phase='science';controller.submit(args)
    science=exp.phase_path(args.monomer,'science')/'payload';science.mkdir(mode=0o700)
    monkeypatch.setenv('SLURM_JOB_ID',jobs['science'])
    intent=store.reserve_intent(producer='qomp',dataset_id=case['dataset_id'],
        checkpoint_sha256=bound['checkpoint_sha256'],
        snapshot_sha256=core.helper('dataset_guard').digest(views['before']),
        context_sha256=bound['context_sha256'],case_sha256=core.digest(core.encode(case)),
        eligibility_sha256=case['eligibility_sha256'],monomer_id=case['monomer_id'],
        compute_tag=outcome['compute_tag'])
    outcome.update(allocation=alloc('science'),dispatch_intent_sha256=intent['sha256'],
        worker_preflight_sha256=core.digest(worker_path.read_bytes()),qcengine_internal_retries=0)
    for role,name in dict(before='owned-before-entry.json',entry='owned-after-entry.json',
                          record='owned-after-record.json',delivered='owned-after-delivery.json').items():
        save(science/name,views[role])
    save(science/'dispatch-intent.json',intent)
    save(science/'server-native-envelope.json',json.loads(receipts['envelope']))
    (science/'server-tensors.npz').write_bytes(tensor)
    save(science/'qomp-outcome.json',outcome)
    audit_helper=controller.helper('bank_audit')
    audit_helper._MODULES.update(audited_bank=core,bank_store=store,qomp_expansion=exp)
    controller.verify(args)
    args.phase='audit';controller.submit(args)
    audit=exp.phase_path(args.monomer,'audit')/'payload';audit.mkdir(mode=0o700)
    report=json.loads(receipts['audit'])
    report.update(allocation=alloc('audit'),case_outcome_sha256=core.digest(core.encode(outcome)))
    report['captured_preservation']=audit_helper.validate_preservation(
        science,outcome,case,bound,core.normalize_specification(case['specification']),producer='qomp')
    receipts.update(audit=core.encode(report),outcome=core.encode(outcome))
    after,proofs=core.advance(bound['checkpoint_raw'],bound['checkpoint_sha256'],
        bound['context_raw'],bound['context_sha256'],bound['proof_blobs'],
        receipts=receipts,expected_audit_sha256=core.digest(receipts['audit']),
        captured=views,tensor_raw=tensor,**exp.reducer_options(bound['qomp_binding']))
    store.store_receipts(receipts,producer='qomp')
    (audit/'audit-report.json').write_bytes(receipts['audit'])
    (audit/'bank-context.json').write_bytes(bound['context_raw'])
    (audit/'bank-checkpoint.json').write_bytes(core.encode(after))
    save(audit/'bank-advance.json',dict(generation=2,audit_sha256=core.digest(receipts['audit']),
        context_path=str(audit/'bank-context.json'),context_sha256=bound['context_sha256'],
        checkpoint_path=str(audit/'bank-checkpoint.json'),checkpoint_sha256=core.digest(core.encode(after))))
    monkeypatch.setattr(audit_helper,'approved_client',lambda *a,**kw:NS(SYNTHETIC=True))
    monkeypatch.setattr(audit_helper.helper('dataset_guard'),'snapshot_dataset',
                        lambda *a,**kw:copy.deepcopy(after['snapshot']))
    if controller_seal_failure:
        def refuse(*args): raise ValueError('synthetic final controller source seal')
        monkeypatch.setattr(exp,'validate_source',refuse)
        with pytest.raises(ValueError,match='final controller source seal'):
            controller.verify(args)
        assert not (audit.parent/'verified.json').exists()
        assert not (exp.CAMPAIGN_ROOT/'states/002.json').exists()
        assert json.loads((exp.CAMPAIGN_ROOT/'states/001.json').read_bytes())==first
        return
    state=controller.verify(args)
    assert state['generation']==2 and first['generation']==1
    _,latest,next_case,_,=controller.context(args)
    assert latest==state and next_case['monomer_id']!=args.monomer
    assert len(json.loads(bound['context_raw'])['eligible_ids'])-state['generation']==104


@pytest.mark.parametrize('post_fails',[False,True])
@pytest.mark.parametrize('prepare_fails',[False,True])
def test_audit_candidate_reduction_and_full_postseal_precede_any_success_publication(
        monkeypatch,tmp_path,post_fails,prepare_fails):
    from types import SimpleNamespace as NS
    e=load('qomp_execution')
    original=e.helper
    events=[]
    controls=NS(qomp_enter=lambda:None)
    def prepare(args,report):
        events.append('reduce')
        if prepare_fails: raise OSError('synthetic reduction failure')
        return dict(SYNTHETIC=True)
    exp=NS(validate_paths=lambda *a:None,load_checkpoint=lambda *a:None,
           prepare_audit_publication=prepare)
    writer=NS(commit_publication=lambda *a:events.append('publish-checkpoint'))
    monkeypatch.setattr(e,'helper',lambda name:
        NS(qomp_engine=lambda:controls,allocation=lambda:{}) if name=='phoenix_installed_checks'
        else exp if name=='qomp_expansion' else writer if name=='bank_audit' else original(name))
    monkeypatch.setattr(e,'context',lambda args:{})
    monkeypatch.setattr(e,'SOURCE',Path(e.__file__).resolve().parents[2])
    monkeypatch.setattr(e,'git',lambda source,*a:'' if a[0]=='status' else 'e'*40)
    monkeypatch.setattr(e,'source_changes',lambda *a:[])
    monkeypatch.setattr(e,'qualified',lambda:NS(SCIENTIFIC_COMMIT='f'*40,
                                               verify_descriptor_operations=lambda *a:None))
    def seal():
        events.append('full-seal')
        if events.count('full-seal')==2 and post_fails:
            raise ValueError('synthetic postseal failure')
    monkeypatch.setattr(e,'runtime_check',seal)
    args=NS(producer='qomp',campaign_scope='qomp-serial',remaining_case=True,commit='e'*40)
    out=tmp_path/'audit-report.json'
    if prepare_fails or post_fails:
        with pytest.raises(OSError if prepare_fails else ValueError):
            e.run_phase('audit',args,lambda args:(out,dict(status='SYNTHETIC_ONLY')))
        assert not out.exists() and 'publish-checkpoint' not in events
    else:
        e.run_phase('audit',args,lambda args:(out,dict(status='SYNTHETIC_ONLY')))
        assert events==['full-seal','reduce','full-seal','publish-checkpoint']
        assert json.loads(out.read_bytes())['status']=='SYNTHETIC_ONLY'


@pytest.mark.parametrize('suffix',['.1','.extern'])
def test_qomp_accounting_rejects_failed_or_extra_computational_rows(suffix):
    exp=load('qomp_expansion')
    text=''.join('123'+s+'|COMPLETED|0:0\n' for s in ('','.batch','.0'))
    with pytest.raises(ValueError):
        exp.completed_rows(text+'123'+suffix+'|FAILED|1:0\n','123')
    if suffix=='.extern':
        exp.completed_rows(text+'123.extern|COMPLETED|0:0\n','123')
