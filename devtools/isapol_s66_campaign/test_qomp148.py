"""Pure synthetic guards. Never scientific/production evidence or HTTP."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


def load(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('_q148test_'+name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
    exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__)
    return m


def synthetic(monkeypatch):
    c=load('qomp_campaign')
    _,inputs,pool=load('test_grid400_bank_case').fixture()
    geometry=[0.,0.,0.,3.,0.,0.,6.,0.,0.]
    entries=json.loads(inputs['entries'])
    for row,entry,report in zip(pool['bank']['monomers'],entries['entries'],pool['allocation']['reports']):
        row['runtime_input']['molecule']['geometry']=geometry
        entry['molecule']['geometry']=geometry
        request=row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
        request['declaration']=dict(frames=[[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]]*3)
        request['implementation_sha256']=c.helper('qualified_qomp').IMPLEMENTATION
        row['runtime_input_sha256']=c.sha(c.encode(row['runtime_input']))
        report.update(status='PASS_DESCRIPTOR_PLAN_ONLY',runtime_input_sha256=row['runtime_input_sha256'])
    inputs['entries']=json.dumps(entries).encode()
    pool['bank']['input_sha256']['entries']=c.sha(inputs['entries'])
    pool.update(eligible_ids=[r['monomer_id'] for r in pool['bank']['monomers']],refused_ids=[])
    qualified=c.helper('qualified_qomp')
    pool['bank']['producer']=dict(scientific_commit=qualified.SCIENTIFIC_COMMIT,
        implementation_sha256=qualified.IMPLEMENTATION,status='SYNTHETIC_UNIT_ONLY')
    pool['authority']=c.authority()
    accepted=[f'monomer-{i:04d}' for i in range(1,50,2)] # Intentionally NOT1..25.
    context=dict(dataset_id=357,first_entry=accepted[0]+'-'+'a'*12,first_record=1000,
        seed_receipts_sha256={},producer=dict(version='SYNTHETIC_OLD_NOT_QOMP'),
        dataset_name='SYNTHETIC_ORIGINAL',specification_name='SYNTHETIC_SPEC')
    context_raw=c.encode(context)
    rows=[dict(monomer_id=mid,record_id=1000+i,entry_name=mid+'-'+'a'*12)
          for i,mid in enumerate(accepted[1:],1)]
    checkpoint=dict(dataset_id=357,generation=25,bootstrap_context_sha256=c.sha(context_raw),
        ledger=rows,audited_records=list(range(1000,1025)),
        snapshot=dict(records={str(i):dict(status='complete') for i in range(1000,1025)}))
    checkpoint_raw=c.encode(checkpoint)
    state=dict(status='ORDINARY_AUDIT_ACCEPTED',generation=25,
        bank_context_sha256=c.sha(context_raw),bank_checkpoint_sha256=c.sha(checkpoint_raw))
    anchors=dict(state=c.encode(state),context=context_raw,checkpoint=checkpoint_raw)
    monkeypatch.setattr(c,'ORIGINAL_SHA',{k:c.sha(v) for k,v in anchors.items()})
    cases,carry=c.prepared_cases(pool,inputs,anchors)
    return c,inputs,pool,anchors,cases,carry


def test_original_ids_derived_noncontiguous_and_only106_drafts(monkeypatch):
    c,_,_,anchors,cases,carry=synthetic(monkeypatch)
    assert len(cases)==106 and len(carry['records'])==25
    assert carry['accepted_ids']==[f'monomer-{i:04d}' for i in range(1,50,2)]
    assert carry['missing_ids'][0]=='monomer-0002'
    assert set(cases).isdisjoint(carry['accepted_ids'])
    assert cases['monomer-0002']['dataset_mode']=='fresh'
    assert sum(d['dataset_mode']=='fresh' for d in cases.values())==1
    assert all('dataset_id' not in d for d in cases.values())
    assert carry['original_producer']['version']=='SYNTHETIC_OLD_NOT_QOMP'
    for role in anchors:
        with pytest.raises(ValueError,match='anchor'):
            c.original25(anchors|{role:anchors[role]+b' '})
    with pytest.raises(ValueError):
        c.original25({})


def test_actual_frozen_cba_inventory_not_current147_or148_hashes():
    c=load('qomp_campaign')
    authority=c.authority()
    assert len(authority['operational_sources_sha256'])==19
    assert authority['operational_commit']==c.DESCRIPTOR_OPS
    profiles=load('descriptor_profiles').profile('qomp',scope='full-bank')
    current=load('phoenix_descriptor_bank_cb63305').operation_hashes(producer='qomp',scope='full-bank')
    assert current!=authority['operational_sources_sha256']
    assert set(profiles['child_sources'])!=set(c.CBA_CHILDREN)
    pool=load('descriptor_pool')
    with pytest.raises(ValueError,match='producing-cba'):
        pool.verify_pool({}, {},b'',b'',b'',{},
            authority=authority|dict(operational_sources_sha256=current),producer='qomp')


def test_wrong_record_tag_profile_and_runtime_cannot_bind(monkeypatch):
    c,_,_,_,cases,_=synthetic(monkeypatch)
    case=next(d for d in cases.values() if d['dataset_mode']=='fresh')
    e=load('qomp_execution')
    runtime=e.helper('phoenix_worker_probe').runtime_input(case)
    runtime['specification']['protocols']=e.helper('phoenix_pilot').execution_specification({})['protocols']
    tag=c.NAMESPACE+case['monomer_id']+'-123'
    kwargs=dict(runtime=runtime,compute_tag=tag,worker_shape=(8,20),record_id=9999)
    assert e.validate_binding(case,**kwargs)==c.PROTECTED_RECORDS
    for changed in (dict(record_id=5234353),dict(compute_tag='foreign'),
                    dict(worker_shape=(24,120)),dict(runtime={})):
        with pytest.raises(ValueError):
            e.validate_binding(case,**(kwargs|changed))
    other=copy.deepcopy(case)
    other['execution_profile']=load('grid400_bank_policy').execution_profile('s20')
    with pytest.raises(ValueError):
        e.validate_binding(other,**kwargs)
    other=copy.deepcopy(case)
    other['producer']['implementation_sha256']=dict(core='old',isapol_python='old')
    with pytest.raises(ValueError):
        e.validate_binding(other,**kwargs)


@pytest.mark.parametrize('tags',[['*'],['ccasp-*'],['ccasp-native-qomp148-other-123'],['same']])
def test_unrelated_wildcard_or_same_cohort_active_manager_refused(tags):
    with pytest.raises(ValueError):
        load('qomp_execution').manager_isolation([NS(compute_tags=tags)],'same')


def test_intended_distinct_manager_tags_are_not_mutated():
    load('qomp_execution').manager_isolation([NS(compute_tags=['unrelated-literal'])],'same')


def test_phase_commands_have_real_wrapper_and_closed_resources(monkeypatch,tmp_path):
    c,_,_,_,cases,_=synthetic(monkeypatch)
    case=next(d for d in cases.values() if d['dataset_mode']=='fresh')
    source=c.ROOT/'gits/psi4.ccasp-native-qomp-production148-v2'
    for phase in ('worker','science','audit'):
        kw={} if phase=='worker' else dict(preflight='/observed/worker.json',preflight_sha='a'*64)
        if phase=='audit': kw['science']='/observed/science'
        cmd=c.phase_command(case,source=source,commit='c'*40,phase=phase,
            run=c.RUNS/('ccasp-native-s66-qomp148-'+phase+'002'),pilot=c.CASE_ROOT/'first.json',**kw)
        assert '--hold' not in cmd and '--no-requeue' in cmd
        assert '--qos=embers' in cmd and any(s.endswith('/qomp148.sbatch') for s in cmd)
        assert '--cpus-per-task='+('4' if phase=='audit' else '8') in cmd
        assert '--mem='+('24' if phase=='audit' else '32')+'G' in cmd
        assert '--time='+('02:00:00' if phase=='science' else '00:45:00') in cmd
    with pytest.raises(ValueError):
        c.phase_command(case,source=source,commit='c'*40,phase='wave',run=tmp_path,pilot=tmp_path)
    with pytest.raises(ValueError):
        c.phase_command(case,source=source,commit='c'*40,phase='worker',
            run=c.RUNS/'ccasp-native-s66-qomp148-worker001',pilot=tmp_path)
    assert c.CAMPAIGN_ROOT.name=='ccasp-native-s66-qomp-production148-003'


@pytest.mark.parametrize('primary',[None,RuntimeError('child failure'),KeyboardInterrupt(),SystemExit(143)])
@pytest.mark.parametrize('post_fails',[False,True])
def test_all_seals_precede_publication_and_primary_failure_survives(monkeypatch,tmp_path,primary,post_fails):
    e=load('qomp_execution')
    original=e.helper
    writer=load('phoenix_descriptor_bank_cb63305')
    controls=NS(qomp_enter=lambda:None)
    monkeypatch.setattr(e,'helper',lambda name:
        NS(qomp_engine=lambda:controls,allocation=lambda:{}) if name=='phoenix_installed_checks'
        else NS(validate_paths=lambda *a:None,load_checkpoint=lambda *a:None) if name=='qomp_expansion'
        else original(name))
    monkeypatch.setattr(e,'context',lambda _: {})
    monkeypatch.setattr(e,'SOURCE',Path(e.__file__).resolve().parents[2])
    monkeypatch.setattr(e,'git',lambda source,*args: '' if args[0]=='status' else 'c'*40)
    monkeypatch.setattr(e,'source_changes',lambda *args:[])
    monkeypatch.setattr(e,'qualified',lambda:NS(SCIENTIFIC_COMMIT='f'*40,verify_descriptor_operations=lambda *a:None))
    calls=[]
    def seal():
        calls.append('seal')
        if len(calls)==2 and post_fails: raise ValueError('postseal failure')
    monkeypatch.setattr(e,'runtime_check',seal)
    out=tmp_path/'result.json'
    def perform(args):
        if primary is not None: raise primary
        return out,dict(status='TEST_ONLY')
    args=NS(producer='qomp',campaign_scope='qomp-serial',remaining_case=True,commit='c'*40,run=tmp_path)
    if primary is not None or post_fails:
        with pytest.raises(type(primary) if primary is not None else ValueError) as caught:
            e.run_phase('worker',args,perform)
        if primary is not None: assert caught.value is primary
        assert not out.exists()
    else:
        e.run_phase('worker',args,perform)
        assert json.loads(out.read_bytes())['status']=='TEST_ONLY'
    assert calls==['seal','seal']


def test_live_quota_callback_reads_raw_owner_group_and_current_device(monkeypatch,tmp_path):
    e=load('qomp_execution'); original=e.helper
    root=tmp_path/'root';root.mkdir(mode=0o700)
    scratch=root/'attempt';scratch.mkdir(mode=0o700)
    policy=e.policy()
    monkeypatch.setattr(policy,'SCRATCH',str(root))
    monkeypatch.setattr(e,'policy',lambda:policy)
    monkeypatch.setattr(e.os,'statvfs',lambda _:NS(f_bavail=10**9,f_frsize=4096,f_favail=10**9))
    commands=[]
    def query(cmd,**kw):
        commands.append(cmd)
        return NS(stdout=str(root)+' 0 0 0 - 0 0 0 -\n')
    monkeypatch.setattr(e.subprocess,'run',query)
    result=e.scratch_admission(scratch,policy.execution_profile('s20'))
    assert result['required_files']==2*65536
    assert [c[2] for c in commands]==['-u','-g']
    assert all(cmd[-1]==str(root) for cmd in commands)
    assert all(cmd[:2]==['/usr/bin/lfs','quota'] for cmd in commands)


def test_launch_source_is_executable_bash_and_science_tree_not_overlaid():
    p=Path(__file__).with_name('qomp_launcher.sh')
    assert p.stat().st_mode&0o111
    for name in ('qomp148.sbatch','qomp_launcher.sh'):
        r=subprocess.run(['/bin/bash','-n',str(p.with_name(name))],capture_output=True)
        assert r.returncode==0
    raw=p.read_text()
    assert '-I -S -B' in raw and '/miniconda/envs/p4qca/bin/python' in raw
    assert 'PYTHONPATH=' not in raw


def launcher_fixture(monkeypatch,tmp_path,*,interrupt=False,post_fails=False,code=0):
    e=load('qomp_execution')
    c=e.helper('qomp_campaign')
    monkeypatch.setattr(c,'RUNS',tmp_path)
    view=tmp_path/'view.json';view.write_bytes(b'SYNTHETIC_VIEW')
    monkeypatch.setattr(c,'VIEW',view)
    source=Path(e.__file__).resolve().parents[2]
    monkeypatch.setattr(e,'SOURCE',source)
    monkeypatch.setattr(e,'git',lambda source,*args:'' if args[0]=='status' else 'c'*40)
    scratch=tmp_path/'ccasp-native-unit'
    scratch.mkdir(mode=0o700)
    for name in ('cwd','exchange','home','cache'):
        (scratch/name).mkdir(mode=0o700)
    incoming=b'\x00\x80SIGNED_ZERO_AND_SUBNORMAL_BYTES\x00'
    outgoing=b'\x80\x00EXACT_OUTPUT_BYTES\xff'
    original=scratch/'exchange/data.msgpack';original.write_bytes(incoming)
    run=tmp_path/'run';run.mkdir(mode=0o700);(run/'traces').mkdir(mode=0o700)
    profile=e.policy().execution_profile('s20')
    binding=dict(schema='qomp-ordinary-cli-binding/1',source=str(source),commit='c'*40,
        runtime_view=str(view),scratch=str(scratch),trace=str(run/'traces'),producer='qomp',
        profile=profile,phase='science',job='123')
    raw=c.encode(binding);cfg=run/'qomp-launch.json';cfg.write_bytes(raw)
    monkeypatch.setenv('CCASP_QOMP_BINDING',str(cfg))
    monkeypatch.setenv('CCASP_QOMP_BINDING_SHA256',c.sha(raw))
    monkeypatch.chdir(scratch/'exchange')
    events=[]
    def admitted(raw,argv,**kw):
        assert raw==b'SYNTHETIC_VIEW' and kw['producer']=='qomp' and kw['execution_profile']==profile
        assert (scratch/'cwd/data.msgpack').read_bytes()==incoming
        events.append('admitted')
        return dict(command=['/not-executed/private-python','exact-arg'],
                    environment=dict(ONLY='approved'),cwd=str(scratch/'cwd'))
    def post(raw):
        events.append('postseal')
        if post_fails: raise ValueError('postseal failure')
    rt=NS(ordinary_cli_launch=admitted,verify_runtime_view=post)
    q=e.qualified()
    monkeypatch.setattr(e,'qualified',lambda:NS(decode=q.decode,runtime=lambda:rt))
    class Child:
        pid=777
        def __init__(self,command,**kw):
            assert command[:2]==['/usr/bin/strace','-f']
            assert command[-2:]==['/not-executed/private-python','exact-arg']
            assert kw==dict(cwd=str(scratch/'cwd'),env=dict(ONLY='approved'),start_new_session=True)
            events.append('spawn')
            self.waits=0
        def __enter__(self): return self
        def __exit__(self,*exc): events.append('reaped')
        def wait(self,timeout=None):
            self.waits+=1;events.append(('wait',timeout))
            if interrupt and self.waits==1: raise KeyboardInterrupt('synthetic child signal')
            (scratch/'cwd/data.msgpack').write_bytes(outgoing)
            return code
    monkeypatch.setattr(e.subprocess,'Popen',Child)
    monkeypatch.setattr(e.os,'killpg',lambda pid,sig:events.append(('signal',pid,sig)))
    argv=['--scratch',str(scratch/'task'),'--nthread','8','--memory','20.0GiB','--qcschema','data.msgpack']
    return e,argv,original,incoming,outgoing,events


def test_actual_launcher_executes_admitted_argv_env_cwd_and_exact_bytes(monkeypatch,tmp_path):
    e,argv,path,incoming,outgoing,events=launcher_fixture(monkeypatch,tmp_path)
    assert e.launcher(argv)==0
    assert path.read_bytes()==outgoing and incoming!=outgoing
    assert events==['admitted','spawn',('wait',None),'reaped','postseal']


@pytest.mark.parametrize('interrupt',[False,True])
def test_launcher_nonzero_signal_and_postseal_do_not_publish_a_successful_exchange(monkeypatch,tmp_path,interrupt):
    e,argv,path,incoming,_,events=launcher_fixture(monkeypatch,tmp_path,
        interrupt=interrupt,post_fails=True,code=9)
    with pytest.raises(KeyboardInterrupt if interrupt else subprocess.CalledProcessError) as caught:
        e.launcher(argv)
    assert path.read_bytes()==incoming
    assert events[-1]=='postseal'
    assert any('postseal failure' in note for note in caught.value.__notes__)
    if interrupt:
        assert ('signal',777,signal.SIGTERM) in events and ('wait',60) in events


def test_launcher_rejects_unowned_cwd_before_any_child(monkeypatch,tmp_path):
    e,argv,_,_,_,events=launcher_fixture(monkeypatch,tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='exchange directory'):
        e.launcher(argv)
    assert not events


def test_launcher_final_seal_precedes_successful_exchange(monkeypatch,tmp_path):
    e,argv,path,incoming,_,_=launcher_fixture(monkeypatch,tmp_path,post_fails=True)
    with pytest.raises(ValueError,match='postseal failure'):
        e.launcher(argv)
    assert path.read_bytes()==incoming


@pytest.mark.parametrize('field',['source','view','old_profile','digest'])
def test_launcher_refuses_foreign_roles_before_child(monkeypatch,tmp_path,field):
    e,argv,path,incoming,_,events=launcher_fixture(monkeypatch,tmp_path)
    cfg=Path(e.os.environ['CCASP_QOMP_BINDING'])
    c=e.helper('qomp_campaign')
    value=json.loads(cfg.read_bytes())
    if field=='source': value['source']='/foreign/operations'
    elif field=='view': value['runtime_view']='/foreign/runtime-view.json'
    elif field=='old_profile': value['profile']=load('grid400_bank_policy').execution_profile('s20')
    raw=c.encode(value);cfg.write_bytes(raw)
    monkeypatch.setenv('CCASP_QOMP_BINDING_SHA256',c.sha(raw) if field!='digest' else 'f'*64)
    with pytest.raises(ValueError):
        e.launcher(argv)
    assert path.read_bytes()==incoming and not events


def test_worker_preflight_must_bind_current_operational_revision(monkeypatch):
    _,_,_,_,cases,_=synthetic(monkeypatch)
    draft=next(d for d in cases.values() if d['dataset_mode']=='fresh')
    e=load('qomp_execution')
    sources={'real-worker-helper.py':'d'*64};transport={'synthetic':'not a production binding'}
    report=dict(status='PASS_QOMP_FRESH_WORKER_PREFLIGHT_ONLY',
        producer_profile='qomp',campaign_scope='qomp-pilot',dataset_mode='fresh',
        runtime_unchanged=True,scientific_build_commit=e.qualified().SCIENTIFIC_COMMIT,
        producer_binding_sha256='a'*64,qualification_sha256='b'*64,pilot_sha256='e'*64,
        runtime_input_sha256=draft['runtime_input_sha256'],monomer_id=draft['monomer_id'],
        execution_profile=draft['execution_profile'],operational_sources_sha256=sources,
        operational_commit='c'*40,transport_binding=transport,
        task_config=dict(ncores=8,memory_GiB=20,internal_retries=0))
    args=dict(binding_sha='a'*64,qualification_sha='b'*64,pilot_sha='e'*64,
        worker_sources=sources,transport=transport,remaining_case=False,
        campaign_scope='qomp-pilot',operational_commit='c'*40)
    e.validate_preflight(report,draft,**args)
    for field,value in (
            ('operational_commit','f'*40),('transport_binding',{}),
            ('operational_sources_sha256',{}),('runtime_input_sha256','f'*64),
            ('task_config',dict(ncores=8,memory_GiB=20,internal_retries=1)),
            ('scientific_build_commit','f'*40),('runtime_unchanged',False)):
        with pytest.raises(ValueError):
            e.validate_preflight(report|{field:value},draft,**args)


@pytest.mark.parametrize('failure',[None,'child','signal','child_and_cleanup'])
def test_real_qomp_session_callbacks_factory_and_delivery_unwind(monkeypatch,tmp_path,failure):
    exercise_session_callbacks(monkeypatch,tmp_path,failure,serial=False)


def exercise_session_callbacks(monkeypatch,tmp_path,failure,*,serial):
    """Shared callback oracle; serial SDK binding is exercised in p4qca tests."""
    c,_,_,_,cases,_=synthetic(monkeypatch)
    draft=next(d for d in cases.values() if d['dataset_mode']=='fresh')
    binding=None
    execution=None
    if serial:
        core,exp,_,cases,context,checkpoint,proofs,authority,_=load('test_qomp_expansion').synthetic(monkeypatch)
        exp._MODULES['audited_bank']=core
        raw=core.encode(checkpoint)
        binding=dict(context_raw=context,context_sha256=core.digest(context),
            checkpoint_raw=raw,checkpoint_sha256=core.digest(raw),proof_blobs=proofs,qomp_binding=authority)
        mid=json.loads(context)['eligible_ids'][1]
        draft=core.bind_next_case(cases[mid],raw,core.digest(raw),context,core.digest(context),
                                  proofs,**exp.reducer_options(authority))
        c=exp.helper('qomp_campaign')
        execution=load('qomp_execution')
        execution._MODULES.update(qomp_expansion=exp,qomp_campaign=c)
    session=load('single_record_cb63305')
    runtime=c.helper('phoenix_worker_probe').runtime_input(draft)
    runtime['specification']['protocols']=session.helper('phoenix_pilot').execution_specification({})['protocols']
    tag=c.NAMESPACE+draft['monomer_id']+'-123'
    config=NS(executors=dict(pilot=NS(type='local',max_workers=1,cores_per_worker=8,
                                     memory_per_worker=20.,compute_tags=[tag])))
    effects=[];saved={};active=[False]
    primary=RuntimeError('synthetic child failure')
    class Client:
        def deactivate(self,**kwargs): effects.append('deactivate')
    @contextmanager
    def factory():
        active[0]=True;effects.append('factory-enter')
        try: yield Client
        finally: active[0]=False;effects.append('factory-exit')
    class Manager:
        def __init__(self,config):
            assert active[0]
            self.manager_config=config;self.name='SYNTHETIC_QOMP_MANAGER'
            self.client=Client()
            def cleanup():
                effects.append('cleanup')
                if failure=='child_and_cleanup': raise ValueError('synthetic cleanup failure')
            self.dflow_kernel=NS(cleanup=cleanup)
            self.statistics=NS(active_tasks=0,active_cores=0,active_memory=0.,total_cpu_hours=0.,
                               dict=lambda:dict(total_successful_tasks=1,total_failed_tasks=0,
                                                total_rejected_tasks=0,active_tasks=0))
        def stop(self): effects.append('stop')
        def start(self):
            assert not active[0]
            task=NS(id=77,record_id=9999,compute_tag=tag,function='qcengine.compute',
                required_programs={'psi4':'synthetic'},function_kwargs=dict(program='psi4',
                    input_data=dict(schema_name='qcschema_input',schema_version=1,
                        molecule=runtime['molecule'],**runtime['specification'])))
            self.preprocess_new_tasks([task])
            self.target_task_id=77
            if failure in ('child','child_and_cleanup'): raise primary
            if failure=='signal':
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                return
            self.postprocess_results({77:NS(result_compressed=b'SYNTHETIC_COMPRESSED',
                result=dict(extras={'psi4:native_monomer':{'synthetic':True}}))})
            self.delivery_outcome='accepted'
    package=ModuleType('qcfractalcompute');package.__path__=[]
    module=ModuleType('qcfractalcompute.compute_manager');module.ComputeManager=Manager
    monkeypatch.setitem(sys.modules,package.__name__,package)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    original=session.helper
    monkeypatch.setattr(session,'helper',lambda name:
        NS(grid400_manager_client_factory=factory) if name=='dataset_fresh_cb63305'
        else execution if name=='qomp_execution' and serial else original(name))
    kwargs=dict(record_id=9999,compute_tag=tag,runtime=runtime,payload=tmp_path,
        save=lambda name,data:saved.setdefault(name,data),worker_shape=(8,20.),
        producer='qomp',case=draft,campaign_scope='qomp-serial' if serial else 'qomp-pilot',
        **(dict(checkpoint_binding=binding) if serial else {}))
    previous=signal.getsignal(signal.SIGTERM)
    if failure:
        with pytest.raises(RuntimeError) as caught:
            session.run_session(config,**kwargs)
        if failure.startswith('child'):
            assert caught.value is primary
    else:
        result=session.run_session(config,**kwargs)
        assert result==dict(manager='SYNTHETIC_QOMP_MANAGER',task_id=77,
                           local_native={'synthetic':True})
        assert (tmp_path/'local-result-77.zstd').read_bytes()==b'SYNTHETIC_COMPRESSED'
    assert effects[:2]==['factory-enter','factory-exit']
    assert effects[-2:]==['deactivate','cleanup']
    assert saved['claimed-input.json']['record_id']==9999
    if failure=='child_and_cleanup':
        assert 'manager-delivery.json' not in saved
        assert any('synthetic cleanup failure' in n for n in primary.__notes__)
    else:
        assert saved['manager-delivery.json']['interrupted'] is (failure=='signal')
    assert signal.getsignal(signal.SIGTERM)==previous


def test_original_snapshot_association_and_provenance_refusals(monkeypatch):
    c,_,_,anchors,_,carry=synthetic(monkeypatch)
    checkpoint=json.loads(anchors['checkpoint'])
    baseline=checkpoint['snapshot']
    baseline.update(metadata=dict(id=357,name=carry['original_dataset_name'],
        owner_user='awallace43',owner_group=None,provenance={'ORIGINAL':'not-QOMP'}),
        entries={r['entry_name']:dict(synthetic=True) for r in carry['records']},
        associations={r['entry_name']:{'ORIGINAL_SPEC':r['record_id']} for r in carry['records']},
        specifications={'ORIGINAL_SPEC':dict(unchanged=True)})
    live=copy.deepcopy(baseline)
    for i in range(5234353,5234361):
        live['records'][str(i)]=dict(status='waiting')
        live['entries']['waiting-'+str(i)]=dict(synthetic=True)
        live['associations']['waiting-'+str(i)]={'ORIGINAL_SPEC':i}
    updated=anchors|dict(checkpoint=c.encode(checkpoint))
    state=json.loads(anchors['state'])
    state['bank_checkpoint_sha256']=c.sha(updated['checkpoint'])
    updated['state']=c.encode(state)
    monkeypatch.setattr(c,'ORIGINAL_SHA',{k:c.sha(v) for k,v in updated.items()})
    monkeypatch.setattr(c,'anchors',lambda:updated)
    e=load('qomp_execution');original=e.helper
    monkeypatch.setattr(e,'helper',lambda name:c if name=='qomp_campaign' else original(name))
    assert e.validate_original_snapshot(live)==live
    for field in ('provenance','association','waiting'):
        other=copy.deepcopy(live)
        if field=='provenance': other['metadata']['provenance']={'changed':True}
        elif field=='association':
            other['associations'][carry['records'][0]['entry_name']]={'ORIGINAL_SPEC':9999}
        else: other['records']['5234353']['status']='complete'
        with pytest.raises(ValueError):
            e.validate_original_snapshot(other)


def test_durable_intent_blocks_new_tag_retry_and_changed_case(monkeypatch,tmp_path):
    c,_,_,_,cases,_=synthetic(monkeypatch)
    case=next(d for d in cases.values() if d['dataset_mode']=='fresh')
    e=load('qomp_execution')
    monkeypatch.setattr(e,'INTENT_ROOT',tmp_path/'intents')
    tag=c.NAMESPACE+case['monomer_id']+'-123'
    digest=e.reserve_transaction(case,tag)
    path=e.INTENT_ROOT/(case['eligibility_sha256']+'.json')
    original=path.read_bytes()
    assert c.sha(original)==digest and e.INTENT_ROOT.stat().st_mode&0o777==0o700
    for other in (case,case|dict(carryforward_sha256='f'*64)):
        with pytest.raises(FileExistsError):
            e.reserve_transaction(other,tag[:-3]+'124')
    assert path.read_bytes()==original


@pytest.mark.parametrize('argument',['--module','--version'])
def test_literal_metadata_discovery_does_not_enter_full_runtime_seal(monkeypatch,tmp_path,argument):
    e,_,path,incoming,_,events=launcher_fixture(monkeypatch,tmp_path)
    q=e.qualified()
    def full(*args,**kwargs):
        raise subprocess.TimeoutExpired(['owned-launcher',argument],30)
    def metadata(raw,argv,**kwargs):
        assert argv==[argument] and kwargs['producer']=='qomp'
        events.append('metadata-admitted')
        return dict(command=['/not-executed/private-python','exact-arg'],
                    environment=dict(ONLY='approved'),cwd=str(path.parent.parent/'cwd'))
    rt=NS(ordinary_cli_launch=full,verify_runtime_view=full,
        metadata_cli_launch=metadata,verify_metadata_runtime=lambda raw:events.append('metadata-post'))
    monkeypatch.setattr(e,'qualified',lambda:NS(decode=q.decode,runtime=lambda:rt))
    assert e.launcher([argument])==0
    assert path.read_bytes()==incoming
    assert events==['metadata-admitted','spawn',('wait',None),'reaped','metadata-post']


@pytest.mark.parametrize('argv',[
    ['--module','--qcschema','data.msgpack'],['--version','--module'],
    ['--module','--memory','20.0GiB'],['--version','--nthread','8']])
def test_mixed_metadata_arguments_cannot_select_fast_discovery(monkeypatch,tmp_path,argv):
    e,_,_,_,_,events=launcher_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='cannot be combined'):
        e.launcher(argv)
    assert not events


def metadata_runtime_fixture(monkeypatch,tmp_path):
    f=load('test_qualified_qomp');rt=f.rt
    base,prefix,view=f.fake_runtime(monkeypatch,tmp_path)
    def directory(root,name,entries):
        p=root/name;p.mkdir(parents=True,exist_ok=True)
        for part in (p,*p.parents):
            if part==root: break
            entries[str(part.relative_to(root))]=dict(kind='directory')
    pe,be=(view['scopes'][k]['entries'] for k in ('prefix','base'))
    for name in ('bin','lib/python3.13/site-packages/psi4'):
        directory(prefix,name,pe)
    for name in ('bin','lib/python3.13/lib-dynload'):
        directory(base,name,be)
    for root,entries,names in (
        (prefix,pe,('bin/python','bin/python3','bin/python3.13','bin/psi4',
                    'lib/python3.13/site-packages/psi4/metadata.py','science-large.bin')),
        (base,be,('bin/python3.13',))):
        for name in names:
            raw=('SYNTHETIC '+name).encode();(root/name).write_bytes(raw)
            entries[name]=dict(kind='file',sha256=f.sha(raw))
    (prefix/'lib64').symlink_to('lib',target_is_directory=True)
    pe['lib64']=dict(kind='link',target='lib',resolved=str(prefix/'lib'),target_kind='directory')
    return rt,base,prefix,view


def test_metadata_content_selection_is_small_but_full_molecular_seal_unchanged(monkeypatch,tmp_path):
    rt,base,prefix,view=metadata_runtime_fixture(monkeypatch,tmp_path)
    actual=rt.file_sha;hashed=[]
    def digest(path):
        hashed.append(str(path))
        return actual(path)
    monkeypatch.setattr(rt,'file_sha',digest)
    rt.verify_metadata_runtime(b'synthetic')
    assert set(hashed)=={str(prefix/n) for n in (
        'bin/python','bin/python3','bin/python3.13','bin/psi4','pyvenv.cfg',
        'lib/python3.13/site-packages/psi4/metadata.py')}|{str(base/'bin/python3.13')}
    (prefix/'science-large.bin').write_bytes(b'changed')
    rt.verify_metadata_runtime(b'synthetic') # Full outer seals, NOT metadata, own science hashes.
    with pytest.raises(ValueError,match='content'):
        rt.verify_runtime_view(b'synthetic')


@pytest.mark.parametrize('change',[
    'script','version','interpreter','cfg','pth','zip_file','zip_dir','zip_link',
    'lib64','stdlib_extra'])
def test_metadata_startup_guards_remain_fail_closed(monkeypatch,tmp_path,change):
    rt,base,prefix,_=metadata_runtime_fixture(monkeypatch,tmp_path)
    if change in ('script','version','interpreter','cfg'):
        name={'script':'bin/psi4','version':'lib/python3.13/site-packages/psi4/metadata.py',
              'interpreter':'bin/python','cfg':'pyvenv.cfg'}[change]
        (prefix/name).write_bytes(b'tampered')
    elif change=='pth': (prefix/'evil.pth').write_bytes(b'import evil')
    elif change.startswith('zip_'):
        p=base/'lib/python313.zip'
        if change=='zip_file': p.write_bytes(b'shadow')
        elif change=='zip_dir': p.mkdir()
        else: p.symlink_to('missing')
    elif change=='lib64':
        (prefix/'lib64').unlink();(prefix/'lib64').symlink_to('bin',target_is_directory=True)
    else: (base/'lib/python3.13/extra.py').write_bytes(b'')
    with pytest.raises(ValueError):
        rt.verify_metadata_runtime(b'synthetic')


@pytest.mark.parametrize('argv',[['--module','--qcschema'],['--version','--module'],
    ['--help'],[],('--module',)])
def test_metadata_admission_has_no_arbitrary_skip_option(argv):
    rt=load('runtime_qomp')
    with pytest.raises(ValueError,match='literal'):
        rt.metadata_cli_launch(b'not-a-view',argv,scratch=Path('/never/read'),
            execution_profile={},inherited_threads={},producer='qomp')


@pytest.mark.parametrize('argv',[['--module'],['--version']])
def test_metadata_launch_retains_actual_command_env_profile_cache_gate(monkeypatch,tmp_path,argv):
    rt=load('runtime_qomp')
    scratch=tmp_path/'ccasp-native-discovery';scratch.mkdir(mode=0o700)
    for name in ('home','cache','cwd'): (scratch/name).mkdir(mode=0o700)
    monkeypatch.setattr(rt,'SCRATCH_ROOT',tmp_path)
    seen=[]
    monkeypatch.setattr(rt,'verify_metadata_runtime',lambda raw:seen.append(raw))
    monkeypatch.setattr(rt,'verify_runtime_view',lambda raw:pytest.fail('full seal in metadata'))
    profile=load('qomp_bank_policy').execution_profile('s20')
    threads=dict(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
    admitted=rt.metadata_cli_launch(b'checked',argv,scratch=scratch,producer='qomp',
        execution_profile=profile,inherited_threads=threads)
    assert seen==[b'checked']
    assert admitted['command']==[rt.PREFIX+'/bin/python','-I','-B','-X',
        'pycache_prefix='+str(scratch/'cache'),rt.PREFIX+'/bin/psi4',*argv]
    assert admitted['cwd']==str(scratch/'cwd')
    assert admitted['environment']['LD_LIBRARY_PATH']==rt.PREFIX+'/lib:'+rt.BASE+'/lib'
    with pytest.raises(ValueError,match='literal'):
        rt.metadata_cli_launch(b'checked',argv,scratch=scratch,producer='grid400',
            execution_profile=profile,inherited_threads=threads)
    (scratch/'cache/old.pyc').write_bytes(b'shadow')
    with pytest.raises(ValueError,match='preexisting'):
        rt.metadata_cli_launch(b'checked',argv,scratch=scratch,producer='qomp',
            execution_profile=profile,inherited_threads=threads)


def test_frozen_cli_metadata_exits_before_native_import():
    import ast
    path=Path(__file__).resolve().parents[2]/'psi4/run_psi4.py'
    tree=ast.parse(path.read_bytes())
    native=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Import)
        and any(a.name=='psi4' for a in n.names))
    for name in ('module','version'):
        branch=next(n for n in tree.body if isinstance(n,ast.If)
            and isinstance(n.test,ast.Subscript) and isinstance(n.test.value,ast.Name)
            and n.test.value.id=='args' and isinstance(n.test.slice,ast.Constant)
            and n.test.slice.value==name)
        assert branch.lineno<native
        assert isinstance(branch.body[-1],ast.Expr)
        assert ast.unparse(branch.body[-1].value)=='sys.exit()'


@pytest.mark.parametrize('interrupt',[False,True])
def test_metadata_signal_child_and_postcheck_failure_cannot_succeed(monkeypatch,tmp_path,interrupt):
    e,_,path,incoming,_,events=launcher_fixture(monkeypatch,tmp_path,interrupt=interrupt,code=9)
    q=e.qualified()
    def metadata(raw,argv,**kwargs):
        return dict(command=['/not-executed/private-python','exact-arg'],
                    environment=dict(ONLY='approved'),cwd=str(path.parent.parent/'cwd'))
    def post(raw): raise ValueError('metadata postcheck failed')
    rt=NS(metadata_cli_launch=metadata,verify_metadata_runtime=post)
    monkeypatch.setattr(e,'qualified',lambda:NS(decode=q.decode,runtime=lambda:rt))
    with pytest.raises(KeyboardInterrupt if interrupt else subprocess.CalledProcessError) as caught:
        e.launcher(['--module'])
    assert path.read_bytes()==incoming
    assert any('metadata postcheck failed' in n for n in caught.value.__notes__)
    if interrupt:
        assert ('signal',777,signal.SIGTERM) in events and ('wait',60) in events
