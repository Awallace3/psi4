"""Real reviewed SDK/fake HTTP QOMP wave controls; never a server/solver."""
import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from qcportal.metadata_models import InsertMetadata,InsertCountsMetadata
from qcportal.singlepoint import SinglepointDatasetNewEntry
from qcportal.dataset_models import DatasetSubmitBody

from importlib.util import spec_from_file_location,module_from_spec


def load(name):
    path=Path(__file__).with_name(name+'.py')
    spec=spec_from_file_location('_qomp_wave_http_'+name,path)
    module=module_from_spec(spec)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def transaction(monkeypatch,tmp_path,*,readonly=False):
    f=load('test_qomp_wave').completed_fixture(monkeypatch,tmp_path,2)
    transport=load('bank_wave_transport')
    transport._MODULES['bank_wave']=f.wave
    campaign=transport.helper('bank_wave_campaign')
    campaign._MODULES.update(bank_wave=f.wave,qomp_wave_policy=f.policy)
    monkeypatch.setattr(campaign,'revalidate_binding',lambda binding:None)
    binding=copy.deepcopy(f.binding)
    mid=f.plan['member_ids'][0]
    binding.update(monomer_id=mid)
    members={m:{k:s[k] for k in ('science_job_id','compute_tag','case_sha256',
        'worker_preflight_sha256')} for m,s in f.reg['members'].items()}
    if readonly:
        scope=None
        cls=transport.readonly_client_class(binding,monomer_id=mid)
    else:
        binding.update(registration=None,release_permit=None)
        monkeypatch.setenv('SLURM_JOB_ID','8000')
        scope=transport.WaveScope(binding,f.cases,members,f.reg['registration_role'])
        scope.select(mid)
        cls=transport.client_class(scope)
    client=object.__new__(cls)
    legacy=load('transport_cb63305_tests').legacy_fixtures
    replies=[(200,asdict(InsertMetadata(inserted_idx=[0]))),
             (200,asdict(InsertCountsMetadata(n_inserted=1,n_existing=0)))]
    client.__dict__.update(legacy.client(replies,adapted=False).__dict__)
    return f,transport,scope,client


def test_qomp_typed_registration_posts_only_bound_entry_and_one_record(monkeypatch,tmp_path):
    f,transport,scope,client=transaction(monkeypatch,tmp_path)
    mid=f.plan['member_ids'][0];case=f.cases[mid]
    base=f"api/v1/datasets/singlepoint/{f.plan['dataset_id']}/"
    meta=client.make_request('post',base+'entries/bulkCreate',InsertMetadata,
        body=[SinglepointDatasetNewEntry(**case['entry'])])
    assert meta.n_inserted==1 and meta.n_existing==0
    meta=client.make_request('post',base+'submit',InsertCountsMetadata,
        body=DatasetSubmitBody(entry_names=[case['entry']['name']],
            specification_names=[case['specification_name']],
            compute_tag=f.reg['members'][mid]['compute_tag'],find_existing=False))
    assert meta.n_inserted==1 and meta.n_existing==0
    scope.finish(mid)
    assert len(client._req_session.sent)==2
    with pytest.raises(ValueError):
        client.make_request('post',base+'submit',InsertCountsMetadata,
            body=DatasetSubmitBody(entry_names=[case['entry']['name']],
                specification_names=[case['specification_name']],find_existing=False,
                compute_tag=f.reg['members'][mid]['compute_tag']))
    assert len(client._req_session.sent)==2


@pytest.mark.parametrize('readonly',[False,True])
@pytest.mark.parametrize('endpoint',[
    'api/v1/datasets/singlepoint','api/v1/datasets/singlepoint/357/entries/bulkCreate',
    'api/v1/datasets/singlepoint/9999/specifications','api/v1/records/reset',
    'api/v1/datasets/singlepoint/1/submit'])
def test_foreign_or_unapproved_mutations_never_reach_http(monkeypatch,tmp_path,readonly,endpoint):
    _,_,_,client=transaction(monkeypatch,tmp_path,readonly=readonly)
    with pytest.raises(ValueError):
        client.make_request('post',endpoint,dict,body={})
    assert not client._req_session.sent


@pytest.mark.parametrize('bad',['tag','existing','runtime'])
def test_wrong_scientific_payload_or_tag_consumes_attempt_before_http(monkeypatch,tmp_path,bad):
    f,_,scope,client=transaction(monkeypatch,tmp_path)
    mid=f.plan['member_ids'][0];case=f.cases[mid]
    base=f"api/v1/datasets/singlepoint/{f.plan['dataset_id']}/"
    if bad=='runtime':
        body=[SinglepointDatasetNewEntry(**copy.deepcopy(case['entry']))]
        body[0].additional_keywords['function_kwargs']['native_monomer']['resources']['max_work']+=1
        role='entries/bulkCreate';model=InsertMetadata
    else:
        scope.request('post',base+'entries/bulkCreate')
        body=DatasetSubmitBody(entry_names=[case['entry']['name']],
            specification_names=[case['specification_name']],find_existing=bad=='existing',
            compute_tag='foreign' if bad=='tag' else f.reg['members'][mid]['compute_tag'])
        role='submit';model=InsertCountsMetadata
    with pytest.raises(ValueError):
        client.make_request('post',base+role,model,body=body)
    with pytest.raises(ValueError,match='already attempted'):
        client.make_request('post',base+role,model,body=body)
    assert not client._req_session.sent


@pytest.mark.parametrize('bad',[None,'wildcard','unrelated','duplicate','own'])
def test_only_validated_active_wave_peer_tags_are_authorized(monkeypatch,tmp_path,bad):
    f=load('test_qomp_wave').completed_fixture(monkeypatch,tmp_path,2)
    execution=load('qomp_execution')
    runtime=execution.helper('bank_wave_runtime')
    runtime._MODULES['bank_wave']=f.wave
    monkeypatch.setattr(runtime.helper('bank_wave_campaign'),'revalidate_binding',lambda binding:None)
    own,peer=[f.reg['members'][mid]['compute_tag'] for mid in f.plan['member_ids']]
    tags=[peer]
    if bad=='wildcard': tags=['*']
    elif bad=='unrelated': tags=['ccasp-native-qomp148-monomer-0131-99999']
    elif bad=='own': tags=[own]
    managers=[NS(compute_tags=tags)]
    if bad=='duplicate': managers*=2
    if bad is None:
        execution.manager_isolation(managers,own,wave_binding=f.binding)
    else:
        with pytest.raises(ValueError):
            execution.manager_isolation(managers,own,wave_binding=f.binding)


@pytest.mark.parametrize('failure',[None,'child','signal','child_and_cleanup'])
def test_registered_own_record_claim_ack_delivery_and_signal_unwind(monkeypatch,tmp_path,failure):
    f=load('test_qomp_wave').completed_fixture(monkeypatch,tmp_path,2)
    f.binding['monomer_id']=f.plan['member_ids'][0]
    load('test_qomp148').exercise_session_callbacks(
        monkeypatch,tmp_path,failure,serial=False,wave_fixture=f)


@pytest.mark.parametrize('post_fails',[False,True])
def test_full_typed_registration_publishes_only_after_postseal(monkeypatch,tmp_path,post_fails):
    f,transport,scope,client=transaction(monkeypatch,tmp_path)
    # register_once owns slot selection; transaction() selected its first slot
    # only for the standalone one-entry transport tests.
    scope.active=None
    controller=load('bank_wave_controller')
    controller._MODULES.update(bank_wave=f.wave,bank_wave_transport=transport,
        bank_wave_campaign=transport.helper('bank_wave_campaign'),bank_store=f.wave.helper('bank_store'))
    store=controller.helper('bank_store')
    monkeypatch.setattr(store,'QOMP_INTENTS',tmp_path/'registration-intents')
    q=controller.helper('qomp_execution')
    monkeypatch.setattr(q,'live_originals',lambda client:{'SYNTHETIC':True})
    monkeypatch.setattr(client,'get_server_information',lambda:dict(name='CDS QCFractal Server',version='0.70'))
    monkeypatch.setattr(client,'query_managers',lambda **kw:[])
    legacy=load('transport_cb63305_tests').legacy_fixtures
    replies=[]
    for _ in f.plan['member_ids']:
        replies.extend([(200,asdict(InsertMetadata(inserted_idx=[0]))),
                        (200,asdict(InsertCountsMetadata(n_inserted=1,n_existing=0)))])
    client._req_session=legacy.MockSession(replies)
    observed=[f.previous['snapshot']]
    for mid in f.plan['member_ids']:
        views=f.captures[mid]
        observed.extend([views['before'],views['entry'],views['entry'],views['record']])
    observed.append(f.registered)
    snapshots=iter(observed)
    monkeypatch.setattr(transport,'_snapshot_dataset',lambda *a,**kw:copy.deepcopy(next(snapshots)))
    events=[];saved={}
    def save(name,value):
        events.append(name);saved[name]=value
    def post():
        events.append('POSTSEAL')
        if post_fails: raise ValueError('synthetic registration postseal')
    if post_fails:
        with pytest.raises(ValueError,match='postseal'):
            controller.register_once(client,scope,save=save,before_publish=post)
        assert 'registration.json' not in saved
    else:
        reg=controller.register_once(client,scope,save=save,before_publish=post)
        assert reg['dataset_id']==f.plan['dataset_id']
        assert events[-2:]==['POSTSEAL','registration.json']
        assert scope.finished==f.plan['member_ids']
    assert len(client._req_session.sent)==2*len(f.plan['member_ids'])
    assert 'registration/dispatch-intent.json' in saved
    with pytest.raises(FileExistsError):
        store._reserve_wave_intent(**controller.helper('bank_wave_campaign').intent_fields(
            scope.binding,scope.members,scope.registration_role))


@pytest.mark.parametrize('post_fails',[False,True])
def test_whole_wave_finalizer_postseal_precedes_proof_and_checkpoint_publication(
        monkeypatch,tmp_path,post_fails):
    f=load('test_qomp_wave').completed_fixture(monkeypatch,tmp_path,2)
    controller=load('bank_wave_controller')
    controller._MODULES['bank_wave']=f.wave
    campaign=controller.helper('bank_wave_campaign')
    monkeypatch.setattr(campaign,'revalidate_binding',lambda binding:None)
    transport=controller.helper('bank_wave_transport')
    monkeypatch.setattr(transport,'require_readonly_client',lambda *a,**kw:None)
    monkeypatch.setattr(transport,'snapshot_dataset',lambda *a,**kw:copy.deepcopy(f.final))
    store=controller.helper('bank_store')
    monkeypatch.setattr(store,'QOMP_WAVE_PROOFS',tmp_path/'wave-proofs')
    f.binding.update(monomer_id=f.plan['member_ids'][0],science_barrier=f.barrier,
        science_barrier_raw=f.wave.encode(f.barrier),
        science_barrier_sha256=f.wave.digest(f.wave.encode(f.barrier)))
    def write(path,raw):
        path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);path.write_bytes(raw)
    directory=f.binding['plan_path'].parent
    write(directory/'registration/registered-snapshot.json',f.wave.encode(f.registered))
    for mid in f.plan['member_ids']:
        sci=directory/'members'/mid/'science'
        audit=directory/'members'/mid/'audit'
        write(directory/'cases'/(mid+'.json'),f.wave.encode(f.cases[mid]))
        write(sci/'payload/qomp-outcome.json',f.outcomes[mid])
        write(sci/'payload/server-native-envelope.json',
              f.blobs[f.link['receipts_sha256'][mid]['envelope']])
        write(sci/'payload/owned-after-delivery.json',f.wave.encode(f.deliveries[mid]))
        write(sci/'accounting.txt',f.science_accounting[mid])
        write(audit/'payload/audit-report.json',f.audits[mid])
        write(audit/'accounting.txt',f.audit_accounting[mid])
        write(audit/'submission.json',f.wave.encode(dict(
            job_id=f.wave.decoded(f.audits[mid])['allocation']['JobId'],command=['SYNTHETIC_AUDIT_COMMAND'])))
        for role,value in f.captures[mid].items():
            write(Path(f.reg['members'][mid]['captures'][role]['path']),f.wave.encode(value))
    monkeypatch.setattr(controller,'phase_command',lambda *a:['SYNTHETIC_AUDIT_COMMAND'])
    allocation=dict(f.reg['registration_role']['allocation'],JobId='12000')
    events=[];saved={}
    original=store._store_wave_proofs
    def proofs(*a,**kw):
        events.append('STORE_PROOFS');return original(*a,**kw)
    monkeypatch.setattr(store,'_store_wave_proofs',proofs)
    def save(name,value): events.append(name);saved[name]=value
    def post():
        events.append('POSTSEAL')
        if post_fails: raise KeyboardInterrupt('synthetic finalizer TERM')
    if post_fails:
        with pytest.raises(KeyboardInterrupt):
            controller.publish_wave(NS(),f.binding,allocation=allocation,save=save,before_publish=post)
        assert 'bank-checkpoint.json' not in saved and 'wave-advance.json' not in saved
        assert 'STORE_PROOFS' not in events
    else:
        result=controller.publish_wave(NS(),f.binding,allocation=allocation,save=save,before_publish=post)
        assert result['generation']==3 and result['dataset_id']==f.plan['dataset_id']
        assert events.index('POSTSEAL')<events.index('STORE_PROOFS')<events.index('bank-checkpoint.json')
