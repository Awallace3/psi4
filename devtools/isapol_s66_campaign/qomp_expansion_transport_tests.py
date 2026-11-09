"""Synthetic QOMP append with real SDK/fake HTTP; never a server or solver."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from qcportal.metadata_models import InsertMetadata,InsertCountsMetadata

from importlib.util import spec_from_file_location,module_from_spec


def load(name):
    p=Path(__file__).with_name(name+'.py')
    s=spec_from_file_location('_qexp_http_'+name,p)
    m=module_from_spec(s);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
    return m


@pytest.mark.parametrize('failure',[None,'child','signal','child_and_cleanup'])
def test_serial_manager_callbacks_factory_delivery_and_unwind(monkeypatch,tmp_path,failure):
    # This positive path intentionally uses the reviewed0.63 SDK in p4qca.
    # No version spoof, live HTTP, real manager subprocess or native import.
    load('test_qomp148').exercise_session_callbacks(
        monkeypatch,tmp_path,failure,serial=True)


def transaction(monkeypatch,tmp_path,*,failure=None):
    fixtures=load('test_qomp_expansion')
    core,exp,_,cases,context,checkpoint,proofs,binding,delivered=fixtures.synthetic(monkeypatch)
    previous=core.encode(checkpoint)
    options=exp.reducer_options(binding)
    mid=json.loads(context)['eligible_ids'][1]
    case=core.bind_next_case(cases[mid],previous,core.digest(previous),
        context,core.digest(context),proofs,**options)
    receipts,captures,tensor=delivered(case,checkpoint['snapshot'])
    tag=json.loads(receipts['outcome'])['compute_tag']
    adapter=load('bank_dataset')
    transport=adapter.helper('bank_transport');original=transport.helper
    monkeypatch.setattr(transport,'helper',lambda n,_old=original:
                        core if n=='audited_bank' else _old(n))
    scope=adapter.prepare_transaction(case,copy.deepcopy(case),tag,producer='qomp',
        checkpoint_raw=previous,context_raw=context,proof_blobs=proofs,qomp_binding=binding)
    cls=adapter.client_class(scope)
    client=object.__new__(cls)
    old=load('transport_cb63305_tests')
    replies=[(200,asdict(InsertMetadata(inserted_idx=[0]))),
             (200,asdict(InsertCountsMetadata(n_inserted=1,n_existing=0)))]
    client.__dict__.update(old.legacy_fixtures.client(replies,adapted=False).__dict__)
    monkeypatch.setattr(client,'get_server_information',
                        lambda:dict(name='CDS QCFractal Server',version='0.70'))
    monkeypatch.setattr(client,'query_managers',
                        lambda **kw:[NS(compute_tags=['*'])] if failure=='manager' else [])
    append=load('bank_append')
    store=load('bank_store')
    monkeypatch.setattr(store,'QOMP_INTENTS',tmp_path/'intents')
    monkeypatch.setenv('SLURM_JOB_ID',tag.rsplit('-',1)[1])
    ex=load('qomp_execution')
    ex._MODULES['qomp_expansion']=exp
    exp._MODULES['audited_bank']=core
    base=core.helper('dataset_guard')
    snapshots=[copy.deepcopy(captures[k]) for k in ('before','before','entry','entry','record')]
    if failure=='stale':
        snapshots[0]['metadata']['name']='foreign-live-state'
    monkeypatch.setattr(base,'snapshot_dataset',lambda *a,**kw:snapshots.pop(0))
    original=append.helper
    monkeypatch.setattr(append,'helper',lambda n:
        core if n=='audited_bank' else store if n=='bank_store' else base if n=='dataset_guard'
        else ex if n=='qomp_execution' else original(n))
    saved={};events=[]
    def save(name,data):
        assert name not in saved
        saved[name]=copy.deepcopy(data)
    def admit():
        events.append('new-live-release-observation')
        if failure=='quota': raise ValueError('synthetic fresh group quota refusal')
        return dict(SYNTHETIC=True)
    kwargs=dict(dataset_adapter=adapter,prepared_case=cases[mid],
        checkpoint_raw=previous,checkpoint_sha256=core.digest(previous),
        context_raw=context,context_sha256=core.digest(context),proof_blobs=proofs,
        authenticated_owner='awallace43',save=save,producer='qomp',
        qomp_binding=binding,before_submit=admit)
    return append,client,scope,case,kwargs,saved,events,core,exp,captures,receipts,tensor,store


def test_real_sdk_append_no_create_or_spec_then_audited_checkpoint(monkeypatch,tmp_path):
    a,client,scope,case,kw,saved,events,core,exp,captures,receipts,tensor,store=transaction(
        monkeypatch,tmp_path)
    created=a.append_once(client,scope,case,**kw)
    assert created['dataset_id']==case['dataset_id']==9999
    assert events==['new-live-release-observation']
    assert scope.attempted=={'entry','submit'}
    assert len(client._req_session.sent)==2 and not client.refreshes
    assert saved['owned-before-entry.json']['records']==captures['before']['records']
    outcome=json.loads(receipts['outcome'])
    outcome['dispatch_intent_sha256']=created['dispatch_intent_sha256']
    receipts['outcome']=core.encode(outcome)
    audit=json.loads(receipts['audit'])
    audit['case_outcome_sha256']=core.digest(receipts['outcome'])
    audit['captured_preservation']['dispatch_intent_sha256']=created['dispatch_intent_sha256']
    receipts['audit']=core.encode(audit)
    after,proofs=core.advance(kw['checkpoint_raw'],kw['checkpoint_sha256'],
        kw['context_raw'],kw['context_sha256'],kw['proof_blobs'],
        receipts=receipts,expected_audit_sha256=core.digest(receipts['audit']),
        captured=captures,tensor_raw=tensor,**exp.reducer_options(kw['qomp_binding']))
    assert after['generation']==2 and len(after['audited_records'])==2
    # Actual durable intent, not a fabricated audit witness.
    assert store.verify_intent(saved['dispatch-intent.json'],producer='qomp',
        dataset_id=9999,checkpoint_sha256=kw['checkpoint_sha256'],
        snapshot_sha256=core.helper('dataset_guard').digest(captures['before']),
        context_sha256=kw['context_sha256'],case_sha256=scope.case_sha256,
        eligibility_sha256=case['eligibility_sha256'],monomer_id=case['monomer_id'],
        compute_tag=scope.compute_tag)==created['dispatch_intent_sha256']
    with pytest.raises(ValueError):
        client.make_request('post','api/v1/datasets/singlepoint',dict,body={})
    assert len(client._req_session.sent)==2


@pytest.mark.parametrize('failure,count',[('stale',0),('manager',0),('quota',1)])
def test_stale_prefix_manager_or_fresh_release_quota_stops_without_replay(
        monkeypatch,tmp_path,failure,count):
    a,client,scope,case,kw,saved,events,_,_,_,_,_,_=transaction(monkeypatch,tmp_path,failure=failure)
    with pytest.raises(ValueError):
        a.append_once(client,scope,case,**kw)
    assert len(client._req_session.sent)==count
    assert 'owned-after-record.json' not in saved
    if failure=='quota':
        assert 'dispatch-intent.json' in saved and scope.attempted=={'entry'}
        assert events==['new-live-release-observation']
