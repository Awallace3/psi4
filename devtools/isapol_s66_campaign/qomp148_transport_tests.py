"""Real QCPortal/ACK algebra on SYNTHETIC data and fake HTTP only."""
from collections import defaultdict
from dataclasses import asdict
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from urllib.parse import urlsplit

import pytest
import requests
from qcportal.cache import PortalCache
from qcportal.metadata_models import InsertMetadata, InsertCountsMetadata, TaskReturnMetadata

from importlib.util import spec_from_file_location,module_from_spec


def load(name):
    path=Path(__file__).with_name(name+'.py')
    spec=spec_from_file_location('_q148http_'+name,path)
    m=module_from_spec(spec);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__)
    return m


def transaction(monkeypatch):
    c,_,_,_,cases,_=load('test_qomp148').synthetic(monkeypatch)
    case=next(d for d in cases.values() if d['dataset_mode']=='fresh')
    module=load('bank_dataset')
    tag=c.NAMESPACE+case['monomer_id']+'-123'
    scope=module.prepare_transaction(case,copy.deepcopy(case),tag,producer='qomp')
    cls=module.client_class(scope)
    client=object.__new__(cls)
    fixtures=load('transport_cb63305_tests')
    client.__dict__.update(fixtures.legacy_fixtures.client([],adapted=False).__dict__)
    client.cache=PortalCache(client.address,None,16*1024**2)
    client.server_info={'api_limits':defaultdict(lambda:100)}
    client.api_limits=client.server_info['api_limits']
    entry=copy.deepcopy(case['entry']);entry['molecule']['id']=9997
    metadata=dict(id=9999,dataset_type='singlepoint',name=case['dataset_name'],
        description='SYNTHETIC_ONLY',tagline='',tags=[],group='default',visibility=True,
        provenance={},extras={},default_tag=tag,default_priority=1,
        owner_user='awallace43',owner_group=None)
    def response(request,**kwargs):
        route=urlsplit(request.url).path.lstrip('/')
        client._req_session.sent.append((request,kwargs))
        if route=='api/v1/datasets/singlepoint': body=9999
        elif route=='api/v1/datasets/9999': body=metadata
        elif route.endswith('/entries/bulkCreate') or (route.endswith('/specifications') and request.method=='POST'):
            body=asdict(InsertMetadata(inserted_idx=[0]))
        elif route.endswith('/entries/bulkFetch'): body={entry['name']:entry}
        elif route.endswith('/entry_names'): body=[entry['name']]
        elif route.endswith('/specifications/bulkFetch'):
            body={case['specification_name']:dict(name=case['specification_name'],
                 specification=case['specification'],description=None)}
        elif route.endswith('/records/bulkFetch'): body=[]
        elif route.endswith('/submit'): body=asdict(InsertCountsMetadata(n_inserted=1,n_existing=0))
        else: raise AssertionError('HTTP must stay in fake reviewed endpoints: '+route)
        r=requests.Response();r.status_code=200;r.request=request
        r.headers['Content-Type']='application/json';r._content=json.dumps(body).encode()
        return r
    client._req_session.send=response
    return c,module,scope,client,case


def test_complete_fresh_registration_real_ack_delivery_and_independent_audit(monkeypatch,tmp_path):
    from qcportal.managers.models import ManagerName
    from qcfractalcompute.compute_manager import ComputeManager,ManagerStatistics
    c,module,scope,client,case=transaction(monkeypatch)
    saved={}
    save=lambda name,data:saved.setdefault(name,data)
    dataset,_=module.register_once(client,scope,case,save)
    inserted=module.submit_once(client,scope,dataset,case,save)
    assert dataset.id==9999 and inserted.n_inserted==1 and inserted.n_existing==0
    assert scope.attempted=={'create','entry','specification','submit'}
    e=load('qomp_execution')
    monkeypatch.setattr(e,'INTENT_ROOT',tmp_path/'intents')
    intent_sha=e.reserve_transaction(case,scope.compute_tag)
    runtime=e.helper('phoenix_worker_probe').runtime_input(case)
    runtime['specification']['protocols']=e.helper('phoenix_pilot').execution_specification({})['protocols']
    e.validate_binding(case,runtime=runtime,compute_tag=scope.compute_tag,worker_shape=(8,20),record_id=9999)
    claimed=dict(record_id=9999,compute_tag=scope.compute_tag,function='qcengine.compute',
        required_programs={'psi4':'synthetic'},function_kwargs=dict(program='psi4',input_data=dict(
            schema_name='qcschema_input',schema_version=1,molecule=runtime['molecule'],
            **runtime['specification'])))
    e.helper('phoenix_pilot').validate_claim(claimed,runtime,record_id=9999,compute_tag=scope.compute_tag)
    numeric_fixture=load('test_validate_numeric')
    envelope,raw=numeric_fixture.example()
    request=case['entry']['additional_keywords']['function_kwargs']['native_monomer']
    envelope['request']=copy.deepcopy(request)
    envelope['result']['declaration']=copy.deepcopy(request['declaration'])
    envelope['implementation']=dict(sha256=e.qualified().IMPLEMENTATION)
    envelope['artifacts']={'tensors.npz':dict(size=len(raw),sha256=c.sha(raw))}
    witness=load('phoenix_pilot_cb63305').verify_delivery_envelope(
        copy.deepcopy(envelope),envelope,request,e.qualified().IMPLEMENTATION,producer='qomp')
    audit=load('runtime_audit_cb63305')
    assert audit.verify_delivery_identity(dict(success=True,extras={'psi4:native_monomer':envelope}),
        envelope,witness,producer='qomp')==witness
    numeric=numeric_fixture.validator.validate_numeric(envelope,raw)
    assert numeric['status']=='PASS_NUMERIC_TRANSPORT_ONLY'
    assert numeric['isotropic_entries']==27 and numeric['independent_solver_certified'] is False
    mixin=load('single_claim').SingleClaimMixin
    class Manager(mixin,ComputeManager):
        @property
        def n_active_tasks(self): return {'pilot':0}
        @property
        def n_total_active_tasks(self): return 0
        def _acquire_complete_tasks(self):
            return self.synthetic.pop(0) if self.synthetic else {}
        def stop(self): self.stopped=True
    fixtures=load('transport_cb63305_tests')
    with module.helper('dataset_fresh_cb63305').grid400_manager_client_factory() as selected:
        mc=object.__new__(selected)
        mc.__dict__.update(fixtures.legacy_fixtures.client([
            (200,None),(200,asdict(TaskReturnMetadata(accepted_ids=[77]))),(200,None)
        ],adapted=False).__dict__)
        mc.manager_name_data=ManagerName(cluster='synthetic',hostname='unit',uuid='123')
        mc.username='synthetic'
        mc.server_info={'api_limits':{'manager_tasks_return':1}}
        mc.activate('0.63',{'psi4':['synthetic']},[scope.compute_tag])
        manager=object.__new__(Manager)
        manager.client=mc;manager.logger=logging.getLogger('synthetic-qomp-ack')
        manager._deferred_tasks=defaultdict(dict);manager._record_id_map={77:9999}
        manager.statistics=ManagerStatistics()
        manager.manager_config=NS(executors={'pilot':NS(cores_per_worker=8,memory_per_worker=20)})
        manager.target_task_id=77;manager.delivery_outcome=None;manager.stopped=False
        manager.synthetic=[{'pilot':{77:NS(result_compressed=b'SYNTHETIC_NOT_SCIENCE',success=True,walltime=1.)}}]
        for _ in range(3): manager._update(new_tasks=False)
        assert manager.delivery_outcome=='accepted' and manager.stopped
        assert not manager._deferred_tasks and manager.statistics.total_successful_tasks==1
        mc.deactivate(active_tasks=0,active_cores=0,active_memory=0.,total_cpu_hours=0.)
        assert len(mc._req_session.sent)==3 and not mc.refreshes
    outcome=dict(schema='native-expanded-pilot-outcome/1',
        status='DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION',
        producer_profile='qomp',campaign_scope='qomp-pilot',dataset_mode='fresh',
        dataset_name=c.DATASET_NAME,specification_name=c.SPECIFICATION_NAME,
        runtime_unchanged=True,find_existing=False,maximum_records=1,dataset_id=9999,
        record_id=9999,task_id=77,manager='SYNTHETIC_MANAGER',compute_tag=scope.compute_tag)
    outcome['qomp_bootstrap_intent_sha256']=intent_sha
    delivery=dict(record_id=9999,task_id=77,manager=outcome['manager'],outcome='accepted',
        interrupted=False,local_statistics=dict(total_successful_tasks=1,total_failed_tasks=0,
            total_rejected_tasks=0,active_tasks=0))
    created=dict(dataset_id=9999,record_id=9999,compute_tag=scope.compute_tag,n_inserted=1,n_existing=0)
    server=dict(record_id=9999,status='complete',history_count=1,history_managers=[outcome['manager']])
    claim=dict(record_id=9999,task_id=77,manager=outcome['manager'])
    audit.validate_delivery(outcome,delivery,created,claim,server,producer='qomp',campaign_scope='qomp-pilot')
    stored=e.helper('dataset_guard').plain(dataset.get_entry(case['entry']['name'],force_refetch=True))
    spec=e.helper('audited_bank').normalize_specification(case['specification'])
    spec['keywords'].update(case['entry']['additional_keywords'])
    record=dict(status='complete',history=[dict(manager_name=outcome['manager'])],specification=spec,
        native_envelope_sha256=e.helper('dataset_guard').digest(envelope),tensor_sha256=c.sha(raw))
    snap=dict(entries={case['entry']['name']:stored},
        associations={case['entry']['name']:{case['specification_name']:9999}},
        records={'9999':record},metadata=dict(id=9999,name=c.DATASET_NAME,owner_user='awallace43'),
        specifications={case['specification_name']:spec})
    (tmp_path/'bank-initial-snapshot.json').write_bytes(c.encode(snap))
    (tmp_path/'server-native-envelope.json').write_bytes(c.encode(envelope))
    (tmp_path/'server-tensors.npz').write_bytes(raw)
    assert e.validate_initial_capture(tmp_path,outcome,case)==snap
    with pytest.raises(ValueError,match='already attempted'):
        module.submit_once(client,scope,dataset,case,save)
    writes=[urlsplit(req.url).path for req,_ in client._req_session.sent if req.method=='POST']
    assert sum(p.endswith('/submit') for p in writes)==1


@pytest.mark.parametrize('endpoint',[
    'api/v1/datasets/singlepoint/357/submit','api/v1/datasets/singlepoint/9998/entries/bulkCreate',
    'api/v1/records/bulkModify','api/v1/records/5234353/cancel',
    'api/v1/datasets/singlepoint/9999/background_submit',
    'api/v1/datasets/%33%35%37','/api/v1/datasets/357','api//v1/datasets/357'])
def test_foreign_and_denied_http_never_reaches_network(monkeypatch,endpoint):
    _,_,scope,client,_=transaction(monkeypatch)
    scope.request('post','api/v1/datasets/singlepoint');scope.bind_created(9999)
    with pytest.raises(ValueError):
        client.make_request('post',endpoint,dict,body={})
    assert not client._req_session.sent


@pytest.mark.parametrize('method',['put','patch','delete'])
def test_non_post_mutations_refused(monkeypatch,method):
    _,_,scope,client,_=transaction(monkeypatch)
    with pytest.raises(ValueError):
        client.make_request(method,'api/v1/datasets/singlepoint',dict,body={})
    assert not client._req_session.sent


@pytest.mark.parametrize('change',['old_producer','old_profile','foreign_tag','changed_runtime','append'])
def test_transaction_refuses_crossings_before_client_construction(monkeypatch,change):
    c,module,_,_,case=transaction(monkeypatch)
    changed=copy.deepcopy(case);tag=c.NAMESPACE+case['monomer_id']+'-124'
    if change=='old_producer': changed['producer_profile']='grid400'
    elif change=='old_profile': changed['execution_profile']=load('grid400_bank_policy').execution_profile('s20')
    elif change=='foreign_tag': tag='ccasp-native-grid400-bank-'+case['monomer_id']+'-124'
    elif change=='changed_runtime': changed['entry']['additional_keywords']['function_kwargs']['properties']=[]
    else: changed['dataset_mode']='append'
    with pytest.raises(ValueError):
        module.prepare_transaction(changed,changed,tag,producer='qomp')


def test_original357_metadata_read_is_possible_but_old_writes_stay_denied(monkeypatch):
    _,_,scope,_,_=transaction(monkeypatch)
    for part in ('entries','records','specifications'):
        scope.request('post','api/v1/datasets/singlepoint/357/'+part+'/bulkFetch')
    assert not scope.attempted
    for suffix in ('entries/bulkCreate','submit','copy','background_submit'):
        with pytest.raises(ValueError):
            scope.request('post','api/v1/datasets/singlepoint/357/'+suffix)
    assert not scope.attempted
