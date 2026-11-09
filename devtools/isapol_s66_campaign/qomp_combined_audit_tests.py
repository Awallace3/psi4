"""Combined REAL audit.perform -> postseal -> member/controller regression.

Synthetic SDK metadata only. OS/qualification, native numeric/FD and old357
observation boundaries are injected; audit/preflight/report/wave preservation/
envelope/claim/member reduction/controller verification are NOT replaced.
No native runtime, private config, QCA, scheduler or scientific calculation.
"""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
from qcportal.compression import compress,CompressionEnum


def load(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('_combined_audit_'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def fixture(monkeypatch,tmp_path):
    f=load('test_qomp_wave').completed_fixture(monkeypatch,tmp_path,1)
    core=f.wave
    mid=f.plan['member_ids'][0];case=f.cases[mid];slot=f.reg['members'][mid]
    directory=f.binding['plan_path'].parent
    science=directory/'members'/mid/'science';data=science/'payload'
    run=directory/'members'/mid/'audit'
    build=tmp_path/'build'
    for path in (data,run,build): path.mkdir(mode=0o700,parents=True)
    auditor=load('runtime_audit_cb63305')
    q=load('qomp_execution')
    c=f.policy.helper('qomp_expansion').helper('qomp_campaign')
    q._MODULES.update(qomp_campaign=c,qomp_expansion=f.policy.helper('qomp_expansion'),
                     qomp_wave_policy=f.policy)
    runtime=q.helper('bank_wave_runtime');runtime._MODULES['bank_wave']=core
    campaign=runtime.helper('bank_wave_campaign')
    campaign._MODULES.update(bank_wave=core,qomp_wave_policy=f.policy)
    transport=runtime.helper('bank_wave_transport')
    transport._MODULES.update(bank_wave=core,bank_wave_campaign=campaign)
    worker=auditor.helper('phoenix_worker_cb63305')
    bootstrap=auditor.helper('phoenix_pilot_cb63305')
    old=bootstrap.helper
    monkeypatch.setattr(bootstrap,'helper',lambda name:q if name=='qomp_execution' else old(name))
    legacy=auditor.helper('phoenix_pilot')
    events=[]
    def write(path,value):
        path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        raw=core.encode(value) if not isinstance(value,bytes) else value
        path.write_bytes(raw)
        return raw
    pilot=directory/'cases'/(mid+'.json')
    write(pilot,case)
    qualification=build/'installed-validation.json'
    write(qualification,b'SYNTHETIC_QUALIFICATION_BOUNDARY')
    monkeypatch.setattr(c,'QUALIFICATION',qualification)
    profile=case['execution_profile']
    seed=f.policy.helper('qomp_expansion').audit_policy(
        f.policy.binding_from_proofs(f.binding['proof_blobs'])).worker
    current_transport=transport.transport_binding(producer='qomp')
    preflight=dict(status='PASS_QOMP_WAVE_WORKER_PREFLIGHT_ONLY',producer_profile='qomp',
        campaign_scope='qomp-wave',dataset_mode='append',runtime_unchanged=True,
        scientific_build_commit=q.qualified().SCIENTIFIC_COMMIT,operational_commit='e'*40,
        producer_binding_sha256=q.qualified().RECEIPT_SHA256['qualification'],
        qualification_sha256=q.qualified().RECEIPT_SHA256['qualification'],
        pilot_sha256=core.digest(core.encode(case)),runtime_input_sha256=case['runtime_input_sha256'],
        monomer_id=mid,execution_profile=profile,plan_sha256=f.binding['plan_sha256'],
        wave_id=f.plan['wave_id'],checkpoint_sha256=case['checkpoint_sha256'],
        bootstrap_context_sha256=case['bootstrap_context_sha256'],
        task_config=dict(ncores=profile['cores'],memory_GiB=profile['science_memory_GiB'],internal_retries=0),
        operational_sources_sha256=worker.source_hashes(
            'qomp',remaining_case=True,campaign_scope='qomp-wave'),transport_binding=current_transport)
    preflight.update({key:seed[key] for key in ('harness_path','harness_sha256','qcengine_version',
                                              'qcfractalcompute_version','manager_sources_sha256')})
    wp=directory/'members'/mid/'worker/payload/worker-preflight.json'
    worker_raw=write(wp,preflight)
    slot['worker_preflight_sha256']=core.digest(worker_raw)
    f.binding.update(monomer_id=mid,registered_snapshot=f.registered,captures=f.captures,
                     registration=f.reg,registration_raw=core.encode(f.reg),
                     registration_sha256=core.digest(core.encode(f.reg)))
    f.permit['registration_sha256']=f.binding['registration_sha256']
    f.binding.update(release_permit=f.permit,release_permit_raw=core.encode(f.permit),
                     release_permit_sha256=core.digest(core.encode(f.permit)))
    fields=runtime.started_fields(f.binding)
    expected=auditor.helper('phoenix_worker_probe').runtime_input(case)
    expected['specification']['protocols']=legacy.execution_specification(case['specification'])['protocols']
    envelope=core.decoded(f.blobs[f.link['receipts_sha256'][mid]['envelope']])
    envelope.update(result={},molecule_construction=dict(
        input_fix_symmetry=expected['molecule'].get('fix_symmetry'),requested_point_group='c1',
        effective_point_group='c1',max_abs_coordinate_change_bohr=0.))
    envelope_raw=write(data/'server-native-envelope.json',envelope)
    tensor=base64.b64decode(envelope['artifacts']['tensors.npz']['data'])
    write(data/'server-tensors.npz',tensor)
    base=core.helper('dataset_guard')
    for snapshot in (f.deliveries[mid],f.final):
        snapshot['records'][str(slot['record_id'])]['native_envelope_sha256']=base.digest(envelope)
    outcome=core.decoded(f.outcomes[mid])
    sources=set(worker.source_names('qomp',remaining_case=True,campaign_scope='qomp-wave'))|{
        'phoenix_pilot_cb63305.py','phoenix_pilot_cb63305.sbatch','single_record_cb63305.py'}
    source=Path(auditor.__file__).resolve().parents[2]
    source_hashes={name:core.digest((source/'devtools/isapol_s66_campaign'/name).read_bytes())
                   for name in sources}
    scratch=tmp_path/'scratch'
    outcome.update(fields,schema='native-expanded-pilot-outcome/1',task_id=77,
        dataset_name=case['dataset_name'],specification_name=case['specification_name'],
        descriptor_allocation_sha256=case['descriptor_allocation_sha256'],
        transport_binding=current_transport,operational_sources_sha256=source_hashes,
        manager_sources=legacy.MANAGER_FILES,worker_preflight_sha256=core.digest(worker_raw),
        scratch_path=str(scratch/outcome['compute_tag']),trace_files=['synthetic.strace'])
    outcome_raw=write(data/'qomp-outcome.json',outcome)
    f.barrier.update(registration_sha256=f.binding['registration_sha256'],
                     release_permit_sha256=f.binding['release_permit_sha256'])
    f.barrier['members'][mid].update(outcome_sha256=core.digest(outcome_raw),
        delivery_snapshot_sha256=base.digest(f.deliveries[mid]))
    f.binding.update(science_barrier=f.barrier,science_barrier_raw=core.encode(f.barrier),
                     science_barrier_sha256=core.digest(core.encode(f.barrier)))
    write(data/'owned-after-delivery.json',f.deliveries[mid])
    write(data/'expected-runtime.json',expected)
    claim=dict(record_id=slot['record_id'],task_id=77,manager=outcome['manager'],
        compute_tag=slot['compute_tag'],function='qcengine.compute',required_programs=['psi4'],
        function_kwargs=dict(program='psi4',input_data=dict(
            schema_name='qcschema_input',schema_version=1,molecule=expected['molecule'],
            **expected['specification'])))
    write(data/'claimed-input.json',claim);write(data/'claim.json',claim)
    write(data/'manager-delivery.json',dict(record_id=slot['record_id'],task_id=77,
        manager=outcome['manager'],outcome='accepted',interrupted=False,
        local_statistics=dict(total_successful_tasks=1,total_failed_tasks=0,
                              total_rejected_tasks=0,active_tasks=0)))
    write(data/'record-created.json',dict(dataset_id=outcome['dataset_id'],
        record_id=slot['record_id'],compute_tag=slot['compute_tag'],n_inserted=1,n_existing=0))
    write(data/'server-record-outcome.json',dict(record_id=slot['record_id'],
        status='complete',history_count=1,history_managers=[outcome['manager']]))
    write(data/'original25-live-before.json',dict(SYNTHETIC_OLD357_BOUNDARY=True))
    witness=auditor.helper('envelope_identity').verify_native_envelope_identity(envelope,envelope)
    write(data/'envelope-identity.json',witness)
    compressed=compress(dict(success=True,extras={'psi4:native_monomer':envelope}),CompressionEnum.zstd)[0]
    write(data/'local-result-77.zstd',compressed)
    write(data/'traces/synthetic.strace',b'SYNTHETIC_FD_BOUNDARY_NOT_A_NATIVE_TRACE')
    f.outcomes[mid]=outcome_raw
    def revalidate(binding):
        core.validate_plan(binding['plan_raw'],binding['plan_sha256'],**f.args)
        core.validate_registration(f.plan,binding['registration_raw'],binding['registration_sha256'],
            predecessor_snapshot=f.previous['snapshot'],cases=f.cases,captures=f.captures,
            registered_snapshot=f.registered)
        core.validate_release_permit(f.plan,binding['plan_sha256'],f.reg,
            binding['registration_sha256'],binding['release_permit_raw'])
        core.validate_barrier(f.plan,f.reg,binding['release_permit_raw'],binding['science_barrier_raw'],
            outcomes=f.outcomes,accounting=f.science_accounting,deliveries=f.deliveries)
    monkeypatch.setattr(campaign,'revalidate_binding',revalidate)
    monkeypatch.setattr(campaign,'load_binding',lambda args,**kw:revalidate(f.binding) or f.binding)
    # Inject only fixed-path loading/OS/runtime boundaries; keep combined
    # method, transport selection, real preflight and wave predicates intact.
    monkeypatch.setattr(runtime,'validate_paths',lambda args,**kw:None)
    monkeypatch.setattr(worker,'prepare_draft',lambda *a:copy.deepcopy(case))
    monkeypatch.setattr(bootstrap,'load_append_inputs',lambda args:revalidate(f.binding) or f.binding)
    monkeypatch.setattr(q,'context',lambda args:dict(commit=q.qualified().SCIENTIFIC_COMMIT))
    monkeypatch.setattr(q,'git',lambda source,*args:'' if args[0]=='status' else 'e'*40)
    monkeypatch.setattr(q,'SOURCE',source)
    monkeypatch.setattr(q,'source_changes',lambda *a:[])
    monkeypatch.setattr(q.qualified(),'verify_descriptor_operations',lambda *a:None)
    monkeypatch.setattr(q,'runtime_check',lambda:events.append('RUNTIME_SEAL'))
    monkeypatch.setattr(q,'validate_original_snapshot',lambda value:
        value['SYNTHETIC_OLD357_BOUNDARY'] is True or (_ for _ in ()).throw(ValueError('old boundary')))
    monkeypatch.setattr(f.policy,'PRODUCTION_ENABLED',True) # SYNTHETIC invocation, not source activation.
    rt=q.qualified().runtime()
    monkeypatch.setattr(rt,'trace_projection',lambda *a,**kw:dict(
        status='passed',role='molecular_qcschema',executions=[dict(SYNTHETIC=True)]))
    monkeypatch.setattr(q.qualified(),'runtime',lambda:rt)
    monkeypatch.setattr(c,'read',lambda path,*a:b'SYNTHETIC_VIEW_BOUNDARY')
    audit_allocation=dict(f.reg['registration_role']['allocation'],JobId='9000')
    real_checks=auditor.helper('phoenix_installed_checks')
    def digest(path):
        path=Path(path)
        if path==qualification: return q.qualified().RECEIPT_SHA256['qualification']
        if str(path)==seed['harness_path']: return seed['harness_sha256']
        manager=auditor.MANAGER/'lib/python3.13/site-packages/qcfractalcompute'
        if path.is_relative_to(manager): return legacy.MANAGER_FILES[str(path.relative_to(manager))]
        return core.digest(path.read_bytes())
    checks=NS(allocation=lambda:audit_allocation,digest=digest,
        qomp_engine=lambda:NS(qomp_enter=lambda:None,QOMP_GIT=('/usr/bin/git',)))
    old_audit_helper=auditor.helper
    monkeypatch.setattr(auditor,'helper',lambda name:
        q if name=='qomp_execution' else runtime if name=='bank_wave_runtime'
        else worker if name=='phoenix_worker_cb63305' else bootstrap if name=='phoenix_pilot_cb63305'
        else c if name=='qomp_campaign' else legacy if name=='phoenix_pilot'
        else checks if name=='phoenix_installed_checks'
        else NS(validate_numeric=lambda env,raw:dict(
            status='PASS_NUMERIC_TRANSPORT_ONLY',tensor_sha256=core.digest(raw)))
        if name=='validate_numeric' else old_audit_helper(name))
    q._MODULES['phoenix_installed_checks']=checks
    monkeypatch.setattr(auditor,'RUNS',tmp_path);monkeypatch.setattr(auditor,'SCRATCH',scratch)
    accounting=load('test_bank_wave').accounting
    def command(cmd,**kw):
        if cmd[0].endswith('/sacct'):
            raw=accounting(cmd[cmd.index('-j')+1])
            return raw.decode() if kw.get('text') else raw
        if 'show' in cmd:
            relative=cmd[-1].split(':',1)[1]
            assert relative.startswith('devtools/')
            return (source/relative).read_bytes()
        raise AssertionError('unexpected external command')
    monkeypatch.setattr(auditor.subprocess,'check_output',command)
    args=NS(producer='qomp',campaign_scope='qomp-wave',remaining_case=True,
        advance_checkpoint=False,advance_bank_checkpoint=False,build=build,case=science,
        run=run,commit='e'*40,pilot=pilot,worker_preflight=wp,
        worker_preflight_sha256=core.digest(worker_raw),qualification=qualification,
        inputs=tmp_path/'inputs',descriptors=tmp_path/'descriptors',wave_plan=f.binding['plan_path'],
        wave_plan_sha256=f.binding['plan_sha256'],wave_registration=directory/'registration.json',
        wave_registration_sha256=f.binding['registration_sha256'],
        wave_science_barrier=directory/'science-barrier.json',
        wave_science_barrier_sha256=f.binding['science_barrier_sha256'])
    # The SAME real validator accepts the actual worker's wave transport.
    # Consequently the RED in perform is selection drift, not a bad fixture.
    bootstrap.validate_preflight(preflight,case,manifest_sha=digest(qualification),
        qualification_sha=digest(qualification),pilot_sha=digest(pilot),
        worker_sources=preflight['operational_sources_sha256'],transport=current_transport,
        producer='qomp',remaining_case=True,campaign_scope='qomp-wave',
        operational_commit='e'*40,wave_binding=f.binding)
    write(wp.parent.parent/'verified.json',dict(phase='worker',plan_sha256=f.binding['plan_sha256'],
                                              worker_preflight_sha256=core.digest(worker_raw)))
    return NS(**locals())


def test_real_combined_qomp_wave_audit_postseal_report_passes_member_and_controller(
        monkeypatch,tmp_path):
    t=fixture(monkeypatch,tmp_path)
    outputs=[]
    def real_audit(args):
        result=t.auditor.perform(args)
        assert not result[0].exists() # perform must not use legacy immediate write.
        outputs.append(result)
        return result
    t.q.run_phase('audit',t.args,real_audit)
    assert len(outputs)==1 and t.events.count('RUNTIME_SEAL')==4
    path,report=outputs[0]
    receipts=dict(audit=path.read_bytes(),outcome=t.outcome_raw,case=t.pilot.read_bytes(),
                  envelope=t.envelope_raw)
    t.core._member_receipts(t.f.plan,t.f.reg,t.f.permit,t.f.binding['science_barrier_sha256'],
        t.mid,receipts,t.f.deliveries[t.mid],t.f.final)
    assert all(report[k]==v for k,v in t.runtime.started_fields(t.f.binding).items())
    controller=load('bank_wave_controller')
    controller._MODULES.update(bank_wave=t.core,qomp_wave_policy=t.f.policy)
    obj=object.__new__(controller.Production)
    obj.qomp=True;obj.core=t.core;obj.policy=t.f.policy
    obj.binding=lambda **kw:t.f.binding
    obj.validate_source=lambda:None
    cmd=controller.phase_command(t.f.binding,t.mid,'audit')
    (t.run/'submission.json').write_bytes(t.core.encode(dict(command=cmd,job_id='9000')))
    result=obj.verify('audit')
    assert result==dict(verified='audit')
    assert t.core.decoded((t.run/'verified.json').read_bytes())['job_id']=='9000'


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_real_combined_report_is_not_published_when_final_seal_fails(
        monkeypatch,tmp_path,failure):
    t=fixture(monkeypatch,tmp_path)
    outputs=[]
    def seal():
        t.events.append('RUNTIME_SEAL')
        if t.events.count('RUNTIME_SEAL')==4:
            raise failure('synthetic final seal interruption')
    monkeypatch.setattr(t.q,'runtime_check',seal)
    def real_audit(args):
        result=t.auditor.perform(args)
        assert not result[0].exists()
        outputs.append(result)
        return result
    with pytest.raises(failure,match='synthetic final seal interruption'):
        t.q.run_phase('audit',t.args,real_audit)
    assert len(outputs)==1 and t.events.count('RUNTIME_SEAL')==4
    assert not outputs[0][0].exists()
    assert not (t.run/'verified.json').exists()


def test_inactive_remote_initialize_cannot_publish_an_immutable_plan(monkeypatch,tmp_path):
    controller=load('bank_wave_controller')
    policy=controller.helper('qomp_wave_policy')
    assert policy.PRODUCTION_ENABLED is False
    events=[];plan=tmp_path/'plan.json'
    monkeypatch.setattr(policy,'validate_source',lambda *a:events.append('SOURCE'))
    old=controller.helper
    monkeypatch.setattr(controller,'helper',lambda name:
        NS(helper=lambda _:NS(qomp_engine=lambda:NS(qomp_enter=lambda:events.append('CONTROL'))))
        if name=='qomp_execution' else policy if name=='qomp_wave_policy' else old(name))
    class Production:
        def __init__(self,args): events.append('CONSTRUCT')
        def initialize(self):
            plan.write_bytes(b'SYNTHETIC_IMMUTABLE_PLAN')
            return dict(SYNTHETIC=True)
    monkeypatch.setattr(controller,'Production',Production)
    monkeypatch.setattr(sys,'argv',['controller','remote','--producer','qomp',
        '--action','initialize','--source',str(policy.SOURCE),'--commit','e'*40,
        '--adoption-sha256','a'*64])
    with pytest.raises(RuntimeError,match='activation disabled'):
        controller.main()
    assert not plan.exists() and events==[]


def test_inactive_public_initializer_cannot_publish_a_plan(tmp_path):
    controller=load('bank_wave_controller')
    obj=object.__new__(controller.Production)
    obj.qomp=True;obj.policy=NS(PRODUCTION_ENABLED=False)
    plan=tmp_path/'plan.json'
    obj._initialize_qomp=lambda:plan.write_bytes(b'SYNTHETIC_IMMUTABLE_PLAN')
    with pytest.raises(RuntimeError,match='activation disabled'):
        obj.initialize()
    assert not plan.exists()
