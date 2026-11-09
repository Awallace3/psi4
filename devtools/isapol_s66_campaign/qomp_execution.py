"""QOMP serial-pilot adapter to existing ordinary manager/SCI/audit engines."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

SOURCE = Path('/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-native-qomp-production148-v3')
INTENT_ROOT = Path('/storage/home/hcoda1/3/awallace43/ccasp-native-s66-qomp148-dispatch-intents')
_MODULES = {}
SOURCE_NAMES = ('qomp_campaign.py','qomp_execution.py','qomp_launcher.sh','qomp148.sbatch',
    'qomp_bank_policy.py','qualified_qomp.py','runtime_qomp.py','resource_plan_qomp.py',
    'resource_plan_isa_grid_v1.py','resource_plan_ed561666.py','resource_plan_grid400.py',
    'grid400_bank_policy.py','descriptor_pool.py','bank_runtime_gate.py',
    'bank_dataset.py','bank_transport.py','bank_readonly.py','bank_store.py',
    'dataset_guard.py','phoenix_pilot_cb63305.py','single_record_cb63305.py','envelope_identity.py',
    'runtime_audit.py','runtime_audit_cb63305.py','validate_numeric.py','audited_bank.py')
SOURCE_NAMES += ('qomp_expansion.py','bank_append.py','bank_audit.py','native_controller.py')


def helper(name):
    if name not in ('qomp_campaign','qualified_qomp','qomp_bank_policy','bank_dataset',
        'bank_store','phoenix_installed_checks','phoenix_descriptor_bank_cb63305',
        'phoenix_worker_probe','phoenix_pilot','envelope_identity','dataset_guard','audited_bank',
        'qomp_expansion','bank_append','bank_audit'):
        raise ValueError('closed qomp execution helper')
    if name not in _MODULES:
        path=Path(__file__).with_name(name+'.py')
        spec=importlib.util.spec_from_file_location('_qomp148exec_'+name,path)
        m=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=m
        exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__)
        _MODULES[name]=m
    return _MODULES[name]


def qualified():
    return helper('qualified_qomp')


def policy():
    return helper('qomp_bank_policy')


def dataset():
    return helper('bank_dataset')


def git(source,*args):
    return helper('phoenix_installed_checks').qomp_engine().qomp_git(source,*args)


def git_argv():
    return helper('phoenix_installed_checks').qomp_engine().QOMP_GIT


def source_changes(source,before,after):
    return helper('phoenix_descriptor_bank_cb63305').qomp_changed_paths(
        source,before,after,helper('phoenix_installed_checks').qomp_engine())


def context(args):
    c=helper('qomp_campaign')
    c.need((args.build.resolve(),args.qualification.resolve(),args.inputs.resolve(),args.descriptors.resolve())
           ==(c.BUILD,c.QUALIFICATION,c.INPUTS,c.DESCRIPTORS),'closed QOMP execution input roles required')
    helper('qomp_expansion').validate_paths(args)
    producer=qualified().verify_receipts(qualified().read_receipts(c.RECEIPTS))
    return dict(schema='qomp-verified-installed-source-binding/1',
        commit=qualified().SCIENTIFIC_COMMIT,installed_package=str(c.BUILD/'env/lib/python3.13/site-packages/psi4'),
        producer=producer,historical_build_status='FAILED')


def runtime_check():
    c=helper('qomp_campaign')
    return qualified().runtime().verify_runtime_view(c.read(c.VIEW))


def observe(scratch):
    """Actual current raw filesystem/user/effective-group capture, no old snapshot."""
    c=helper('qomp_campaign')
    root=Path(policy().SCRATCH)
    before=scratch.stat()
    c.need(scratch.resolve()==scratch and before.st_uid==os.geteuid()
           and (before.st_mode&0o777)==0o700,'private owned scratch required')
    fs=os.statvfs(scratch)
    texts={}
    for owner,flag,identifier in (('user','-u',os.getuid()),('group','-g',before.st_gid)):
        r=subprocess.run(['/usr/bin/lfs','quota',flag,str(identifier),str(root)],
            check=True,capture_output=True,text=True,timeout=45)
        c.need(0<len(r.stdout)<=65536,'bounded current raw quota required')
        texts[owner+'_quota']=r.stdout
    return dict(filesystem=str(root),scratch_device=before.st_dev,quota_device=root.stat().st_dev,
        free_bytes=fs.f_bavail*fs.f_frsize,free_files=fs.f_favail,
        observed_at=time.monotonic(),**texts)


def scratch_admission(scratch,profile):
    return policy().admit_scratch([profile],lambda:observe(scratch))


def validate_binding(case,*,runtime,compute_tag,worker_shape,record_id,checkpoint_binding=None,**kwargs):
    c=helper('qomp_campaign')
    serial=case.get('campaign_scope')=='qomp-serial'
    c.need(not kwargs and case['campaign']==c.CAMPAIGN
        and case['campaign_scope'] in ('qomp-pilot','qomp-serial')
        and case['producer_profile']=='qomp' and case['dataset_mode']==('append' if serial else 'fresh')
        and case['producer']['implementation_sha256']==qualified().IMPLEMENTATION
        and case['producer']['scientific_commit']==qualified().SCIENTIFIC_COMMIT
        and type(record_id) is int and record_id>0 and record_id not in c.PROTECTED_RECORDS,
        'one reconstructed fresh QOMP record required')
    profile=policy().validate_profile(case['execution_profile'])
    raw=helper('phoenix_worker_probe').runtime_input(case)
    declared=policy().validate_execution_profile(dict(runtime_input=raw,
        intended_memory_bytes=case['intended_memory_bytes']))
    c.need(declared==profile and c.sha(c.encode(raw))==case['runtime_input_sha256'],
           'QOMP admitted runtime/profile differs')
    raw['specification']['protocols']=helper('phoenix_pilot').execution_specification({})['protocols']
    c.need(c.encode(raw)==c.encode(runtime) and worker_shape==(profile['cores'],profile['science_memory_GiB'])
        and re.fullmatch(re.escape(c.NAMESPACE+case['monomer_id']+'-')+'[1-9][0-9]*',compute_tag),
        'QOMP claim/tag/protocol/worker shape differs')
    if serial:
        checkpoint=helper('qomp_expansion').validate_case(case,checkpoint_binding)
        protected=c.PROTECTED_RECORDS+tuple(checkpoint['audited_records'])
        c.need(record_id not in protected,'QOMP audited-prefix record cannot be claimed')
        return protected
    c.need(checkpoint_binding is None,'fresh pilot cannot consume an append binding')
    return c.PROTECTED_RECORDS


def validate_preflight(preflight,draft,*,binding_sha,qualification_sha,pilot_sha,
                       worker_sources,transport,remaining_case,campaign_scope,operational_commit,
                       checkpoint_binding=None):
    c=helper('qomp_campaign')
    serial=campaign_scope=='qomp-serial'
    c.need(remaining_case is serial and campaign_scope in ('qomp-pilot','qomp-serial')
        and preflight['status']==('PASS_QOMP_SERIAL_WORKER_PREFLIGHT_ONLY' if serial
                                  else 'PASS_QOMP_FRESH_WORKER_PREFLIGHT_ONLY')
        and preflight['producer_profile']=='qomp' and preflight['campaign_scope']==campaign_scope
        and preflight['dataset_mode']==('append' if serial else 'fresh') and preflight['runtime_unchanged'] is True
        and preflight['scientific_build_commit']==qualified().SCIENTIFIC_COMMIT
        and type(operational_commit) is str and re.fullmatch('[0-9a-f]{40}',operational_commit)
        and preflight['operational_commit']==operational_commit
        and preflight['producer_binding_sha256']==binding_sha
        and preflight['qualification_sha256']==qualification_sha
        and preflight['pilot_sha256']==pilot_sha
        and preflight['runtime_input_sha256']==draft['runtime_input_sha256']
        and preflight['monomer_id']==draft['monomer_id']
        and preflight['execution_profile']==policy().validate_profile(draft['execution_profile'])
        and preflight['operational_sources_sha256']==worker_sources
        and preflight['transport_binding']==transport,
        'bound actual QOMP worker/source/transport preflight required')
    profile=draft['execution_profile']
    c.need(preflight['task_config']==dict(ncores=profile['cores'],
        memory_GiB=profile['science_memory_GiB'],internal_retries=0),'QOMP worker task configuration drift')
    if serial:
        c.need(all(preflight[k]==draft[k] for k in ('checkpoint_sha256','bootstrap_context_sha256')),
               'QOMP worker audited predecessor differs')
        exp=helper('qomp_expansion')
        exp.validate_case(draft,checkpoint_binding)
        seed=exp.audit_policy(checkpoint_binding['qomp_binding']).worker
        c.need(all(preflight[k]==seed[k] for k in (
            'harness_path','harness_sha256','qcengine_version','qcfractalcompute_version',
            'manager_sources_sha256'))
            and preflight['transport_binding']['ordinary_stack']==seed['transport_binding']['ordinary_stack'],
            'QOMP SDK/harness/ordinary transport drift from actual accepted remote pilot')


def intent_raw(draft,tag):
    c=helper('qomp_campaign')
    c.need(draft['dataset_mode']=='fresh'
        and re.fullmatch(re.escape(c.NAMESPACE+draft['monomer_id']+'-')+'[1-9][0-9]*',tag),
        'exact fresh QOMP bootstrap intent required')
    return c.encode(dict(schema='qomp148-bootstrap-intent/1',case_sha256=c.sha(c.encode(draft)),
        eligibility_sha256=draft['eligibility_sha256'],carryforward_sha256=draft['carryforward_sha256'],
        compute_tag=tag,dataset_name=c.DATASET_NAME,monomer_id=draft['monomer_id']))


def reserve_transaction(draft,tag):
    """One durable bootstrap intent; lost responses/partials are never repaired."""
    c=helper('qomp_campaign')
    raw=intent_raw(draft,tag)
    store=helper('bank_store')
    root=INTENT_ROOT
    store._private_root(root)
    # Eligibility, not job/tag, is the durable one-writer key across retries.
    store._write_once(root/(draft['eligibility_sha256']+'.json'),raw)
    return c.sha(raw)


def validate_initial_capture(data,outcome,draft):
    c=helper('qomp_campaign')
    base=helper('dataset_guard')
    intent=intent_raw(draft,outcome['compute_tag'])
    c.need(c.read(INTENT_ROOT/(draft['eligibility_sha256']+'.json'))==intent
        and outcome['qomp_bootstrap_intent_sha256']==c.sha(intent),
        'durable QOMP bootstrap intent differs')
    snap=json.loads(c.read(data/'bank-initial-snapshot.json'))
    entry=draft['entry']['name']
    record=str(outcome['record_id'])
    c.need(set(snap['entries'])=={entry} and set(snap['associations'])=={entry}
        and set(snap['records'])=={record} and snap['metadata']['id']==outcome['dataset_id']
        and snap['metadata']['name']==c.DATASET_NAME and snap['metadata']['owner_user']=='awallace43'
        and set(snap['specifications'])=={c.SPECIFICATION_NAME}
        and outcome['dataset_id'] not in c.PROTECTED_DATASETS,'one new QOMP dataset record required')
    base.validate_stored_entry(snap['entries'][entry],draft['entry'])
    finished=snap['records'][record]
    envelope=json.loads(c.read(data/'server-native-envelope.json'))
    tensors=c.read(data/'server-tensors.npz',16*1024**2)
    expected=helper('audited_bank').normalize_specification(draft['specification'])
    expected['keywords'].update(draft['entry']['additional_keywords'])
    c.need(finished['status']=='complete' and len(finished['history'])==1
        and finished['history'][0]['manager_name']==outcome['manager']
        and finished['native_envelope_sha256']==base.digest(envelope)
        and finished['tensor_sha256']==c.sha(tensors)
        and finished['specification']==expected,'QOMP captured delivery changed')
    return snap


def manager_isolation(managers,tag):
    c=helper('qomp_campaign')
    c.need(len(managers)<200,'saturated manager observation cannot prove isolation')
    for manager in managers:
        tags=manager.compute_tags
        c.need(type(tags) is list and all(type(t) is str for t in tags),'manager compute_tags required')
        c.need(not any(t==tag or any(ch in t for ch in '*?[') or t.startswith(c.NAMESPACE)
                       for t in tags),'new serial cohort tag overlaps an active/wildcard manager')


def live_originals(client):
    """Read-only357 preservation check before any new-cohort server write."""
    live=helper('dataset_guard').snapshot_dataset(client,dataset_id=357,max_entries=33)
    validate_original_snapshot(live)
    return live


def validate_original_snapshot(live):
    """Also independently recheck the saved pre-write observation at audit."""
    c=helper('qomp_campaign')
    original=c.original25(c.anchors())
    checkpoint=qualified().decode(c.anchors()['checkpoint'])
    baseline=checkpoint['snapshot']
    c.need(live['metadata']['id']==357 and live['metadata']['name']==original['original_dataset_name']
        and live['metadata']['owner_user']==baseline['metadata']['owner_user']
        and live['metadata'].get('owner_group')==baseline['metadata'].get('owner_group')
        and live['metadata'].get('provenance')==baseline['metadata'].get('provenance')
        and live['specifications']==baseline['specifications']
        and len(live['records'])==len(live['entries'])==33,'old357 identity/spec/33-row view changed')
    carried={str(r['record_id']) for r in original['records']}
    c.need(set(live['records'])-carried=={str(i) for i in range(5234353,5234361)},
           'historical waiting record identities changed')
    for row in original['records']:
        rid,name=str(row['record_id']),row['entry_name']
        c.need(live['records'][rid]==baseline['records'][rid]
            and live['entries'][name]==baseline['entries'][name]
            and live['associations'][name]==baseline['associations'][name],
            'original accepted provenance changed')
    c.need(all(live['records'][rid]['status']=='waiting' for rid in set(live['records'])-carried),
           'old waiting intents changed; reconcile read-only before new writes')
    return live


def launch_binding(args,draft,payload,scratch,phase):
    c=helper('qomp_campaign')
    value=dict(schema='qomp-ordinary-cli-binding/1',source=str(Path(__file__).resolve().parents[2]),
        commit=args.commit,scratch=str(scratch),trace=str(payload/'traces'),
        job=os.environ['SLURM_JOB_ID'],phase=phase,profile=draft['execution_profile'],
        runtime_view=str(c.VIEW),producer='qomp')
    raw=c.encode(value)
    path=payload/'qomp-launch.json'
    helper('bank_store')._write_once(path,raw)
    return dict(CCASP_QOMP_BINDING=str(path),CCASP_QOMP_BINDING_SHA256=c.sha(raw),
                CCASP_QOMP_SOURCE=value['source'])


def run_phase(phase,args,perform):
    """All seals before final receipt; errors/signals cannot become success."""
    c=helper('qomp_campaign')
    engine=helper('phoenix_installed_checks').qomp_engine()
    engine.qomp_enter()
    c.need(args.producer=='qomp' and args.campaign_scope=='qomp-serial'
           and args.remaining_case,'new operational source is append-only QOMP serial')
    c.need(all(getattr(args,k,None) is None for k in (
        'owned_checkpoint','checkpoint_sha256','checkpoint_proofs',
        'accepted_audit','accepted_outcome','accepted_pilot','accepted_envelope',
        'wave_plan','wave_plan_sha256','wave_registration','wave_registration_sha256',
        'wave_science_barrier','wave_science_barrier_sha256')),
        'QOMP serial cannot consume old/wave bindings')
    helper('phoenix_installed_checks').allocation()  # No runtime/proof hashing on a login node.
    helper('qomp_expansion').validate_paths(args,phase)
    helper('qomp_expansion').load_checkpoint(args)
    context(args)
    source=Path(__file__).resolve().parents[2]
    c.need(source==SOURCE and phase in ('worker','science','audit')
           and git(source,'rev-parse','HEAD')==args.commit and not git(source,'status','--porcelain'),
           'clean exact QOMP operations required')
    qualified().verify_descriptor_operations(source,source_changes(source,qualified().SCIENTIFIC_COMMIT,args.commit))
    error=None
    try:
        runtime_check()
        result=perform(args)
        publication=(helper('qomp_expansion').prepare_audit_publication(args,result[1])
                     if phase=='audit' else None)
    except BaseException as exc:
        error=exc
        raise
    finally:
        try:
            runtime_check()
            c.need(git(source,'rev-parse','HEAD')==args.commit and not git(source,'status','--porcelain'),
                   'QOMP source changed')
        except BaseException as post:
            if error is None:
                raise
            error.add_note('QOMP post-seal: '+(type(post).__name__+': '+str(post))[:1024])
    path,report=result
    if publication is not None:
        helper('bank_audit').commit_publication(publication,path.parent)
    helper('phoenix_descriptor_bank_cb63305').write_json_once(path,report)
    print(json.dumps(dict(status=report['status'],report=str(path))))


def launcher(argv):
    """Execute the actual admitted command/env/CWD under decoded-FD tracing."""
    c=helper('qomp_campaign')
    path=Path(os.environ['CCASP_QOMP_BINDING'])
    c.need(path.is_relative_to(c.RUNS) and path.name=='qomp-launch.json','owned launcher binding required')
    raw=c.read(path,1024**2)
    c.need(c.sha(raw)==os.environ['CCASP_QOMP_BINDING_SHA256'],'launcher binding changed')
    binding=qualified().decode(raw)
    c.need(binding['schema']=='qomp-ordinary-cli-binding/1' and binding['producer']=='qomp'
        and binding['source']==str(SOURCE) and Path(__file__).resolve().parents[2]==SOURCE
        and binding['runtime_view']==str(c.VIEW),'foreign launcher source/view')
    c.need(git(SOURCE,'rev-parse','HEAD')==binding['commit'] and not git(SOURCE,'status','--porcelain'),
           'launcher operations changed')
    profile=policy().validate_profile(binding['profile'])
    rt=qualified().runtime()
    metadata=type(argv) is list and argv in (['--module'],['--version'])
    c.need(not any(a in ('--module','--version') for a in argv) or metadata,
           'literal metadata discovery cannot be combined with other arguments')
    scratch=Path(binding['scratch'])
    exchange=None
    if '--qcschema' in argv:
        original=Path.cwd()/'data.msgpack'
        c.need(original.is_relative_to(scratch) and original.parent!=scratch/'cwd',
               'fresh QCE input exchange directory required')
        incoming=c.read(original,16*1024**2)
        original_info=original.lstat()
        exchange=(original,scratch/'cwd/data.msgpack')
        with exchange[1].open('xb') as stream:
            stream.write(incoming)  # Exact serialized bytes, no schema/numeric conversion.
    admit=rt.metadata_cli_launch if metadata else rt.ordinary_cli_launch
    admitted=admit(c.read(c.VIEW),argv,scratch=scratch,producer='qomp',
        execution_profile=profile,inherited_threads=dict(OMP_NUM_THREADS=str(profile['cores']),
            MKL_NUM_THREADS=str(profile['cores']),OPENBLAS_NUM_THREADS='1'))
    trace=Path(binding['trace'])
    c.need(trace==path.parent/'traces' and trace.is_dir(),'owned trace path required')
    command=['/usr/bin/strace','-f','-qq','-yy','-s','4096','-e','trace=%file,%process',
             '-o',str(trace/('psi4-'+str(os.getpid())+'.strace')),*admitted['command']]
    error=None
    previous={}
    def interrupted(number,frame):
        raise KeyboardInterrupt('QOMP CLI signal '+str(number))
    for sig in (signal.SIGTERM,signal.SIGINT):
        previous[sig]=signal.signal(sig,interrupted)
    try:
        with subprocess.Popen(command,cwd=admitted['cwd'],env=admitted['environment'],
                              start_new_session=True) as child:
            try:
                code=child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code,command)
    except BaseException as exc:
        error=exc
        raise
    finally:
        try:
            (rt.verify_metadata_runtime if metadata else rt.verify_runtime_view)(c.read(c.VIEW))
        except BaseException as post:
            if error is None:
                raise
            error.add_note('QOMP CLI post-seal: '+str(post)[:1024])
        finally:
            for sig,handler in previous.items():
                signal.signal(sig,handler)
    if exchange is not None:
        outgoing=c.read(exchange[1],64*1024**2)
        fd=os.open(exchange[0],os.O_WRONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'wb') as stream:
            current=os.fstat(stream.fileno())
            c.need(stat.S_ISREG(current.st_mode)
                   and (current.st_dev,current.st_ino)==(original_info.st_dev,original_info.st_ino),
                   'owned QCE exchange target changed')
            stream.truncate()
            stream.write(outgoing)  # Publish only after all required seals passed.
    return 0


if __name__=='__main__':
    if sys.argv[1:2]!=['launcher']:
        raise SystemExit('closed QOMP launcher entry required')
    raise SystemExit(launcher(sys.argv[2:]))
