"""Closed ordinary-audited QOMP serial expansion, not recovery or acceptance.

Bootstrap authority is an explicit parent-observed literal digest of actual
pilot/audit metadata. Missing evidence never becomes a guessed context.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS

PILOT_OPS = '6033ce8ee20d2a7273e6627a5274b213aaaf7a77'
PILOT_JOB = '15371663'
WORKER_JOB = '15371059'
WORKER_SHA = 'da6a6078992d5ea05a6b85f476654a5d7b6acd9df0ca4b1ba8fe3fe0c3e839ce'
PILOT_AUDIT_SHA = '43f42a12737f7b13af3ba7b4bd62b4b95820a4f56c443e8e5d12f117dae2270f'
PILOT_OUTCOME_SHA = 'c35b4257448163c32de4722f8d91ae8fff38dd5473582b4ed26fccb0eda416b6'
PILOT_CASE_SHA = '67478167059a141bc93ad70420fc036f762a6c9cd60457a22cc85c96370e6fa1'
PILOT_TENSOR_SHA = '445eced6837f2fb893cf00d991559130d3f925fe81c6482bc6fbc22d599d7a32'
# Source6033 SCI.perform adds this wrapper to worker.source_hashes. Its other
# two added keys already belong to the6033 QOMP worker inventory.
PILOT_SCI_WRAPPER_SHA = 'c7e730ba3f4f37a75ff2a0f0a9a0b9d5c03f98507aad86658dc9a4e07bba6735'
ROOT=Path('/storage/project/r-cs207-0/awallace43')
SOURCE=ROOT/'gits/psi4.ccasp-native-qomp-production148-v3'
CAMPAIGN_ROOT=ROOT/'runs/ccasp-native-s66-qomp148-bank003'
CASE_ROOT=CAMPAIGN_ROOT/'cases'
BOOTSTRAP=CAMPAIGN_ROOT/'inputs/bootstrap'
ADMISSION_SCRATCH=Path('/storage/scratch1/3/awallace43/ccasp-native-qomp148-serial-controller003')
_MODULES = {}
MANIFEST_FIELDS = frozenset(('schema','compiled_science','installed_tests','qualifying_operations',
    'descriptor_operations','science_operations','audit_operations','science_job','worker_job',
    'worker_sha256','audit_job','dataset_id','record_id','seed_receipts_sha256','snapshot_sha256',
    'tensor_sha256','science_accounting_sha256','audit_accounting_sha256'))


def helper(name):
    if name not in ('qomp_campaign','qualified_qomp','bank_store','native_controller',
                    'audited_bank','dataset_guard','qomp_execution','bank_audit'):
        raise ValueError('closed QOMP serial helper required')
    if name not in _MODULES:
        p=Path(__file__).with_name(name+'.py')
        s=importlib.util.spec_from_file_location('_qomp_serial_'+name,p)
        m=importlib.util.module_from_spec(s);sys.modules[s.name]=m
        exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
        _MODULES[name]=m
    return _MODULES[name]


def need(ok,message):
    if not ok: raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value,sort_keys=True,allow_nan=False).encode()


def audit_policy(binding):
    """Closed selector used by shared reducers, NEVER module-global monkeypatches."""
    need(type(binding) is dict and set(binding)=={
        'manifest_raw','manifest_sha256','original_anchors','extras'},
        'explicit parent-pinned QOMP bootstrap authority required')
    c,q=helper('qomp_campaign'),helper('qualified_qomp')
    raw,digest=binding['manifest_raw'],binding['manifest_sha256']
    need(type(raw) is bytes and len(raw)<=65536 and type(digest) is str
        and re.fullmatch('[0-9a-f]{64}',digest) and sha(raw)==digest,
        'literal QOMP bootstrap authority differs')
    manifest=q.decode(raw)
    need(type(manifest) is dict and set(manifest)==MANIFEST_FIELDS
        and manifest['schema']=='qomp-ordinary-bootstrap-authority/1'
        and manifest['compiled_science']==q.SCIENTIFIC_COMMIT
        and manifest['installed_tests']==q.TEST_COMMIT
        and manifest['qualifying_operations']==q.QUALIFYING_OPS
        and manifest['descriptor_operations']==c.DESCRIPTOR_OPS
        and manifest['science_operations']==manifest['audit_operations']==PILOT_OPS
        and manifest['science_job']==PILOT_JOB and manifest['worker_job']==WORKER_JOB
        and manifest['worker_sha256']==WORKER_SHA,
        'actual603 pilot/qualification/source roles required')
    need(type(manifest['dataset_id']) is int and manifest['dataset_id']>0
        and manifest['dataset_id'] not in c.PROTECTED_DATASETS
        and type(manifest['record_id']) is int and manifest['record_id']>0
        and manifest['record_id'] not in c.PROTECTED_RECORDS
        and type(manifest['audit_job']) is str
        and re.fullmatch('[1-9][0-9]*',manifest['audit_job'])
        and manifest['audit_job'] not in (PILOT_JOB,WORKER_JOB),
        'independent new-cohort pilot/audit identities required')
    pins=manifest['seed_receipts_sha256']
    need(type(pins) is dict and set(pins)=={'audit','outcome','case','envelope'}
        and all(type(v) is str and re.fullmatch('[0-9a-f]{64}',v) for v in pins.values())
        and pins['audit']==PILOT_AUDIT_SHA and pins['outcome']==PILOT_OUTCOME_SHA
        and pins['case']==PILOT_CASE_SHA and manifest['tensor_sha256']==PILOT_TENSOR_SHA,
        'complete actual seed receipt pins required')
    extras=binding['extras']
    need(type(extras) is dict and set(extras)=={'worker','science_accounting','audit_accounting'},
         'actual worker and both terminal accounting captures required')
    for role in extras:
        expected=WORKER_SHA if role=='worker' else manifest[role+'_sha256']
        need(type(extras[role]) is bytes and len(extras[role])<=1024**2
             and sha(extras[role])==expected,'QOMP bootstrap extra differs: '+role)
    worker=q.decode(extras['worker'])
    need(worker['status']=='PASS_QOMP_FRESH_WORKER_PREFLIGHT_ONLY'
        and worker['operational_commit']==PILOT_OPS
        and worker['allocation']['JobId']==WORKER_JOB and worker['runtime_unchanged'] is True
        and worker['scientific_build_commit']==q.SCIENTIFIC_COMMIT,
        'actual successful pilot worker receipt required')
    completed_rows(extras['science_accounting'].decode(),PILOT_JOB)
    completed_rows(extras['audit_accounting'].decode(),manifest['audit_job'])
    carry=c.original25(binding['original_anchors'])
    return NS(producer='qomp',science=q.SCIENTIFIC_COMMIT,implementation=q.IMPLEMENTATION,
        qualification_sha256=q.RECEIPT_SHA256['qualification'],worker=worker,
        seed_science_sources=dict(worker['operational_sources_sha256'],
            **{'phoenix_pilot_cb63305.sbatch':PILOT_SCI_WRAPPER_SHA}),
        policy=NS(CAMPAIGN=c.CAMPAIGN,PROTECTED_DATASETS=c.PROTECTED_DATASETS,
                  PROTECTED_RECORDS=c.PROTECTED_RECORDS),
        cases=NS(DATASET_NAME=c.DATASET_NAME,SPECIFICATION_NAME=c.SPECIFICATION_NAME),
        fresh_scope='qomp-pilot',append_scope='qomp-serial',
        fresh_status='PASS_QOMP_ORDINARY_PILOT_EVIDENCE',
        append_status='PASS_QOMP_SERIAL_CASE_EVIDENCE',tag_prefix=c.NAMESPACE,
        context_schema='qomp-audited-bank-context/1',
        checkpoint_schema='qomp-audited-bank-checkpoint/1',
        eligible=carry['missing_ids'],carry=carry,manifest=manifest,
        bootstrap_sha256=digest,carry_sha256=sha(encode(carry)))


def reducer_options(binding):
    audit_policy(binding)
    return dict(producer='qomp',qomp_binding=binding)


def completed_rows(text,job):
    """Reuse ordinary terminal predicate; additionally close this one-task job."""
    helper('native_controller').completed_rows(text,job)
    seen=set()
    for line in text.splitlines():
        if not line.strip(): continue
        row=line.strip().split('|')
        need(len(row)>=3 and not any(row[3:]) and row[0] in {
            job,job+'.batch',job+'.0',job+'.extern'} and row[0] not in seen
            and row[1:3]==['COMPLETED','0:0'],
            'QOMP computational/accounting rows must be unique and entirely successful')
        seen.add(row[0])


def load_binding(expected_sha256):
    """An absent or unobserved real pilot bundle is a hard refusal."""
    need(type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}',expected_sha256),
         'parent-observed actual QOMP bootstrap digest required')
    store=helper('bank_store')
    read=lambda name:store._read_regular(BOOTSTRAP/name,maximum=1024**2)
    binding=dict(manifest_raw=read('authority.json'),manifest_sha256=expected_sha256,
        original_anchors=helper('qomp_campaign').anchors(),
        extras={role:read(name) for role,name in dict(worker='worker-preflight.json',
            science_accounting='science-accounting.txt',audit_accounting='audit-accounting.txt').items()})
    audit_policy(binding)
    return binding


def phase_path(mid,phase):
    helper('native_controller').require_monomer(mid)
    need(phase in ('worker','science','audit'),'closed QOMP serial phase required')
    return CAMPAIGN_ROOT/'runs'/mid/phase


def phase_command(source,commit,case,state,phase,*,bootstrap_sha256,worker_sha256=None):
    need(Path(source)==SOURCE and type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit)
         and re.fullmatch('[0-9a-f]{64}',bootstrap_sha256 or ''),
         'closed source and parent bootstrap pin required')
    mid=case['monomer_id']
    run=phase_path(mid,phase)
    profile=helper('qomp_execution').policy().validate_profile(case['execution_profile'])
    need(case['dataset_mode']=='append' and case['campaign_scope']=='qomp-serial'
         and case['checkpoint_sha256']==state['bank_checkpoint_sha256']
         and case['bootstrap_context_sha256']==state['bank_context_sha256'],
         'exact selected audited predecessor required')
    cores,memory,time=(4,24,'00:45:00') if phase=='audit' else (
        profile['cores'],profile['job_memory_GiB'],
        profile['time_limit'] if phase=='science' else '00:45:00')
    command=['/opt/slurm/current/bin/sbatch','--parsable','--no-requeue',
        '--export=USER,LOGNAME,HOME,PATH,LANG,LC_ALL',
        '--account=gts-cs207-chemx','--partition=cpu-small','--qos=embers',
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(cores),'--mem='+str(memory)+'G',
        '--time='+time,'--output='+str(run/'logs/job-%j.out'),'--error='+str(run/'logs/job-%j.err'),
        str(SOURCE/'devtools/isapol_s66_campaign/qomp148.sbatch'),
        str(SOURCE),commit,phase,str(run),str(CASE_ROOT/(mid+'.json'))]
    if phase!='worker':
        need(type(worker_sha256) is str and re.fullmatch('[0-9a-f]{64}',worker_sha256),
             'verified member worker receipt digest required')
        command += [str(phase_path(mid,'worker')/'payload/worker-preflight.json'),worker_sha256]
    if phase=='audit':
        command.append(str(phase_path(mid,'science')))
    return command+[state['bank_context'],state['bank_context_sha256'],
        state['bank_checkpoint'],state['bank_checkpoint_sha256'],bootstrap_sha256]


def actual_pool():
    c=helper('qomp_campaign')
    pool,inputs=c.load_pool()
    cases,carry=c.prepared_cases(pool,inputs,c.anchors())
    return pool,inputs,cases,carry


def load_checkpoint(args):
    binding=load_binding(args.bootstrap_sha256)
    paths=(args.bank_context,args.bank_checkpoint)
    hashes=(args.bank_context_sha256,args.bank_checkpoint_sha256)
    store=helper('bank_store')
    for path,digest,name in zip(paths,hashes,('bank-context.json','bank-checkpoint.json')):
        need(isinstance(path,Path) and path.resolve().is_relative_to(CAMPAIGN_ROOT)
             and path.name==name and type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest),
             'recorded QOMP serial context/checkpoint paths required')
    context_raw,checkpoint_raw=(store._read_regular(p,maximum=64*1024**2) for p in paths)
    need((sha(context_raw),sha(checkpoint_raw))==hashes,
         'recorded QOMP context/checkpoint bytes differ before proof reads')
    context,checkpoint=(helper('qualified_qomp').decode(raw) for raw in (context_raw,checkpoint_raw))
    required=set(context['seed_receipts_sha256'].values())
    need(type(checkpoint['ledger']) is list and len(checkpoint['ledger'])<=105,
         'bounded QOMP serial proof ledger required')
    for row in checkpoint['ledger']:
        required.update(row['receipts_sha256'].values())
    proofs=store.read_proofs(required,producer='qomp')
    helper('audited_bank').validate_checkpoint(checkpoint_raw,hashes[1],context_raw,hashes[0],proofs,
                                              **reducer_options(binding))
    return dict(context_raw=context_raw,context_sha256=hashes[0],
        checkpoint_raw=checkpoint_raw,checkpoint_sha256=hashes[1],
        proof_blobs=proofs,qomp_binding=binding)


def unbound_case(mid):
    _,_,cases,_=actual_pool()
    need(mid in cases,'only reconstructed missing106 members are eligible')
    return cases[mid]


def verify_prepared_from_paths(args):
    need(args.producer=='qomp' and args.campaign_scope=='qomp-serial' and args.remaining_case,
         'explicit append-only QOMP serial scope required')
    binding=load_checkpoint(args)
    mid=args.pilot.stem
    need(args.pilot.resolve()==CASE_ROOT/(mid+'.json'),'closed next QOMP serial case path required')
    expected=helper('audited_bank').bind_next_case(unbound_case(mid),
        binding['checkpoint_raw'],binding['checkpoint_sha256'],
        binding['context_raw'],binding['context_sha256'],binding['proof_blobs'],
        **reducer_options(binding['qomp_binding']))
    raw=helper('bank_store')._read_regular(args.pilot,maximum=32*1024**2)
    need(raw==encode(expected),'entire reconstructed QOMP serial case bytes differ')
    return expected


def initialize(args):
    """Record actual603 ordinary pilot; NO server calls or synthetic seed fallback."""
    validate_source(args.source,args.commit)
    binding=load_binding(args.bootstrap_sha256)
    p=audit_policy(binding)
    pool,_,cases,_=actual_pool()
    store,core,io=helper('bank_store'),helper('audited_bank'),helper('native_controller')
    read=lambda name,cap:store._read_regular(BOOTSTRAP/name,maximum=cap)
    receipts={role:read(name,32*1024**2) for role,name in dict(
        audit='audit-report.json',outcome='qomp-outcome.json',
        case='case.json',envelope='server-native-envelope.json').items()}
    snapshot=helper('qualified_qomp').decode(read('bank-initial-snapshot.json',32*1024**2))
    tensor=read('server-tensors.npz',16*1024**2)
    context,checkpoint=core.initialize(pool,cases,receipts,p.manifest['seed_receipts_sha256']['audit'],
        snapshot,tensor,**reducer_options(binding))
    validate_source(args.source,args.commit)
    store._directory(CAMPAIGN_ROOT,private=True)
    for name in ('context','seed','states','cases','selections','runs'):
        io.mkdir_once(CAMPAIGN_ROOT/name)
    ADMISSION_SCRATCH.mkdir(mode=0o700,exist_ok=False)
    store.prepare_storage(producer='qomp')
    need(store.store_receipts(receipts,producer='qomp')==p.manifest['seed_receipts_sha256'],
         'actual seed proof commitments differ')
    context_path=CAMPAIGN_ROOT/'context/bank-context.json'
    checkpoint_path=CAMPAIGN_ROOT/'seed/bank-checkpoint.json'
    store._write_once(context_path,context)
    checkpoint_raw=encode(checkpoint)
    store._write_once(checkpoint_path,checkpoint_raw)
    plan=dict(schema='qomp-audited-serial-plan/1',source=str(SOURCE),commit=args.commit,
        controller_sha256=sha(Path(__file__).with_name('bank_controller.py').read_bytes()),
        adapter_sha256=args.adapter_seal,refused_ids=[],
        bootstrap_sha256=args.bootstrap_sha256,eligible_ids=p.eligible,
        carried_original_ids=p.carry['accepted_ids'],eligibility_sha256=checkpoint['eligibility_sha256'],
        prepared_sha256={mid:sha(encode(case)) for mid,case in cases.items()})
    io.write_once(CAMPAIGN_ROOT/'plan.json',plan)
    state=dict(generation=1,status='EXTERNAL_603_ORDINARY_PILOT_ACCEPTED',
        bank_context=str(context_path),bank_context_sha256=sha(context),
        bank_checkpoint=str(checkpoint_path),bank_checkpoint_sha256=sha(checkpoint_raw))
    io.write_once(CAMPAIGN_ROOT/'states/001.json',state)
    return state


def validate_source(source,commit):
    source=Path(source)
    q=helper('qomp_execution')
    need(source==SOURCE and type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit)
        and q.git(source,'rev-parse','HEAD')==commit and not q.git(source,'status','--porcelain'),
        'exact clean QOMP serial operational source required')
    q.qualified().verify_descriptor_operations(source,
        q.source_changes(source,q.qualified().SCIENTIFIC_COMMIT,commit))


def validate_paths(args,phase=None):
    mid=helper('native_controller').require_monomer(args.pilot.stem)
    need(args.pilot.resolve()==CASE_ROOT/(mid+'.json'),'closed QOMP serial member case required')
    allowed=(phase_path(mid,phase),) if phase is not None else tuple(
        phase_path(mid,p) for p in ('worker','science','audit'))
    need(args.run.resolve() in allowed,'closed QOMP serial member/role run required')
    if hasattr(args,'case'):
        need(args.case.resolve()==phase_path(mid,'science'),'same-member QOMP science capture required')


def validate_case(case,binding):
    core=helper('audited_bank')
    checkpoint=core.validate_checkpoint(binding['checkpoint_raw'],binding['checkpoint_sha256'],
        binding['context_raw'],binding['context_sha256'],binding['proof_blobs'],
        **reducer_options(binding['qomp_binding']))
    context=json.loads(binding['context_raw'])
    need(checkpoint['generation']<len(context['eligible_ids'])
        and case['monomer_id']==context['eligible_ids'][checkpoint['generation']]
        and case['dataset_id']==context['dataset_id']
        and case['checkpoint_sha256']==binding['checkpoint_sha256']
        and case['bootstrap_context_sha256']==binding['context_sha256'],
        'only the immediate audited QOMP successor may execute')
    unbound=dict(case,dataset_mode='awaiting-audited-bootstrap',campaign_scope='qomp-pilot')
    for name in ('dataset_id','checkpoint_sha256','bootstrap_context_sha256'):
        unbound.pop(name,None)
    need(sha(encode(unbound))==context['prepared_case_sha256'][case['monomer_id']],
         'QOMP serial case differs from original admitted draft')
    return checkpoint


def prepare_audit_publication(args,report):
    """Validate/reduce BEFORE outer postseals; caller publishes report LAST."""
    binding=load_checkpoint(args)
    draft=verify_prepared_from_paths(args)
    data=args.case/'payload'
    store=helper('bank_store')
    outcome=helper('qualified_qomp').decode(store._read_regular(
        data/'qomp-outcome.json',maximum=32*1024**2))
    audit=helper('bank_audit')
    client=audit.approved_client(draft,outcome,binding,producer='qomp')
    _,_,cases,_=actual_pool()
    pool=dict(eligible_ids=sorted(cases),refused_ids=[])
    raw_audit=json.dumps(report,indent=2,allow_nan=False).encode()
    return audit.prepare_publication(client,pool=pool,prepared=cases,raw_audit=raw_audit,
        data=data,case_path=args.pilot,draft=draft,append_inputs=binding,
        output=args.run/'payload',producer='qomp')


def validate_initial_state(state,binding):
    p=audit_policy(binding['qomp_binding'])
    checkpoint=json.loads(binding['checkpoint_raw'])
    need(state['status']=='EXTERNAL_603_ORDINARY_PILOT_ACCEPTED'
         and state['generation']==checkpoint['generation']==1
         and checkpoint['audited_records']==[p.manifest['record_id']]
         and Path(state['bank_context'])==CAMPAIGN_ROOT/'context/bank-context.json'
         and Path(state['bank_checkpoint'])==CAMPAIGN_ROOT/'seed/bank-checkpoint.json',
         'initial QOMP state must retain actual603 pilot provenance')


def controller_quota(case):
    return helper('qomp_execution').scratch_admission(ADMISSION_SCRATCH,case['execution_profile'])
