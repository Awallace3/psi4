"""Closed QOMP wave authority. No dispatch, current-hash approval or activation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS

AUTHORITY_SHA256='091f87bbea8562787b476122df602991d6b2e9fbec23e1190876b6cbe14ae11a'
CONTEXT_SHA256='5b5b9037bfaf93da2312e5f050f51610455ed8f84a654203f282d39896cd5576'
SERIAL_OPS='37aa33fd215a87832e382c483ab1ec8adb499708'
ROOT=Path('/storage/project/r-cs207-0/awallace43')
SOURCE=ROOT/'gits/psi4.ccasp-native-qomp-production148-v4'
WAVE_ROOT=ROOT/'runs/ccasp-native-s66-qomp148-waves004'
SCRATCH_ROOT=Path('/storage/scratch1/3/awallace43/ccasp-native-qomp148-waves004')
PROTECTED_JOBS=frozenset(('15233737','15233740','15233742','15233744',
                         '15233746','15233748','15233750','15233752'))
SCOPE='qomp-wave'
# Actual bounded scientific failure-lifecycle evidence + parent review still
# gate expansion. Metadata discovery failure is NOT such evidence.
PRODUCTION_ENABLED=False
_MODULES={}


def helper(name):
    if name not in ('qomp_expansion','qomp_bank_policy','bank_store','qomp_campaign',
                    'bank_wave','bank_wave_campaign','qomp_execution'):
        raise ValueError('closed QOMP wave authority helper required')
    if name not in _MODULES:
        path=Path(__file__).with_name(name+'.py')
        spec=importlib.util.spec_from_file_location('_qomp_wave_policy_'+name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
        _MODULES[name]=module
    return _MODULES[name]


def encode(value):
    return json.dumps(value,sort_keys=True,allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def settings(plan):
    manifest=plan.get('producer_authority')
    if (plan.get('producer_profile')!='qomp' or type(manifest) is not dict
            or sha(encode(manifest))!=AUTHORITY_SHA256):
        raise ValueError('parent-pinned QOMP wave authority required')
    return NS(producer='qomp',scope=SCOPE,dataset=manifest['dataset_id'],
        science=manifest['compiled_science'],profile=helper('qomp_bank_policy'),
        protected=helper('qomp_expansion').helper('qomp_campaign').PROTECTED_RECORDS,
        namespace='ccasp-native-qomp148-',manifest=manifest,
        prepared_scope='qomp-pilot',checkpoint_schema='qomp-audited-bank-wave-checkpoint/1',
        audit_status='PASS_QOMP_WAVE_CASE_EVIDENCE')


def extra_proofs(binding):
    """Carry actual authority bytes, not a manufactured serial anchor."""
    e=helper('qomp_expansion')
    e.audit_policy(binding)
    if binding['manifest_sha256']!=AUTHORITY_SHA256:
        raise ValueError('actual audited603 authority pin required')
    values=[binding['manifest_raw'],*binding['original_anchors'].values(),
            *binding['extras'].values()]
    return {sha(raw):raw for raw in values}


def binding_from_proofs(proofs):
    e=helper('qomp_expansion')
    c=e.helper('qomp_campaign')
    if type(proofs) is not dict or AUTHORITY_SHA256 not in proofs:
        raise ValueError('complete QOMP authority proofs required')
    raw=proofs[AUTHORITY_SHA256]
    if sha(raw)!=AUTHORITY_SHA256:
        raise ValueError('QOMP wave authority proof differs')
    manifest=e.helper('qualified_qomp').decode(raw)
    required=set(c.ORIGINAL_SHA.values())|{manifest[key] for key in (
        'worker_sha256','science_accounting_sha256','audit_accounting_sha256')}
    if not required<=set(proofs):
        raise ValueError('complete original/worker/accounting authority proofs required')
    binding=dict(manifest_raw=raw,manifest_sha256=AUTHORITY_SHA256,
        original_anchors={role:proofs[digest] for role,digest in c.ORIGINAL_SHA.items()},
        extras=dict(worker=proofs[manifest['worker_sha256']],
            science_accounting=proofs[manifest['science_accounting_sha256']],
            audit_accounting=proofs[manifest['audit_accounting_sha256']]))
    e.audit_policy(binding)
    if any(sha(raw)!=digest for digest,raw in extra_proofs(binding).items()):
        raise ValueError('QOMP authority proof bytes changed')
    return binding


def reducer_options(proofs):
    return dict(producer='qomp',qomp_binding=binding_from_proofs(proofs))


def bootstrap_binding():
    """Read new owned copies only; never consume deployedv3 paths."""
    e=helper('qomp_expansion')
    c=e.helper('qomp_campaign')
    store=helper('bank_store')
    read=lambda p:store._read_regular(p,maximum=1024**2)
    root=WAVE_ROOT/'inputs'
    binding=dict(manifest_raw=read(root/'bootstrap/authority.json'),
        manifest_sha256=AUTHORITY_SHA256,
        original_anchors={role:read(root/'original25'/relative)
                          for role,relative in c.ORIGINAL_PATHS.items()},
        extras={role:read(root/'bootstrap'/name) for role,name in dict(
            worker='worker-preflight.json',science_accounting='science-accounting.txt',
            audit_accounting='audit-accounting.txt').items()})
    extra_proofs(binding)
    return binding


def prepared():
    binding=bootstrap_binding()
    c=helper('qomp_expansion').helper('qomp_campaign')
    pool,inputs=c.load_pool()
    cases,carry=c.prepared_cases(pool,inputs,binding['original_anchors'])
    return pool,cases,carry


def load_adoption(expected):
    """Actual serial cutover, not a projected/fabricated wave checkpoint."""
    core,store=helper('bank_wave'),helper('bank_store')
    root=WAVE_ROOT/'inputs/adopted'
    read=lambda name:store._read_regular(root/name,maximum=64*1024**2)
    manifest_raw=read('adoption.json')
    manifest=core.decoded(manifest_raw,core.sha(expected))
    if (set(manifest)!={'schema','producing_operations','bootstrap_authority_sha256',
            'state_sha256','context_sha256','checkpoint_sha256',
            'science_accounting_sha256','audit_accounting_sha256','verification_sha256'}
            or manifest['schema']!='qomp-wave-serial-cutover/1'
            or manifest['producing_operations']!=SERIAL_OPS
            or manifest['bootstrap_authority_sha256']!=AUTHORITY_SHA256
            or manifest['context_sha256']!=CONTEXT_SHA256):
        raise ValueError('parent-sealed actual37aa serial cutover required')
    context_raw,checkpoint_raw,state_raw=(read(name) for name in (
        'bank-context.json','bank-checkpoint.json','state.json'))
    for role,raw in (('context',context_raw),('checkpoint',checkpoint_raw),('state',state_raw)):
        core.decoded(raw,manifest[role+'_sha256'])
    context,checkpoint,state=map(core.decoded,(context_raw,checkpoint_raw,state_raw))
    required=core._serial_required(checkpoint,context)
    if len(required)>4*106:
        raise ValueError('bounded actual serial proof prefix required')
    proofs={digest:store._read_regular(root/'proofs'/(core.sha(digest)+'.json'),
                                      maximum=32*1024**2) for digest in required}
    proofs.update(extra_proofs(bootstrap_binding()))
    helper('qomp_expansion').helper('audited_bank').validate_checkpoint(
        checkpoint_raw,manifest['checkpoint_sha256'],context_raw,CONTEXT_SHA256,
        {key:proofs[key] for key in required},**reducer_options(proofs))
    generation=checkpoint['generation']
    old=ROOT/'runs/ccasp-native-s66-qomp148-bank003'
    last=context['eligible_ids'][generation-1]
    original_context=(old/'context/bank-context.json' if generation==1
                      else old/'runs'/last/'audit/payload/bank-context.json')
    if (state.get('generation')!=generation
            or state.get('bank_context_sha256')!=CONTEXT_SHA256
            or state.get('bank_checkpoint_sha256')!=manifest['checkpoint_sha256']
            or state.get('bank_context')!=str(original_context)):
        raise ValueError('cutover state must preserve original37aa context/recorded generation')
    links=context['seed_receipts_sha256'] if generation==1 else checkpoint['ledger'][-1]['receipts_sha256']
    audit,outcome=(core.decoded(proofs[links[role]]) for role in ('audit','outcome'))
    if generation==1:
        if (state.get('status')!='EXTERNAL_603_ORDINARY_PILOT_ACCEPTED'
                or state.get('bank_checkpoint')!=str(old/'seed/bank-checkpoint.json')
                or manifest['verification_sha256'] is not None):
            raise ValueError('external603 seed provenance must remain explicit')
    else:
        verified_raw=read('verified.json')
        verified=core.decoded(verified_raw,manifest['verification_sha256'])
        if (state.get('status')!='ORDINARY_AUDIT_ACCEPTED'
                or state.get('bank_checkpoint')!=str(old/'runs'/last/'audit/payload/bank-checkpoint.json')
                or audit.get('audit_commit')!=SERIAL_OPS
                or outcome.get('operational_commit')!=SERIAL_OPS
                or verified.get('phase')!='audit' or verified.get('state')!=state
                or verified.get('audit_sha256')!=links['audit']
                or verified.get('job_id')!=audit['allocation']['JobId']):
            raise ValueError('actual positively verified37aa serial successor required')
    for role,job in (('science',outcome['allocation']['JobId']),('audit',audit['allocation']['JobId'])):
        raw=read(role+'-accounting.txt')
        if core.digest(raw)!=manifest[role+'_accounting_sha256']:
            raise ValueError('actual cutover terminal accounting differs')
        helper('qomp_expansion').completed_rows(raw.decode(),job)
    return dict(manifest=manifest,state=state,context_raw=context_raw,
        checkpoint_raw=checkpoint_raw,proof_blobs=proofs)


def load_previous(plan):
    core=helper('bank_wave')
    campaign=helper('bank_wave_campaign')
    adopted=load_adoption(plan['adoption_sha256'])
    context_path=WAVE_ROOT/'inputs/adopted/bank-context.json'
    campaign._closed(Path(plan['bootstrap_context_path']),context_path)
    if plan['bootstrap_context_sha256']!=CONTEXT_SHA256:
        raise ValueError('unchanged actual context required')
    path=Path(plan['predecessor_path'])
    if plan['predecessor_sha256']==adopted['manifest']['checkpoint_sha256']:
        campaign._closed(path,WAVE_ROOT/'inputs/adopted/bank-checkpoint.json')
        previous_raw,proofs=adopted['checkpoint_raw'],adopted['proof_blobs']
    else:
        if not path.is_relative_to(WAVE_ROOT) or len(path.relative_to(WAVE_ROOT).parts)!=4:
            raise ValueError('closed own-wave predecessor path required')
        wave_id=path.relative_to(WAVE_ROOT).parts[0]
        if re.fullmatch('wave-[0-9]{4}-[0-9]{4}',wave_id) is None:
            raise ValueError('own-wave predecessor namespace required')
        campaign._closed(path,WAVE_ROOT/wave_id/'finalizer/payload/bank-checkpoint.json')
        previous_raw=campaign._raw(path)
        previous=core.decoded(previous_raw,plan['predecessor_sha256'])
        if previous.get('serial_anchor_sha256')!=adopted['manifest']['checkpoint_sha256']:
            raise ValueError('actual adopted serial anchor must remain unchanged')
        proofs=helper('bank_store')._read_wave_proofs(set(previous['proof_sha256']),producer='qomp')
    previous=core.validate_checkpoint(previous_raw,plan['predecessor_sha256'],
        adopted['context_raw'],CONTEXT_SHA256,proofs)
    if previous['generation']!=plan['generation_before']:
        raise ValueError('recorded immediate wave predecessor required')
    return adopted['context_raw'],previous_raw,proofs


def validate_source(source,commit):
    e=helper('qomp_execution')
    if (Path(source)!=SOURCE or Path(__file__).resolve().parents[2]!=SOURCE or type(commit) is not str
            or re.fullmatch('[0-9a-f]{40}',commit) is None
            or e.git(SOURCE,'rev-parse','HEAD')!=commit
            or e.git(SOURCE,'status','--porcelain')):
        raise ValueError('exact clean new QOMP wave operational source required')
    e.qualified().verify_descriptor_operations(SOURCE,
        e.source_changes(SOURCE,e.qualified().SCIENTIFIC_COMMIT,commit))
