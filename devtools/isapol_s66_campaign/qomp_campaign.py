"""Closed measured QOMP pool and original-provenance carryforward, not adoption."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/storage/project/r-cs207-0/awallace43')
RUNS = ROOT/'runs'
INPUTS = RUNS/'ccasp-native-s66-qomp-profile146-preparation001'
RAW_INPUTS = RUNS/'ccasp-native-s66-cb63305f41-preparation001'
DESCRIPTORS = RUNS/'ccasp-native-s66-qomp-profile146-descriptors001/payload'
RECEIPTS = RUNS/'ccasp-native-s66-qomp-runtime146-check001/inputs/receipts'
VIEW = RECEIPTS.parent/'runtime-view.json'
BUILD = RUNS/'ccasp-native-s66-qomp-fb11-build002'
QUALIFICATION = RECEIPTS/'qualification/installed-validation.json'
CAMPAIGN_ROOT = RUNS/'ccasp-native-s66-qomp-production148-003'
PHASE_ROOTS = {phase:RUNS/('ccasp-native-s66-qomp148-'+phase+'002')
               for phase in ('worker','science','audit')}
ORIGINALS = CAMPAIGN_ROOT/'inputs/original25'
CASE_ROOT = CAMPAIGN_ROOT/'cases'
CAMPAIGN = 'qomp-missing106-serial-pilot-v1'
DATASET_NAME = 'S66x8-native-CCASP-qomp-fb11-missing106-v1'
SPECIFICATION_NAME = 'pbe0-atz-native-qomp-request3'
NAMESPACE = 'ccasp-native-qomp148-'
WORKER_NAMESPACE = 'ccasp-native-worker-qomp148-'
PROTECTED_DATASETS = (353,354,355,356,357)
PROTECTED_RECORDS = tuple(range(5234314,5234361))
BANK_SHA256 = 'c20cc8453a07ef230aeeffd06b7c7b8209ab89c9c83e70b39e9f2ae4b586d37a'
RESOURCE_SHA256 = 'bf584efe3ca4bc8e95d800013b7e03c089e44dab26ace4d7dd4b848304f29843'
ALLOCATION_SHA256 = '0c2bfd5ef611cb7a3b461ee692c980054165e769f3d4809e9b29f6b81b0b447b'
PARENT_PROOF_SHA256 = '2b216fb2f1778330c17ba624020afe142a12885d25a68c08ecf5b90b4d36d07d'
DESCRIPTOR_OPS = 'cba4cd0f4dc5c6cdad9e15eda2a99f22305d799d'
DESCRIPTOR_TREE = '630a0f5b3282c7fa4243e104d23711f651a306d0'
CBA_OPS_DIGEST = 'b7eff3e34475c675a3e8e5f1725257a6c9a50a77763cf5ecf51a541ce9644d21'
CBA_CHILDREN = ('descriptor_probe_cb63305.py','descriptor_probe.py',
    'resource_plan_cb63305.py','descriptor_profiles.py','resource_plan_ed561666.py',
    'resource_plan_grid400.py','resource_plan_isa_grid_v1.py','resource_plan_qomp.py',
    'qualified_qomp.py','runtime_qomp.py')
ORIGINAL_PATHS = dict(state='states/025.json',
    context='ccasp-native-s66-grid400-monomer-0001-audit/payload/bank-context.json',
    checkpoint='ccasp-native-s66-grid400-monomer-0025-audit/payload/bank-checkpoint.json')
ORIGINAL_SHA = dict(state='10752591a184dd0beb8810081a6361b7a93ca9230177da4145c4259d5d744339',
    context='a92028bc139edcffe5c6f03cc8dd1945c30183149a235b94852882dc2678fec9',
    checkpoint='d4ace10ad0823800f2b0a678feaa88539a30907c8a6a869e9363f7a91ff7ced8')


def helper(name):
    if name not in ('qualified_qomp','resource_plan_qomp','descriptor_pool',
                    'prepare_descriptor_bank_cb63305','qomp_bank_policy','pilot_cb63305',
                    'phoenix_worker_probe','phoenix_installed_checks','bank_store'):
        raise ValueError('closed QOMP campaign helper')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_qomp148_'+name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def need(ok,message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return json.dumps(value,sort_keys=True,allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path,maximum=64*1024**2):
    return helper('bank_store')._read_regular(Path(path),maximum=maximum)


def authority():
    """Historical producing files, NEVER current147/148 operation inventories."""
    engine = helper('phoenix_installed_checks').qomp_engine()
    env = engine.qomp_control_environment(__import__('os').environ)
    source = Path(__file__).resolve().parents[2]
    engine.qomp_config_gate(source,env)
    need(engine.qomp_git(source,'rev-parse',DESCRIPTOR_OPS+'^{tree}')==DESCRIPTOR_TREE,
         'historical producing-cba tree differs')
    # Freeze the actual producer's OP_FILES and profile via its reviewed blobs.
    def blob(name):
        return subprocess.check_output([*engine.QOMP_GIT,'-C',str(source),'show',
            DESCRIPTOR_OPS+':devtools/isapol_s66_campaign/'+name],env=env)
    profiles = {}
    exec(compile(blob('descriptor_profiles.py'),'frozen-cba-profiles','exec'),profiles)
    import ast
    tree = ast.parse(blob('phoenix_descriptor_bank_cb63305.py'))
    parent = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='OP_FILES' for t in n.targets))
    p = profiles['profile']('qomp',scope='full-bank')
    names = tuple(dict.fromkeys(parent+p['parent_extra_sources']))
    operations = {name:sha(blob(name)) for name in names}
    need(len(operations)==19 and sha(encode(operations))==CBA_OPS_DIGEST
         and tuple(p['child_sources'])==CBA_CHILDREN,'historical producing-cba inventory differs')
    return dict(bank_sha256=BANK_SHA256,resources_sha256=RESOURCE_SHA256,
        allocation_sha256=ALLOCATION_SHA256,operational_commit=DESCRIPTOR_OPS,
        job_id='15364005',operational_sources_sha256=operations)


def original25(blobs):
    need(type(blobs) is dict and set(blobs)==set(ORIGINAL_SHA),'three original anchor roles required')
    q=helper('qualified_qomp')
    for role,expected in ORIGINAL_SHA.items():
        need(type(blobs[role]) is bytes and sha(blobs[role])==expected,'original anchor hash differs: '+role)
    state,context,checkpoint=(q.decode(blobs[k]) for k in ('state','context','checkpoint'))
    need(state['status']=='ORDINARY_AUDIT_ACCEPTED' and state['generation']==25
         and state['bank_context_sha256']==ORIGINAL_SHA['context']
         and state['bank_checkpoint_sha256']==ORIGINAL_SHA['checkpoint']
         and checkpoint['bootstrap_context_sha256']==ORIGINAL_SHA['context']
         and checkpoint['generation']==25 and checkpoint['dataset_id']==context['dataset_id']==357,
         'original generation/dataset/link differs')
    first=re.fullmatch(r'(monomer-\d{4})-[0-9a-f]{12}',context['first_entry'])
    need(first is not None and type(context['first_record']) is int,'original first entry/record differs')
    records=[dict(monomer_id=first[1],record_id=context['first_record'],
        entry_name=context['first_entry'],receipts_sha256=context['seed_receipts_sha256'])]
    need(type(checkpoint['ledger']) is list and len(checkpoint['ledger'])==24,'original24 audited links required')
    records += [copy.deepcopy(row) for row in checkpoint['ledger']]
    ids=[r['monomer_id'] for r in records]
    record_ids=[r['record_id'] for r in records]
    universe={f'monomer-{i:04d}' for i in range(1,132)}
    need(len(set(ids))==len(set(record_ids))==25 and set(ids)<=universe
         and all(type(n) is int for n in checkpoint['audited_records'])
         and set(record_ids)==set(checkpoint['audited_records'])
         and len(checkpoint['audited_records'])==25,'original audited identities differ')
    for row in records:
        need(row['entry_name'].startswith(row['monomer_id']+'-'),'original entry/monomer mismatch')
        saved=checkpoint['snapshot']['records'][str(row['record_id'])]
        need(saved['status']=='complete','original carried record is not complete')
    return dict(schema='qomp-original-provenance-carryforward/1',dataset_id=357,
        anchor_sha256=dict(ORIGINAL_SHA),original_producer=context['producer'],
        original_dataset_name=context['dataset_name'],original_specification_name=context['specification_name'],
        records=records,accepted_ids=sorted(ids),missing_ids=sorted(universe-set(ids)),
        qualification='Pinned historical ordinary-audit anchors only; no tensor/record promotion, '
                      'not a current server query or new QOMP output.')


def load_pool(inputs=INPUTS,qualification=QUALIFICATION,descriptors=DESCRIPTORS):
    need(tuple(map(Path,(inputs,qualification,descriptors)))==(INPUTS,QUALIFICATION,DESCRIPTORS),
         'fixed measured QOMP paths required')
    prepare=helper('prepare_descriptor_bank_cb63305')
    raw={k:read(RAW_INPUTS/(k.replace('_','-')+'.json')) for k in prepare.INPUT_SHA256}
    receipts=helper('qualified_qomp').read_receipts(RECEIPTS)
    helper('qualified_qomp').runtime().load_view(read(VIEW))
    need(sha(read(DESCRIPTORS.parent/'parent-verified.json'))==PARENT_PROOF_SHA256,
         'actual independent descriptor proof differs')
    reports={f'monomer-{i:04d}-descriptors.json':read(DESCRIPTORS/f'monomer-{i:04d}-descriptors.json')
             for i in range(1,132)}
    pool=helper('descriptor_pool').verify_pool(raw,receipts,read(INPUTS/'resources-qomp.json'),
        read(INPUTS/'descriptor-bank.json'),read(DESCRIPTORS/'descriptor-allocation.json'),
        reports,authority=authority(),producer='qomp')
    return pool,raw


def prepared_cases(pool,inputs,anchors):
    carry=original25(anchors)
    need(pool['eligible_ids']==[f'monomer-{i:04d}' for i in range(1,132)]
         and not pool['refused_ids'],'complete actual131 admitted QOMP pool required')
    policy=helper('qomp_bank_policy')
    rows={r['monomer_id']:r for r in pool['bank']['monomers']}
    inventory={r['id']:r for r in json.loads(inputs['inventory'])['monomers']}
    cases={}
    for mid in carry['missing_ids']:
        row=rows[mid]
        execution=policy.validate_execution_profile(row)
        measured=next(r for r in pool['allocation']['reports'] if r['monomer_id']==mid)
        pins=dict(producer='qomp',monomer=mid,dataset_name=DATASET_NAME,
            specification_name=SPECIFICATION_NAME,schema='native-selected-molecular-pilot/1',
            bank_sha256=BANK_SHA256,allocation_sha256=ALLOCATION_SHA256,
            runtime_input_sha256=row['runtime_input_sha256'],
            selected_report_sha256=measured['descriptor_report_sha256'],descriptor_ops=DESCRIPTOR_OPS)
        case=helper('pilot_cb63305')._prepare_fresh_selected(inputs,pool['bank'],pool['allocation'],
            pins,execution,formula=inventory[mid]['formula'],
            qualification='New missing106 draft only. Original25 provenance is separate; no model acceptance.')
        case.update(campaign=CAMPAIGN,campaign_scope='qomp-pilot',carryforward_sha256=sha(encode(carry)),
            eligibility_sha256=sha(encode(dict(authority=pool['authority'],missing_ids=carry['missing_ids']))),
            eligible_monomer_count=106,refused_monomer_count=0)
        if mid!=carry['missing_ids'][0]:
            case['dataset_mode']='awaiting-audited-bootstrap'
        cases[mid]=case
    return cases,carry


def pins():
    return dict(producer='qomp',campaign=CAMPAIGN,dataset_name=DATASET_NAME,
        specification_name=SPECIFICATION_NAME,protected_datasets=PROTECTED_DATASETS,
        protected_records=PROTECTED_RECORDS,namespace=NAMESPACE,worker_namespace=WORKER_NAMESPACE)


def verify_input_receipts(allocation,inputs,receipts):
    expected={str(RAW_INPUTS/(k.replace('_','-')+'.json')):sha(raw) for k,raw in inputs.items()}
    q=helper('qualified_qomp')
    expected.update({str(RECEIPTS/name):sha(receipts[role]) for role,name in q.RECEIPT_PATHS.items()})
    expected.update({str(INPUTS/'resources-qomp.json'):RESOURCE_SHA256,
        str(INPUTS/'descriptor-bank.json'):BANK_SHA256,str(VIEW):q.VIEW_SHA256})
    need(allocation['input_receipts_sha256']==expected,'historical21 input roles/hashes differ')


def anchors(root=ORIGINALS):
    return {role:read(Path(root)/name) for role,name in ORIGINAL_PATHS.items()}


def verify_prepared_from_paths(args):
    need(not args.remaining_case and args.campaign_scope=='qomp-pilot'
         and all(getattr(args,k,None) is None for k in (
             'bank_context','bank_checkpoint','owned_checkpoint','wave_plan','wave_registration')),
         'QOMP148A cannot append, replay old context, or run a wave')
    pool,inputs=load_pool(args.inputs,args.qualification,args.descriptors)
    cases,carry=prepared_cases(pool,inputs,anchors())
    first=carry['missing_ids'][0]
    need(args.pilot.resolve()==CASE_ROOT/(first+'.json'),'only the first missing ordinary pilot is runnable')
    expected=cases[first]
    need(helper('qualified_qomp').decode(read(args.pilot))==expected,'entire QOMP pilot draft differs')
    need(helper('qualified_qomp').decode(read(CAMPAIGN_ROOT/'inputs/carryforward.json'))==carry,
         'original provenance manifest differs')
    return expected


def phase_command(case,*,source,commit,phase,run,pilot,preflight=None,preflight_sha=None,science=None):
    """Exact serial operator command data; NO scheduler call is made."""
    need(phase in ('worker','science','audit') and Path(source)==
        ROOT/'gits/psi4.ccasp-native-qomp-production148-v2'
        and re.fullmatch('[0-9a-f]{40}',commit),'closed QOMP serial command required')
    profile=helper('qomp_bank_policy').validate_profile(case['execution_profile'])
    cores,memory,duration=(4,24,'00:45:00') if phase=='audit' else (
        profile['cores'],profile['job_memory_GiB'],
        profile['time_limit'] if phase=='science' else '00:45:00')
    need(Path(run)==PHASE_ROOTS[phase],
         'dedicated new phase root required')
    args=[str(source),commit,phase,str(run),str(pilot)]
    if phase!='worker':
        need(preflight is not None and re.fullmatch('[0-9a-f]{64}',preflight_sha or ''),
             'observed worker preflight path/digest required')
        args += [str(preflight),preflight_sha]
    if phase=='audit':
        need(science is not None,'actual science root required')
        args += [str(science)]
    return ['/opt/slurm/current/bin/sbatch','--parsable','--no-requeue',
        '--account=gts-cs207-chemx','--partition=cpu-small','--qos=embers','--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(cores),'--mem='+str(memory)+'G','--time='+duration,
        '--output='+str(Path(run)/'logs/slurm-%j.out'),'--error='+str(Path(run)/'logs/slurm-%j.err'),
        str(Path(source)/'devtools/isapol_s66_campaign/qomp148.sbatch'),*args]


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare',))
    parser.parse_args()
    pool,inputs=load_pool()
    cases,carry=prepared_cases(pool,inputs,anchors())
    need(len(cases)==106 and len(carry['accepted_ids'])==25,'exact25+106 partition required')
    store=helper('bank_store')
    store._directory(CAMPAIGN_ROOT,private=True)
    CASE_ROOT.mkdir(mode=0o700,exist_ok=False)
    store._write_once(CAMPAIGN_ROOT/'inputs/carryforward.json',encode(carry))
    for mid,case in cases.items():
        store._write_once(CASE_ROOT/(mid+'.json'),encode(case))
    print(json.dumps(dict(status='PREPARED_MISSING106_NOT_SUBMITTED',first=carry['missing_ids'][0],
        carried=25,new_cases=106,carryforward_sha256=sha(encode(carry)))))


if __name__=='__main__':
    main()
