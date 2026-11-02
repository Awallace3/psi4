"""Prepare any remaining cb/request3 case without changing admitted science.

An operational allocation profile is not a molecular RSS/scratch certificate.
No scientific execution, QCArchive call or job submission occurs here.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BANK_SHA256 = '3492e9c0529d5790d69359c5917a6317d7a5c33bf87bdadf0da70b44522b00a1'
DESCRIPTOR_SHA256 = '328191e2481d9e9e6b7c213fa0b98d94804443e136f2dd6761f33ba6bf841f02'
METHYLAMINE_AUDIT_SHA256 = '1e4e4d638200dfecfec76fa11d7b480138386346d3cfac61af32d1f5eede033d'
DATASET_ID = 354
DATASET_NAME = 'S66x8-native-CCASP-cb63305f41-request3-v1'
SPECIFICATION_NAME = 'pbe0-atz-native-request3'
GIB = 1024**3
PROFILES = {
    's20': dict(cores=8, science_memory_GiB=20, job_memory_GiB=32,
                time_limit='02:00:00', scratch_bytes=128*GIB, scratch_files=65536),
    's40': dict(cores=12, science_memory_GiB=40, job_memory_GiB=64,
                time_limit='04:00:00', scratch_bytes=256*GIB, scratch_files=131072),
    's80': dict(cores=24, science_memory_GiB=80, job_memory_GiB=144,
                time_limit='08:00:00', scratch_bytes=512*GIB, scratch_files=131072),
}


def helper(name):
    if name not in ('prepare_descriptor_bank_cb63305', 'descriptor_admission_cb63305',
                    'phoenix_worker_probe', 'pilot_cb63305'):
        raise ValueError('closed generic-case helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_case_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def validate_execution_profile(name, row):
    if type(name) is not str or name not in PROFILES:
        raise ValueError('one explicit finite execution profile required')
    result = dict(name=name, **copy.deepcopy(PROFILES[name]), account='gts-cs207-chemx',
                  partition='cpu-small', qos='embers', concurrency=1)
    request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    memory = row['intended_memory_bytes']
    if (type(memory) is not int or memory != result['science_memory_GiB']*GIB
            or type(request['resources']['max_bytes']) is not int
            or not 0 < request['resources']['max_bytes'] <= memory
            or memory*100 > result['job_memory_GiB']*GIB*65):
        raise ValueError('allocation profile must preserve scientific memory and page-cache headroom')
    return result


def verify_admitted_bank(inputs, receipts, resources, bank_raw, allocation_raw, reports, *, producer='cb63305'):
    if type(producer) is not str or producer not in ('cb63305', 'ed561666'):
        raise ValueError('closed admitted-bank producer required')
    pins = (dict(bank_sha256=BANK_SHA256, allocation_sha256=DESCRIPTOR_SHA256,
                 descriptor_ops='cbed46c755f7c1b63a1df0d9ebf0eeac4088606e', descriptor_job='14970302')
            if producer == 'cb63305' else helper('pilot_cb63305').producer_pins(producer))
    if digest(bank_raw) != pins['bank_sha256'] or digest(allocation_raw) != pins['allocation_sha256']:
        raise ValueError('exact actual all131 descriptor allocation and raw bank required')
    selection = {} if producer == 'cb63305' else dict(producer=producer)
    bank = helper('prepare_descriptor_bank_cb63305').prepare_bank(inputs, receipts, resources, **selection)
    allocation = json.loads(allocation_raw)
    if (bank != json.loads(bank_raw)
            or allocation.get('schema') != 'native-campaign-descriptor-allocation/2'
            or allocation.get('status') != 'PASS_ALL_131_DESCRIPTOR_PLANS_ONLY'
            or allocation.get('runtime_unchanged') is not True
            or allocation.get('scientific_build_commit') != bank['producer']['scientific_commit']
            or allocation.get('operational_commit') != pins['descriptor_ops']
            or allocation.get('bank_sha256') != pins['bank_sha256']
            or allocation.get('allocation', {}).get('JobId') != pins['descriptor_job']):
        raise ValueError('bank and actual installed descriptor producer receipts differ')
    ids = {row['monomer_id'] for row in bank['monomers']}
    guard = helper('descriptor_admission_cb63305')
    if guard.verify_coverage(allocation['reports'], ids) != 'PASS_ALL_131_DESCRIPTOR_PLANS_ONLY':
        raise ValueError('all131 actual descriptor plans must pass')
    if set(reports) != {row['descriptor_report'] for row in allocation['reports']}:
        raise ValueError('complete exact descriptor child-report byte set required')
    for row in allocation['reports']:
        if digest(reports[row['descriptor_report']]) != row['descriptor_report_sha256']:
            raise ValueError('actual descriptor child-report bytes changed')
    return bank, allocation


def prepare_case(inputs, receipts, resources, bank_raw, allocation_raw, reports, *, monomer_id, profile):
    bank, allocation = verify_admitted_bank(inputs, receipts, resources, bank_raw, allocation_raw, reports)
    if monomer_id == 'monomer-0006':
        raise ValueError('audited methylamine is already present; never resubmit it as a remaining case')
    matches = [row for row in bank['monomers'] if row['monomer_id'] == monomer_id]
    if len(matches) != 1:
        raise ValueError('one exact remaining bank representative required')
    row = matches[0]
    execution = validate_execution_profile(profile, row)
    entry = copy.deepcopy(next(e for e in json.loads(inputs['entries'])['entries']
                               if e['attributes']['monomer_id'] == monomer_id))
    runtime = row['runtime_input']
    if entry['name'] != row['entry_name'] or entry['molecule'] != runtime['molecule']:
        raise ValueError('ordered geometry/reference-attribute join differs from admitted input')
    spec = runtime['specification']
    request = spec['keywords']['function_kwargs']['native_monomer']
    if request.get('schema') != 'native-monomer-request/3':
        raise ValueError('explicit native request3 required')
    entry['additional_keywords'] = dict(function_kwargs=copy.deepcopy(spec['keywords']['function_kwargs']))
    measurement = next(item for item in allocation['reports'] if item['monomer_id'] == monomer_id)
    if measurement['runtime_input_sha256'] != row['runtime_input_sha256']:
        raise ValueError('actual descriptor input hash differs from selected case')
    result = dict(schema='native-bank-molecular-case/1', status='PREPARED_NOT_REGISTERED_OR_SUBMITTED',
        monomer_id=monomer_id, dataset_id=DATASET_ID, dataset_name=DATASET_NAME,
        specification_name=SPECIFICATION_NAME, producer=bank['producer'],
        entry=entry, specification=dict(program='psi4', driver=spec['driver'], **spec['model'],
            keywords={k: copy.deepcopy(v) for k, v in spec['keywords'].items() if k != 'function_kwargs'}),
        intended_memory_bytes=row['intended_memory_bytes'], execution_profile=execution,
        runtime_input_sha256=row['runtime_input_sha256'], geometry_sha256=row['geometry_sha256'],
        descriptor_bank_sha256=BANK_SHA256, descriptor_allocation_sha256=DESCRIPTOR_SHA256,
        descriptor_report_sha256=measurement['descriptor_report_sha256'],
        descriptor_dimensions=measurement['dimensions'], input_sha256=bank['input_sha256'],
        resource_declarations_sha256=bank['resource_declarations_sha256'],
        required_first_pilot_audit_sha256=METHYLAMINE_AUDIT_SHA256,
        submission=dict(find_existing=False, maximum_records=1, maximum_concurrency=1, qos='embers'),
        qualification='Remaining-case raw preparation only. Resource profiles add allocation headroom, '
            'not scientific work/byte/cap changes or measured SCF memory/scratch guarantees. '
            'Owned354 preservation, actual worker/resource pilots, molecular delivery and audit remain required.')
    rebuilt = helper('phoenix_worker_probe').runtime_input(result)
    if rebuilt != runtime or json_digest(rebuilt) != row['runtime_input_sha256']:
        raise ValueError('operational case changed the descriptor-admitted scientific runtime')
    result['case_id'] = monomer_id+'-'+profile+'-'+json_digest(result)[:12]
    return result


def prepare_from_paths(inputs_path, build_identity, qualification, descriptors, *, monomer_id, profile):
    prepare = helper('prepare_descriptor_bank_cb63305')
    inputs = {name: (inputs_path/(name.replace('_', '-')+'.json')).read_bytes()
              for name in prepare.INPUT_SHA256}
    q = qualification
    receipts = dict(build_identity=build_identity.read_bytes(), installed_qualification=q.read_bytes(),
        imported_identity=q.with_name('imported-identity.json').read_bytes(),
        junit=q.with_name('junit.xml').read_bytes())
    allocation_raw = (descriptors/'descriptor-allocation.json').read_bytes()
    if digest(allocation_raw) != DESCRIPTOR_SHA256:
        raise ValueError('exact descriptor allocation required before child-report reads')
    rows = json.loads(allocation_raw)['reports']
    return prepare_case(inputs, receipts, (inputs_path/'resource-declarations.json').read_bytes(),
        (inputs_path/'descriptor-bank.json').read_bytes(), allocation_raw,
        {row['descriptor_report']: (descriptors/row['descriptor_report']).read_bytes() for row in rows},
        monomer_id=monomer_id, profile=profile)


def verify_prepared_from_paths(case_path, inputs_path, build_identity, qualification, descriptors):
    """Reproduce the entire case from closed receipts; labels alone prove nothing."""
    prepared = json.loads(case_path.read_bytes())
    reproduced = prepare_from_paths(inputs_path, build_identity, qualification, descriptors,
        monomer_id=prepared['monomer_id'], profile=prepared['execution_profile']['name'])
    if json_digest(prepared) != json_digest(reproduced):
        raise ValueError('entire prepared case differs from pinned-input reconstruction')
    return reproduced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'build-identity', 'qualification', 'descriptors', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--monomer-id', required=True)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    result = prepare_from_paths(args.inputs, args.build_identity, args.qualification, args.descriptors,
                                monomer_id=args.monomer_id, profile=args.profile)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], case_id=result['case_id'],
                          case_sha256=digest(args.output.read_bytes()))))


if __name__ == '__main__':
    main()
