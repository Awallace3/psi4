"""Prepare the first fresh cb/request3 molecular pilot from admitted raw inputs.

No scientific execution, network, dataset mutation or job submission occurs.
Historical dataset353 and its acf54 records are never append targets.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

DESCRIPTOR_ALLOCATION_SHA256 = '328191e2481d9e9e6b7c213fa0b98d94804443e136f2dd6761f33ba6bf841f02'
BANK_SHA256 = '3492e9c0529d5790d69359c5917a6317d7a5c33bf87bdadf0da70b44522b00a1'
DESCRIPTOR_OPS = 'cbed46c755f7c1b63a1df0d9ebf0eeac4088606e'
MONOMER = 'monomer-0006'
DATASET_NAME = 'S66x8-native-CCASP-cb63305f41-request3-v1'
SPECIFICATION_NAME = 'pbe0-atz-native-request3'


def helper(name):
    if name not in ('prepare_descriptor_bank_cb63305', 'descriptor_admission_cb63305',
                    'phoenix_worker_probe'):
        raise ValueError('closed first expanded-pilot helpers required')
    path = Path(__file__).with_name(name+'.py')
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location('_expanded_pilot_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare_pilot(inputs, receipts, resources, bank_raw, allocation_raw, reports):
    """Select the fixed nitrogen pilot without changing the admitted request."""
    prepare = helper('prepare_descriptor_bank_cb63305')
    admission = helper('descriptor_admission_cb63305')
    if (digest(bank_raw) != BANK_SHA256
            or digest(allocation_raw) != DESCRIPTOR_ALLOCATION_SHA256):
        raise ValueError('exact completed all131 descriptor allocation and bank required')
    bank = prepare.prepare_bank(inputs, receipts, resources)
    if json.loads(bank_raw) != bank:
        raise ValueError('descriptor bank does not reproduce from pinned raw inputs')
    allocation = json.loads(allocation_raw)
    science = bank['producer']['scientific_commit']
    if (allocation.get('schema') != 'native-campaign-descriptor-allocation/2'
            or allocation.get('status') != 'PASS_ALL_131_DESCRIPTOR_PLANS_ONLY'
            or allocation.get('runtime_unchanged') is not True
            or allocation.get('scientific_build_commit') != science
            or allocation.get('operational_commit') != DESCRIPTOR_OPS
            or allocation.get('bank_sha256') != BANK_SHA256
            or allocation.get('allocation', {}).get('JobId') != '14970302'):
        raise ValueError('successful immutable expanded descriptor receipt required')
    ids = {row['monomer_id'] for row in bank['monomers']}
    if admission.verify_coverage(allocation['reports'], ids) != 'PASS_ALL_131_DESCRIPTOR_PLANS_ONLY':
        raise ValueError('all131 descriptor plans must pass before the first pilot')
    if set(reports) != {row['descriptor_report'] for row in allocation['reports']}:
        raise ValueError('complete exact child-report byte set required')
    for row in allocation['reports']:
        if digest(reports[row['descriptor_report']]) != row['descriptor_report_sha256']:
            raise ValueError('descriptor child-report bytes differ')
    row = next(row for row in bank['monomers'] if row['monomer_id'] == MONOMER)
    inventory = json.loads(inputs['inventory'])
    molecule = next(m for m in inventory['monomers'] if m['id'] == MONOMER)
    # Actual pinned inventory evidence, not a molecular formula inferred from
    # an unrelated rejection fixture or a substituted optimized geometry.
    if molecule['formula'] != 'CH5N' or molecule['representative'] != dict(
            dimer_id='03_Water-MeNH2_0.90', role='B', R=0.9, fragment_indices=list(range(3, 10))):
        raise ValueError('first nitrogen pilot representative changed')
    entry = copy.deepcopy(next(e for e in json.loads(inputs['entries'])['entries']
                               if e['attributes']['monomer_id'] == MONOMER))
    runtime = row['runtime_input']
    spec = runtime['specification']
    if row['intended_memory_bytes'] != 20*1024**3:
        raise ValueError('first expanded pilot requires its declared20GiB scientific memory')
    if entry['name'] != row['entry_name'] or entry['molecule'] != runtime['molecule']:
        raise ValueError('reference attributes are not joined to the admitted ordered geometry')
    entry['additional_keywords'] = dict(function_kwargs=copy.deepcopy(spec['keywords']['function_kwargs']))
    draft = dict(
        schema='native-expanded-molecular-pilot/1', status='PREPARED_NOT_REGISTERED_OR_SUBMITTED',
        monomer_id=MONOMER, formula='CH5N', dataset_name=DATASET_NAME,
        specification_name=SPECIFICATION_NAME,
        specification=dict(program='psi4', driver=spec['driver'], **spec['model'],
                           keywords={k: copy.deepcopy(v) for k, v in spec['keywords'].items()
                                     if k != 'function_kwargs'}),
        entry=entry, producer=bank['producer'], intended_memory_bytes=row['intended_memory_bytes'],
        runtime_input_sha256=row['runtime_input_sha256'],
        descriptor_allocation_sha256=DESCRIPTOR_ALLOCATION_SHA256,
        descriptor_bank_sha256=BANK_SHA256, descriptor_operational_commit=DESCRIPTOR_OPS,
        input_sha256=bank['input_sha256'],
        resource_declarations_sha256=bank['resource_declarations_sha256'],
        queue_policy=dict(qos='embers', account='gts-cs207-chemx', cpus=4,
                          job_memory_GiB=24, psi4_memory_GiB=20, concurrency=1,
                          maximum_records=1, find_existing=False, existing_ok=False),
        qualification='Prepared fresh expanded-producer methylamine pilot only; '
                      'not worker discovery, molecular execution or QCArchive acceptance. '
                      'References remain entry attributes; dataset353 is never a mutation target.')
    reconstructed = helper('phoenix_worker_probe').runtime_input(draft)
    if reconstructed != runtime or prepare.json_digest(reconstructed) != row['runtime_input_sha256']:
        raise ValueError('pilot runtime differs from actual descriptor-admitted request')
    return draft


def prepare_from_paths(inputs_path, build_identity, qualification, descriptors):
    """Read the same closed raw/receipt roles for preparation and launch gates."""
    prepare = helper('prepare_descriptor_bank_cb63305')
    inputs = {key: (inputs_path/(key.replace('_', '-')+'.json')).read_bytes()
              for key in prepare.INPUT_SHA256}
    q = qualification
    receipts = dict(build_identity=build_identity.read_bytes(),
                    installed_qualification=q.read_bytes(),
                    imported_identity=q.with_name('imported-identity.json').read_bytes(),
                    junit=q.with_name('junit.xml').read_bytes())
    allocation = (descriptors/'descriptor-allocation.json').read_bytes()
    if digest(allocation) != DESCRIPTOR_ALLOCATION_SHA256:
        raise ValueError('exact completed all131 descriptor allocation required before child reads')
    rows = json.loads(allocation)['reports']
    return prepare_pilot(inputs, receipts, (inputs_path/'resource-declarations.json').read_bytes(),
        (inputs_path/'descriptor-bank.json').read_bytes(), allocation,
        {row['descriptor_report']: (descriptors/row['descriptor_report']).read_bytes() for row in rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'build-identity', 'qualification', 'descriptors', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare_from_paths(args.inputs, args.build_identity, args.qualification, args.descriptors)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], monomer_id=result['monomer_id'],
                          pilot_sha256=digest(args.output.read_bytes()))))


if __name__ == '__main__':
    main()
