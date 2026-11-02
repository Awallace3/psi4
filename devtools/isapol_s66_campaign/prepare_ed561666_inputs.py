"""Prepare new-producer raw descriptor inputs, never descriptors or models.

Explicitly rebind only the resource policy label; all 131 resource contracts,
raw geometries and declarations remain unchanged. New actual descriptors must
charge cap32 and may refuse those unchanged budgets.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def prepare_module():
    path = Path(__file__).with_name('prepare_descriptor_bank_cb63305.py')
    spec = importlib.util.spec_from_file_location('_cap32_bank_preparation', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def rebind_resources(raw):
    if type(raw) is not bytes:
        raise ValueError('explicit original resource bytes required')
    values = json.loads(raw)
    if (set(values) != {'schema', 'status', 'inventory_sha256', 'policy_id', 'monomers'}
            or values['schema'] != 'native-s66x8-resource-declarations/2'
            or values['status'] != 'DECLARED_NOT_ADMITTED'
            or values['policy_id'] != 'cb63305-reference-h2h1-ledger-v2'
            or values['inventory_sha256'] != prepare_module().INPUT_SHA256['inventory']
            or type(values['monomers']) is not dict
            or set(values['monomers']) != {f'monomer-{i:04d}' for i in range(1, 132)}):
        raise ValueError('complete original declared cb63305 resource contract set required')
    values['policy_id'] = 'ed561666-reference-h2h1-ledger-v3'
    return json.dumps(values, sort_keys=True, allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'build-identity', 'qualification', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    prepare = prepare_module()
    inputs = {key: (args.inputs/(key.replace('_', '-')+'.json')).read_bytes()
              for key in prepare.INPUT_SHA256}
    q = args.qualification
    receipts = dict(build_identity=args.build_identity.read_bytes(),
        installed_qualification=q.read_bytes(),
        imported_identity=q.with_name('imported-identity.json').read_bytes(),
        junit=q.with_name('junit.xml').read_bytes())
    original = (args.inputs/'resource-declarations.json').read_bytes()
    resources = rebind_resources(original)
    assert json.loads(original)['monomers'] == json.loads(resources)['monomers']
    bank = prepare.prepare_bank(inputs, receipts, resources, producer='ed561666')
    # Claim only a new output directory after complete in-memory validation.
    args.output.mkdir(exist_ok=False)
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    files = {key.replace('_', '-')+'.json': raw for key, raw in inputs.items()}
    files.update({'resource-declarations.json': resources,
                  'descriptor-bank.json': json.dumps(bank, sort_keys=True, allow_nan=False).encode()})
    for name, raw in files.items():
        with (args.output/name).open('xb') as stream:
            stream.write(raw)
    report = dict(status='PREPARED_FOR_DESCRIPTORS_ONLY', producer='ed561666',
        scientific_commit=bank['producer']['scientific_commit'], monomers=131, fragment_contexts=1056,
        original_resource_bytes_sha256=digest(original), contracts_unchanged=True,
        output_sha256={name: digest(raw) for name, raw in files.items()},
        qualification='Raw inputs and finite declarations only. No descriptor admission, SCF, '
                      'molecular model, QCArchive calls, jobs or resource-limit promotion.')
    with (args.output/'preparation-receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
