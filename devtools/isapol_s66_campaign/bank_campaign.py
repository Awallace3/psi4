"""Closed measured-input adapter for the new eligible grid400 bank.

The all131 descriptor allocation must finish and its actual bytes must be
sealed below before this adapter can prepare any production case. Context
and checkpoint hashes are supplied only by the controller after positively
verified ordinary audit jobs, never by forensic acceptance or old seeds.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/storage/project/r-cs207-0/awallace43')
RUNS = ROOT/'runs'
INPUTS = RUNS/'ccasp-native-s66-grid400-1aaa9e0a30-full-preparation001'
DESCRIPTORS = RUNS/'ccasp-native-s66-grid400-1aaa9e0a30-full-descriptors001/payload'
BUILD_IDENTITY = RUNS/'ccasp-native-s66-grid400-1aaa9e0a30-build001/build-identity.json'
QUALIFICATION = RUNS/'ccasp-native-s66-grid400-1aaa9e0a30-installed001/payload/installed-validation.json'
CAMPAIGN_ROOT = RUNS/'ccasp-native-s66-grid400-1aaa9e0a30-bank001'
CASE_ROOT = CAMPAIGN_ROOT/'cases'
BANK_SHA256 = 'f989e6aac0b76a14c2f5214d437552432356ce343a773f52eae0b2ab192b7ff2'
RESOURCE_SHA256 = '26a87f6a44714707c17e501957ef0d7f6a2f80950624efa94022d7cc39cc8b89'
DESCRIPTOR_OPS = 'f7d4bb4314baa7372ed47d5876e9d34475ec9c19'
DESCRIPTOR_JOB = '15153484'
# Actual job15153484: all131 reports independently reduced with recorded
# source/input/producer pins; 116 admitted and15 refused, never a whole-bank pass.
# Do not infer this authority from caller bytes or a forecast/singleton report.
ALLOCATION_SHA256 = 'f3c8cf04297444ea67a79352f4aae8cef7b645262a48176b9b04e89832b88050'
ADDITIONAL_SOURCES = (
    'bank_campaign.py', 'descriptor_pool.py', 'grid400_bank_policy.py',
    'grid400_bank_case.py', 'grid400_bank_execution.py', 'bank_transport.py',
    'bank_dataset.py', 'bank_store.py', 'bank_append.py', 'audited_bank.py',
    'bank_runtime_gate.py', 'dataset_guard.py', 'runtime_audit.py', 'single_record_cb63305.py',
)
_MODULES = {}


def helper(name):
    if name not in ('descriptor_pool', 'descriptor_profiles', 'phoenix_descriptor_bank_cb63305',
                    'prepare_descriptor_bank_cb63305', 'grid400_bank_case', 'grid400_bank_policy',
                    'audited_bank', 'bank_store'):
        raise ValueError('closed measured eligible-bank campaign helpers required')
    if name not in _MODULES:
        path = Path(__file__).with_name(name+'.py')
        spec = importlib.util.spec_from_file_location('_bank_campaign_'+name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        _MODULES[name] = module
    return _MODULES[name]


def pins():
    policy, cases = helper('grid400_bank_policy'), helper('grid400_bank_case')
    return dict(producer='grid400', campaign=policy.CAMPAIGN,
        dataset_name=cases.DATASET_NAME, specification_name=cases.SPECIFICATION_NAME,
        protected_datasets=policy.PROTECTED_DATASETS, protected_records=policy.PROTECTED_RECORDS,
        namespace='ccasp-native-grid400-bank-', worker_namespace='ccasp-native-worker-grid400-bank-')


def authority():
    if type(ALLOCATION_SHA256) is not str or re.fullmatch('[0-9a-f]{64}', ALLOCATION_SHA256) is None:
        raise ValueError('actual completed grid400 descriptor pool has not been sealed for production')
    profile = helper('descriptor_profiles').profile('grid400', scope='full-bank')
    names = helper('phoenix_descriptor_bank_cb63305').OP_FILES+profile['parent_extra_sources']
    source = Path(__file__).resolve().parents[2]
    operations = {name: hashlib.sha256(subprocess.check_output(
        ['git', 'show', DESCRIPTOR_OPS+':devtools/isapol_s66_campaign/'+name], cwd=source)).hexdigest()
        for name in names}
    return dict(bank_sha256=BANK_SHA256, resources_sha256=RESOURCE_SHA256,
        allocation_sha256=ALLOCATION_SHA256, operational_commit=DESCRIPTOR_OPS,
        job_id=DESCRIPTOR_JOB, operational_sources_sha256=operations)


def _read(path, *, maximum=64*1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('bounded plain measured-bank metadata file required')
    return path.read_bytes()


def load_pool(inputs, build_identity, qualification, descriptors):
    bound = authority()  # Fail closed before any candidate metadata reads.
    if tuple(Path(p).resolve() for p in (inputs, build_identity, qualification, descriptors)) != (
            INPUTS, BUILD_IDENTITY, QUALIFICATION, DESCRIPTORS):
        raise ValueError('exact new measured grid400 campaign input/producer paths required')
    prepare = helper('prepare_descriptor_bank_cb63305')
    raw_inputs = {role: _read(INPUTS/(role.replace('_', '-')+'.json')) for role in prepare.INPUT_SHA256}
    receipts = dict(build_identity=_read(BUILD_IDENTITY), installed_qualification=_read(QUALIFICATION),
        imported_identity=_read(QUALIFICATION.with_name('imported-identity.json')),
        junit=_read(QUALIFICATION.with_name('junit.xml')))
    allocation = _read(DESCRIPTORS/'descriptor-allocation.json')
    if hashlib.sha256(allocation).hexdigest() != bound['allocation_sha256']:
        raise ValueError('sealed full descriptor allocation bytes differ before child reads')
    reports = {f'monomer-{i:04d}-descriptors.json': _read(
        DESCRIPTORS/f'monomer-{i:04d}-descriptors.json') for i in range(1, 132)}
    pool = helper('descriptor_pool').verify_pool(raw_inputs, receipts,
        _read(INPUTS/'resource-declarations.json'), _read(INPUTS/'descriptor-bank.json'),
        allocation, reports, authority=bound)
    return pool, raw_inputs


def prepared_cases(pool, raw_inputs):
    return {mid: helper('grid400_bank_case').prepare_case(raw_inputs, pool, mid)
            for mid in pool['eligible_ids']}


def load_checkpoint(args):
    paths = (getattr(args, 'bank_context', None), getattr(args, 'bank_checkpoint', None))
    hashes = (getattr(args, 'bank_context_sha256', None), getattr(args, 'bank_checkpoint_sha256', None))
    for path, sha, name in zip(paths, hashes, ('bank-context.json', 'bank-checkpoint.json')):
        if (not isinstance(path, Path) or not path.resolve().is_relative_to(CAMPAIGN_ROOT)
                or path.name != name or type(sha) is not str or re.fullmatch('[0-9a-f]{64}', sha) is None):
            raise ValueError('recorded ordinary-audit bank context/checkpoint paths and hashes required')
    context_raw, checkpoint_raw = map(_read, paths)
    if tuple(hashlib.sha256(raw).hexdigest() for raw in (context_raw, checkpoint_raw)) != hashes:
        raise ValueError('recorded ordinary-audit bank metadata bytes differ')
    context, checkpoint = json.loads(context_raw), json.loads(checkpoint_raw)
    required = set(context['seed_receipts_sha256'].values())
    ledger = checkpoint.get('ledger')
    if type(ledger) is not list or len(ledger) > 130:
        raise ValueError('bounded ordinary bank audit proof ledger required')
    for link in ledger:
        required.update(link['receipts_sha256'].values())
    proofs = helper('bank_store').read_proofs(required)
    helper('audited_bank').validate_checkpoint(
        checkpoint_raw, hashes[1], context_raw, hashes[0], proofs)
    return dict(context_raw=context_raw, context_sha256=hashes[0],
        checkpoint_raw=checkpoint_raw, checkpoint_sha256=hashes[1], proof_blobs=proofs)


def verify_prepared_from_paths(args, manifest_path):
    pool, inputs = load_pool(args.inputs, manifest_path, args.qualification, args.descriptors)
    raw = _read(args.pilot)
    case = json.loads(raw)
    mid = case.get('monomer_id')
    if mid not in pool['eligible_ids'] or args.pilot.resolve() != CASE_ROOT/(mid+'.json'):
        raise ValueError('exact one eligible campaign case path required')
    expected = helper('grid400_bank_case').prepare_case(inputs, pool, mid)
    if args.remaining_case:
        bound = load_checkpoint(args)
        expected = helper('audited_bank').bind_next_case(expected,
            bound['checkpoint_raw'], bound['checkpoint_sha256'],
            bound['context_raw'], bound['context_sha256'], bound['proof_blobs'])
    elif (expected['dataset_mode'] != 'fresh' or any(getattr(args, name, None) is not None for name in (
            'bank_context', 'bank_context_sha256', 'bank_checkpoint', 'bank_checkpoint_sha256'))):
        raise ValueError('only the first eligible case can bootstrap without prior context')
    if helper('audited_bank').encode(case) != helper('audited_bank').encode(expected):
        raise ValueError('entire bank case differs from measured inputs and ordinary-audit reconstruction')
    return expected
