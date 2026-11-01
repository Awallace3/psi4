"""Exact first-methanol checkpoint: descriptor plans are not molecular results."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

WATER_AUDIT_SHA256 = '21f7e06f759b4dbd9171bcecdcd46a417ad8afe8f06474248c58e116249fd99f'
WATER_ENVELOPE_SHA256 = '1a47ac18fc2be29268cab4c37edcd9c9f5e69cd859534bb0ac9dd46264338bb7'
DESCRIPTOR_ALLOCATION_SHA256 = '40f30c40c550f5150c736a1223f83ca1804c7dcbf2b873e0bdddaed45bc50b34'
METHANOL_DESCRIPTOR_SHA256 = 'fea16baabf2ec34ecb025550d850397d48ea4a2804e09434e097e3512feb4e23'
METHANOL_PILOT_SHA256 = 'e89792fa813e516f2d97c04215d613cc8b3bdf6ba4f12f2e396491e48c256203'
DESCRIPTOR_COMMIT = 'c97d8859c1ca0db421259578f43e98188c0a05bb'
PATH_ARGUMENTS = ('append_baseline', 'previous_audit', 'previous_envelope',
                  'descriptor_allocation', 'descriptor_report')


def helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path, expected=None):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError('checkpoint receipt bytes differ: '+Path(path).name)
    return json.loads(raw), sha


def add_arguments(parser):
    for name in PATH_ARGUMENTS:
        parser.add_argument('--'+name.replace('_', '-'), type=Path)


def verify(args):
    """Return a fully bound checkpoint or None for the archived water mode."""
    paths = [getattr(args, name, None) for name in PATH_ARGUMENTS]
    if not any(paths):
        return None
    if not all(paths):
        raise ValueError('all append checkpoint receipt paths required')
    guard = helper('dataset_guard')
    probe = helper('phoenix_worker_probe')
    draft, pilot_sha = load(args.pilot, METHANOL_PILOT_SHA256)
    audit, audit_sha = load(args.previous_audit, WATER_AUDIT_SHA256)
    water, envelope_sha = load(args.previous_envelope, WATER_ENVELOPE_SHA256)
    allocation, allocation_sha = load(args.descriptor_allocation, DESCRIPTOR_ALLOCATION_SHA256)
    descriptor, descriptor_sha = load(args.descriptor_report, METHANOL_DESCRIPTOR_SHA256)
    baseline, baseline_sha = load(args.append_baseline)
    if (audit['status'] != 'PASS_SINGLE_PILOT_EVIDENCE'
            or audit['record_id'] != guard.WATER_RECORD or audit['runtime_unchanged'] is not True
            or audit['scientific_build_commit'] != guard.SCIENTIFIC_COMMIT
            or allocation['status'] != 'PASS_DESCRIPTOR_PLANS_ONLY'
            or allocation['runtime_unchanged'] is not True
            or allocation['scientific_build_commit'] != guard.SCIENTIFIC_COMMIT
            or allocation['operational_commit'] != DESCRIPTOR_COMMIT
            or descriptor['status'] != 'PASS_DESCRIPTOR_PLAN_ONLY'
            or descriptor['plan']['status'] != 'WITHIN_PREDICTED_LEDGER_LIMITS'
            or descriptor['plan']['refusals'] or descriptor['coordinate_change_bohr'] != 0
            or draft['monomer_id'] != 'monomer-0004'):
        raise ValueError('only descriptor-qualified first methanol after audited water')
    selected = [r for r in allocation['reports'] if r['monomer_id'] == draft['monomer_id']]
    runtime = probe.runtime_input(draft)
    runtime_sha = hashlib.sha256(json.dumps(runtime, allow_nan=False).encode()).hexdigest()
    if (len(selected) != 1 or selected[0]['pilot_sha256'] != pilot_sha
            or selected[0]['descriptor_report_sha256'] != descriptor_sha
            or selected[0]['runtime_input_sha256'] != runtime_sha
            or descriptor['runtime_input_sha256'] != runtime_sha
            or descriptor['implementation']['sha256'] != water['implementation']['sha256']
            or draft['build_identity_sha256'] != allocation['build_identity_sha256']
            or draft['source_files_sha256']['installed_qualification'] != allocation['qualification_sha256']):
        raise ValueError('cross-wired descriptor/input/build receipt chain')
    spec = copy.deepcopy(draft['specification'])
    if spec.get('protocols'):
        raise ValueError('unexpected draft execution protocol')
    spec['protocols'] = dict(error_correction=dict(default_policy=False))
    if (baseline.get('schema') != 'native-owned-append-baseline/1'
            or baseline['next_monomer_id'] != 'monomer-0004'
            or baseline['previous_audit_sha256'] != audit_sha
            or baseline['previous_envelope_sha256'] != envelope_sha
            or baseline['water_envelope_canonical_sha256'] != guard.digest(water)
            or set(baseline['snapshot']['entries']) != {guard.WATER_ENTRY}
            or set(baseline['snapshot']['records']) != {str(guard.WATER_RECORD)}):
        raise ValueError('approved first-methanol baseline differs')
    guard.validate_owned(
        baseline['snapshot'], authenticated_owner='awallace43',
        implementation=water['implementation']['sha256'], specification=spec,
        water_envelope_sha256=guard.digest(water))
    guard.validate_new_entry(baseline['snapshot'], draft)
    return dict(
        schema='native-owned-append-checkpoint/1', monomer_id='monomer-0004',
        dataset_id=guard.DATASET_ID, pilot_sha256=pilot_sha,
        baseline_sha256=baseline_sha, previous_audit_sha256=audit_sha,
        previous_envelope_sha256=envelope_sha, descriptor_allocation_sha256=allocation_sha,
        descriptor_report_sha256=descriptor_sha, runtime_input_sha256=runtime_sha,
        descriptor_operational_commit=DESCRIPTOR_COMMIT,
        descriptor_job=allocation['allocation']['JobId'],
        source_files_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('append_receipts.py', 'dataset_guard.py', 'phoenix_pilot.py',
                         'dataset_append.py', 'phoenix_worker_probe.py', 'single_claim.py',
                         'native_psi4.sh', 'prepare_append_baseline.py')})
