"""Measured fresh monomer0008 cohort; never an owned355 append/recovery path."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

BANK_SHA256 = 'ace7fc9aae6c63ff2a5290672d9590acd15353d1bc00588b3d01b4b9c00d5133'
ALLOCATION_SHA256 = '4e5eb4fd41ec151911da78bde129685fff05ae5707d5a462c622b16a4d385bce'
REPORT_SHA256 = '33004de9c13dbd6d9681f5ecd6660141a9ee226b7b3eb3e1f235b9d415b74c5c'
RUNTIME_SHA256 = '16ce7c1347a9eccbb64c1e9cc982e75f37ccef4ac9845a486a7d87fb40c0809e'
RESOURCE_SHA256 = '6d999a843cc3a6f172eb14df0a533cd0a84d4c1cfc17a8b30f3ee274301fba05'
DESCRIPTOR_OPS = 'a25c3c51509b24e5a5ed4cef8eebf5e4fcbaa347'
DESCRIPTOR_JOB = '15081893'
REPORT_NAME = 'monomer-0008-descriptors.json'
DESCRIPTOR_SOURCES = (
    'phoenix_descriptor_bank_cb63305.py', 'phoenix_descriptor_bank_cb63305.sbatch',
    'descriptor_probe_cb63305.py', 'descriptor_probe.py', 'descriptor_admission_cb63305.py',
    'descriptor_scratch_cb63305.py', 'prepare_descriptor_bank_cb63305.py',
    'propose_descriptor_resources_cb63305.py', 'term_wait.sh', 'qualified_cb63305.py',
    'resource_plan_cb63305.py', 'phoenix_installed_checks.py', 'descriptor_profiles.py',
    'qualified_grid400.py', 'resource_plan_ed561666.py', 'resource_plan_grid400.py')


def helper(name):
    if name not in ('grid400_policy', 'prepare_descriptor_bank_cb63305',
                    'descriptor_admission_cb63305', 'descriptor_profiles', 'resource_plan_grid400'):
        raise ValueError('closed measured grid400 cohort helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_grid400_cohort_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def pins():
    """Actual measured singleton identities, not caller-configured overrides."""
    policy = helper('grid400_policy')
    return dict(producer='grid400', monomer=policy.MONOMER,
        dataset_name='S66x8-native-CCASP-grid400-1aaa9e0a30-request3-v1',
        specification_name='pbe0-atz-native-grid400-request3',
        bank_sha256=BANK_SHA256, allocation_sha256=ALLOCATION_SHA256,
        descriptor_ops=DESCRIPTOR_OPS, descriptor_job=DESCRIPTOR_JOB,
        selected_report_sha256=REPORT_SHA256, runtime_input_sha256=RUNTIME_SHA256,
        schema='native-selected-molecular-pilot/1',
        protected_datasets=policy.PROTECTED_DATASETS, protected_records=policy.PROTECTED_RECORDS,
        namespace='ccasp-native-grid400-', worker_namespace='ccasp-native-worker-grid400-')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def descriptor_operation_pins():
    """Bind the measured historical descriptor snapshot, not today's files."""
    root = Path(__file__).resolve().parents[2]
    return {name: digest(subprocess.check_output(
        ['git', 'show', DESCRIPTOR_OPS+':devtools/isapol_s66_campaign/'+name], cwd=root))
        for name in DESCRIPTOR_SOURCES}


def verify_admitted_bank(inputs, receipts, resources, bank_raw, allocation_raw, reports):
    """Reconstruct the one raw runtime and its measured descriptor proof."""
    if (type(bank_raw) is not bytes or type(allocation_raw) is not bytes
            or type(resources) is not bytes or digest(bank_raw) != BANK_SHA256
            or digest(allocation_raw) != ALLOCATION_SHA256
            or digest(resources) != RESOURCE_SHA256
            or type(reports) is not dict or set(reports) != {REPORT_NAME}
            or type(reports[REPORT_NAME]) is not bytes or digest(reports[REPORT_NAME]) != REPORT_SHA256):
        raise ValueError('exact measured singleton grid400 bank/allocation/report/resource bytes required')
    prepare = helper('prepare_descriptor_bank_cb63305')
    bank = prepare.prepare_grid400_pilot_bank(inputs, receipts, resources)
    if bank != json.loads(bank_raw):
        raise ValueError('selected grid400 raw bank does not reproduce')
    admission = helper('descriptor_admission_cb63305')
    planner = helper('resource_plan_grid400')
    allocation = json.loads(allocation_raw)
    if (allocation.get('schema') != 'native-campaign-selected-descriptor-allocation/1'
            or allocation.get('status') != 'PASS_SELECTED_MONOMER_DESCRIPTOR_PLAN_ONLY'
            or allocation.get('runtime_unchanged') is not True
            or allocation.get('scientific_build_commit') != planner.SCIENTIFIC_COMMIT
            or allocation.get('operational_commit') != DESCRIPTOR_OPS
            or allocation.get('policy_id') != planner.POLICY
            or allocation.get('bank_sha256') != BANK_SHA256
            or allocation.get('producer') != bank['producer']
            or allocation.get('source_policy_sha256') != planner.FILES
            or allocation.get('operational_sources_sha256') != descriptor_operation_pins()
            or allocation.get('allocation', {}).get('JobId') != DESCRIPTOR_JOB
            or admission.verify_grid400_coverage(allocation.get('reports')) != allocation['status']):
        raise ValueError('exact successful measured grid400 descriptor allocation required')
    row, = bank['monomers']
    measurement, = allocation['reports']
    if (row.get('monomer_id') != 'monomer-0008'
            or row.get('runtime_input_sha256') != RUNTIME_SHA256
            or prepare.json_digest(row['runtime_input']) != RUNTIME_SHA256
            or measurement.get('descriptor_report') != REPORT_NAME
            or measurement.get('descriptor_report_sha256') != REPORT_SHA256
            or measurement.get('runtime_input_sha256') != RUNTIME_SHA256):
        raise ValueError('selected grid400 runtime/report identity differs')
    sources = helper('descriptor_profiles').profile('grid400')['child_sources']
    result = admission.verify_descriptor(json.loads(reports[REPORT_NAME]), row,
        json.loads(receipts['imported_identity']), policy=planner.FILES,
        operational_sources_sha256={name: allocation['operational_sources_sha256'][name]
                                   for name in sources},
        runtime_sha256=RUNTIME_SHA256, producer='grid400')
    if result['status'] != 'PASS_DESCRIPTOR_PLAN_ONLY' or any(
            measurement.get(key) != value for key, value in result.items()):
        raise ValueError('selected grid400 descriptor admission differs')
    helper('grid400_policy').validate_execution_profile(row)
    return bank, allocation
