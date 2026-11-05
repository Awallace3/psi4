"""Pure dispatch/audit gates; no native execution, manager or QCA access."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load('_grid400_bootstrap_test', 'phoenix_pilot_cb63305.py')
audit = load('_grid400_audit_test', 'runtime_audit_cb63305.py')
classifier = load('_grid400_cli_test', 'runtime_audit.py')
policy = bootstrap.helper('grid400_policy')


def allocation():
    return dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                NumCPUs='16', NumNodes='1', MinMemoryNode='96G', TimeLimit='04:00:00')


def preflight():
    profile = policy.execution_profile()
    implementation = dict(core='a'*64, isapol_python='b'*64)
    draft = dict(producer_profile='grid400', dataset_mode='fresh', monomer_id='monomer-0008',
        execution_profile=profile, producer=dict(scientific_commit='SYNTHETIC_ONLY',
            implementation_sha256=implementation), runtime_input_sha256='r'*64)
    kwargs = dict(manifest_sha='m'*64, qualification_sha='q'*64, pilot_sha='p'*64,
        worker_sources={'synthetic_source': 'x'*64}, transport={'synthetic': 'not live'},
        producer='grid400', remaining_case=False)
    record = dict(status='PASS_SELECTED_FRESH_WORKER_PREFLIGHT_ONLY', runtime_unchanged=True,
        producer_profile='grid400', dataset_mode='fresh', monomer_id='monomer-0008',
        scientific_build_commit='SYNTHETIC_ONLY', build_identity_sha256=kwargs['manifest_sha'],
        qualification_sha256=kwargs['qualification_sha'], pilot_sha256=kwargs['pilot_sha'],
        runtime_input_sha256=draft['runtime_input_sha256'],
        operational_sources_sha256=kwargs['worker_sources'], transport_binding=kwargs['transport'],
        science=dict(implementation=dict(sha256=implementation)),
        descriptor_verification=dict(status='PASS_DESCRIPTOR_PLAN_ONLY'),
        execution_profile=copy.deepcopy(profile),
        task_config=dict(ncores=16, memory_GiB=60, internal_retries=0))
    return record, draft, kwargs


def test_science_route_is_exactly_the_approved_four_hour_shape():
    bootstrap.admit_allocation(allocation(), producer='grid400', execution_profile=policy.execution_profile())
    with pytest.raises(ValueError):
        bootstrap.admit_allocation(allocation(), producer='grid400')
    for field, value in (('TimeLimit', '08:00:00'), ('MinMemoryNode', '144G'),
                         ('NumCPUs', '24'), ('QOS', 'inferno')):
        with pytest.raises(ValueError):
            bootstrap.admit_allocation(allocation() | {field: value}, producer='grid400',
                                        execution_profile=policy.execution_profile())


@pytest.mark.parametrize('field,value', [
    ('Account', 'foreign'), ('QOS', 'inferno'), ('Partition', 'cpu-large'),
    ('NumCPUs', '16'), ('NumNodes', '2'), ('MinMemoryNode', '96G'),
    ('TimeLimit', '04:00:00')])
def test_audits_own_allocation_must_match_approved_route_and_shape(monkeypatch, field, value):
    own = allocation() | dict(NumCPUs='4', MinMemoryNode='24G', TimeLimit='00:45:00')
    own[field] = value
    args = append_args()
    args.advance_checkpoint = False
    monkeypatch.setattr(audit.argparse.ArgumentParser, 'parse_args', lambda self: args)
    checks = NS(allocation=lambda: own)
    bootstrap_stub = NS(load_append_inputs=lambda args: None)
    original = audit.helper
    monkeypatch.setattr(audit, 'helper', lambda name: (
        checks if name == 'phoenix_installed_checks' else
        bootstrap_stub if name == 'phoenix_pilot_cb63305' else original(name)))
    with pytest.raises(ValueError, match='audit allocation'):
        audit.main()


def test_audit_allocation_admits_only_the_exact_grid400_shape():
    own = allocation() | dict(NumCPUs='4', MinMemoryNode='24G', TimeLimit='00:45:00')
    audit.admit_audit_allocation(own, producer='grid400')
    for field in own:
        with pytest.raises(ValueError):
            audit.admit_audit_allocation({k: v for k, v in own.items() if k != field},
                                         producer='grid400')


def test_grid400_requires_its_matching_fresh_discovery_record():
    record, draft, kwargs = preflight()
    bootstrap.validate_preflight(record, draft, **kwargs)


@pytest.mark.parametrize('change', ['mode', 'producer', 'monomer', 'shape', 'task',
                                   'descriptor', 'implementation', 'sources', 'remaining'])
def test_other_preflights_or_append_modes_cannot_authorize_science(change):
    record, draft, kwargs = preflight()
    if change == 'mode':
        record['dataset_mode'] = 'append'
    elif change == 'producer':
        record['producer_profile'] = 'ed561666'
    elif change == 'monomer':
        draft['monomer_id'] = record['monomer_id'] = 'monomer-0009'
    elif change == 'shape':
        record['execution_profile']['time_limit'] = '08:00:00'
    elif change == 'task':
        record['task_config']['internal_retries'] = 1
    elif change == 'descriptor':
        record['descriptor_verification']['status'] = 'REFUSED_DESCRIPTOR_PLAN'
    elif change == 'implementation':
        record['science']['implementation']['sha256'] = {}
    elif change == 'sources':
        record['operational_sources_sha256'] = {}
    else:
        kwargs['remaining_case'] = True
    with pytest.raises(ValueError):
        bootstrap.validate_preflight(record, draft, **kwargs)


def append_args():
    return NS(producer='grid400', remaining_case=False, owned_checkpoint=None,
        checkpoint_sha256=None, checkpoint_proofs=None,
        campaign_scope='default', bank_context=None, bank_context_sha256=None,
        bank_checkpoint=None, bank_checkpoint_sha256=None,
        advance_checkpoint=False, advance_bank_checkpoint=False,
        accepted_audit=None, accepted_outcome=None, accepted_pilot=None, accepted_envelope=None)


def test_no_owned_checkpoint_receipt_or_advancement_mode_can_be_ignored(monkeypatch):
    assert bootstrap.load_append_inputs(append_args()) is None
    for field in ('remaining_case', 'owned_checkpoint', 'checkpoint_sha256',
                  'checkpoint_proofs', 'accepted_audit', 'accepted_outcome',
                  'accepted_pilot', 'accepted_envelope'):
        args = append_args()
        setattr(args, field, True if field == 'remaining_case' else 'not authorized')
        with pytest.raises(ValueError):
            bootstrap.load_append_inputs(args)
    args = append_args()
    args.advance_checkpoint = True
    monkeypatch.setattr(audit.argparse.ArgumentParser, 'parse_args', lambda self: args)
    with pytest.raises(ValueError, match='checkpoint advancement'):
        audit.main()


def delivery():
    pins = bootstrap.helper('pilot_cb63305').producer_pins('grid400')
    tag = pins['namespace']+'monomer-0008-123'
    outcome = dict(schema='native-expanded-pilot-outcome/1',
        status='DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION',
        producer_profile='grid400', dataset_mode='fresh', monomer_id='monomer-0008',
        dataset_name=pins['dataset_name'], specification_name=pins['specification_name'],
        runtime_unchanged=True, find_existing=False, maximum_records=1,
        dataset_id=9999, record_id=9998, task_id=77, manager='SYNTHETIC_MANAGER',
        compute_tag=tag, allocation={'JobId': '123'},
        scratch_path=str(audit.SCRATCH/tag))
    stats = dict(total_successful_tasks=1, total_failed_tasks=0, total_rejected_tasks=0, active_tasks=0)
    delivered = dict(record_id=9998, task_id=77, manager='SYNTHETIC_MANAGER',
        outcome='accepted', interrupted=False, local_statistics=stats)
    created = dict(dataset_id=9999, record_id=9998, compute_tag=tag, n_inserted=1, n_existing=0)
    claimed = dict(record_id=9998, task_id=77, manager='SYNTHETIC_MANAGER')
    server = dict(record_id=9998, status='complete', history_count=1, history_managers=['SYNTHETIC_MANAGER'])
    return outcome, delivered, created, claimed, server


def test_selected_audit_binds_fresh_scope_and_exact_scratch_tag():
    docs = delivery()
    audit.validate_delivery(*docs, producer='grid400')
    assert audit.scratch_for_outcome(docs[0], expected_monomer='monomer-0008', producer='grid400') == (
        Path(docs[0]['scratch_path']))
    with pytest.raises(ValueError):
        audit.validate_delivery(*docs, producer='grid400', owned354=True)
    with pytest.raises(ValueError):
        audit.scratch_for_outcome(docs[0], expected_monomer='monomer-0008',
                                  producer='grid400', remaining_case=True)


@pytest.mark.parametrize('field,value', [
    ('dataset_id', 355), ('record_id', 5234326), ('dataset_mode', 'append'),
    ('monomer_id', 'monomer-0009'), ('producer_profile', 'ed561666'),
    ('dataset_name', 'foreign'), ('specification_name', 'foreign')])
def test_old_id_or_mislabelled_delivery_cannot_be_audited_as_grid400(field, value):
    docs = delivery()
    docs[0][field] = value
    with pytest.raises(ValueError):
        audit.validate_delivery(*docs, producer='grid400')


def test_decoded_cli_classifier_keeps_exact_isolation_and_explicit16_60_shape():
    env, scratch = '/synthetic/env', '/synthetic/scratch'
    args = [env+'/bin/python', '-I', '-B', '-X', 'pycache_prefix='+scratch+'/cache', env+'/bin/psi4',
            '--scratch', scratch+'/task', '--nthread', '16', '--memory', '60.0GiB',
            '--qcschema', 'data.msgpack']
    execution = dict(syscall='execve("'+args[0]+'", '+json.dumps(args)+', 0x0 /* env */) = 0')
    assert classifier.cli_role(execution, env=env, scratch=scratch,
                                worker_shape=(16, 60.)) == 'molecular_qcschema'
    with pytest.raises(ValueError):
        classifier.cli_role(execution, env=env, scratch=scratch)


@pytest.mark.parametrize('label', ['cb63305f41', 'ed561666fc', 'grid400-1aaa9e0a30'])
def test_shared_shell_paths_keep_old_layouts_and_bind_new_installed_receipt(label):
    script = Path(__file__).with_name('producer_paths.sh')
    command = ['bash', '--noprofile', '--norc', '-c',
        'set -euo pipefail; source "$1"; ccasp_select_paths "$2"; '
        'printf "%s\\n" "$CCASP_BUILD" "$CCASP_QUALIFICATION" "$CCASP_INPUTS" "$CCASP_DESCRIPTORS"',
        'path-test', str(script), label]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    root = '/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-'+label
    qualification = '-installed001' if label == 'grid400-1aaa9e0a30' else '-checks001'
    assert result.stdout.splitlines() == [
        root+'-build001', root+qualification+'/payload/installed-validation.json',
        root+'-preparation001', root+'-descriptors001/payload']
