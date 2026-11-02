"""Pure preflight/launch bindings; live singleton transport has separate tests."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location(
    'expanded_bootstrap_fixture', Path(__file__).with_name('phoenix_pilot_cb63305.py'))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def fixture():
    draft = dict(monomer_id='monomer-0006', runtime_input_sha256='runtime-only-fixture',
        producer=dict(scientific_commit='cb63305-unit-fixture',
                      implementation_sha256={'synthetic': 'not scientific evidence'}))
    kwargs = dict(manifest_sha='manifest-fixture', qualification_sha='qualification-fixture',
        pilot_sha='pilot-fixture', worker_sources={'source.py': 'synthetic'},
        transport={'synthetic': 'unit-only'})
    preflight = dict(status='PASS_EXPANDED_WORKER_PREFLIGHT_ONLY', runtime_unchanged=True,
        monomer_id=draft['monomer_id'], scientific_build_commit=draft['producer']['scientific_commit'],
        build_identity_sha256=kwargs['manifest_sha'], qualification_sha256=kwargs['qualification_sha'],
        pilot_sha256=kwargs['pilot_sha'], runtime_input_sha256=draft['runtime_input_sha256'],
        operational_sources_sha256=kwargs['worker_sources'], transport_binding=kwargs['transport'],
        science=dict(implementation={'sha256': draft['producer']['implementation_sha256']}),
        descriptor_verification=dict(status='PASS_DESCRIPTOR_PLAN_ONLY'))
    return preflight, draft, kwargs


def test_worker_discovery_does_not_mean_molecular_acceptance():
    preflight, draft, kwargs = fixture()
    pilot.validate_preflight(preflight, draft, **kwargs)
    assert preflight['status'] != 'PASS_SINGLE_PILOT_EVIDENCE'


def test_cap32_fresh_discovery_not_inferred_as_owned_append_from_s40():
    preflight, draft, kwargs = fixture()
    cases = pilot.helper('case_cb63305')
    profile = dict(name='s40', **cases.PROFILES['s40'], account='gts-cs207-chemx',
                   partition='cpu-small', qos='embers', concurrency=1)
    draft.update(producer_profile='ed561666', dataset_mode='fresh', execution_profile=profile)
    preflight.update(status='PASS_SELECTED_FRESH_WORKER_PREFLIGHT_ONLY',
        producer_profile='ed561666', dataset_mode='fresh', execution_profile=profile,
        task_config=dict(ncores=12, memory_GiB=40, internal_retries=0))
    pilot.validate_preflight(preflight, draft, producer='ed561666', **kwargs)
    for changed in (dict(preflight, status='PASS_BANK_CASE_WORKER_PREFLIGHT_ONLY'),
                    dict(preflight, producer_profile='cb63305'),
                    dict(preflight, dataset_mode='owned354')):
        with pytest.raises(ValueError):
            pilot.validate_preflight(changed, draft, producer='ed561666', **kwargs)
    with pytest.raises(ValueError):
        pilot.validate_preflight(preflight, draft, **kwargs)


@pytest.mark.parametrize('field,value', [
    ('status', 'PASS_WORKER_PREFLIGHT_ONLY'), ('runtime_unchanged', False),
    ('monomer_id', 'monomer-0004'), ('scientific_build_commit', 'acf54'),
    ('build_identity_sha256', 'foreign'), ('qualification_sha256', '197-case-fixture'),
    ('pilot_sha256', 'foreign'), ('runtime_input_sha256', 'changed-request'),
    ('operational_sources_sha256', {}), ('transport_binding', {}),
    ('science', {'implementation': {'sha256': {'foreign': 'runtime'}}}),
    ('descriptor_verification', {'status': 'REFUSED_DESCRIPTOR_PLAN'})])
def test_old_foreign_or_changed_worker_preflight_cannot_authorize_a_live_run(field, value):
    preflight, draft, kwargs = fixture()
    preflight[field] = copy.deepcopy(value)
    with pytest.raises(ValueError, match='exact expanded'):
        pilot.validate_preflight(preflight, draft, **kwargs)


def test_only_exact_approved_embers_molecular_shape_is_admitted():
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='02:00:00')
    pilot.admit_allocation(allocation)
    for key, value in [('QOS', 'inferno'), ('Account', 'foreign'), ('Partition', 'cpu-large'),
                       ('NumCPUs', '8'), ('NumNodes', '2'), ('MinMemoryNode', '20G'),
                       ('TimeLimit', '08:00:00')]:
        with pytest.raises(ValueError, match='exact effective'):
            pilot.admit_allocation(allocation | {key: value})


@pytest.mark.parametrize('name', ['s20', 's40', 's80'])
def test_remaining_molecular_shape_cannot_borrow_a_preflight_or_old_resource_route(name):
    cases = pilot.helper('case_cb63305')
    profile = dict(name=name, **cases.PROFILES[name], account='gts-cs207-chemx',
                   partition='cpu-small', qos='embers', concurrency=1)
    allocation = dict(Account=profile['account'], QOS='embers', Partition='cpu-small',
        NumCPUs=str(profile['cores']), NumNodes='1', MinMemoryNode=str(profile['job_memory_GiB'])+'G',
        TimeLimit=profile['time_limit'])
    pilot.admit_allocation(allocation, execution_profile=profile)
    for key, value in [('QOS', 'inferno'), ('NumCPUs', '4'), ('MinMemoryNode', '24G'),
                       ('TimeLimit', '00:45:00')]:
        with pytest.raises(ValueError):
            pilot.admit_allocation(allocation | {key: value}, execution_profile=profile)
    changed = dict(profile, scratch_bytes=1)
    with pytest.raises(ValueError, match='profile changed'):
        pilot.admit_allocation(allocation, execution_profile=changed)
    preflight, draft, kwargs = fixture()
    draft['execution_profile'] = profile
    preflight.update(status='PASS_BANK_CASE_WORKER_PREFLIGHT_ONLY', execution_profile=profile,
        task_config=dict(ncores=profile['cores'], memory_GiB=profile['science_memory_GiB'],
                         internal_retries=0))
    pilot.validate_preflight(preflight, draft, **kwargs)
    preflight['task_config']['internal_retries'] = 1
    with pytest.raises(ValueError, match='scientific worker profile'):
        pilot.validate_preflight(preflight, draft, **kwargs)


def append_args():
    return NS(remaining_case=False, owned_checkpoint=None, checkpoint_sha256=None,
              accepted_audit=None, accepted_outcome=None, accepted_pilot=None, accepted_envelope=None)


def test_owned_append_arguments_cannot_be_ignored_or_inferred():
    args = append_args()
    assert pilot.load_append_inputs(args) is None
    args.owned_checkpoint = Path('not-opened')
    with pytest.raises(ValueError, match='cannot be ignored'):
        pilot.load_append_inputs(args)
    args.remaining_case = True
    with pytest.raises(ValueError, match='all closed acceptance receipts'):
        pilot.load_append_inputs(args)


def test_cap32_append_requires_closed_receipts_and_cannot_be_inferred_in_fresh_mode():
    args = append_args()
    args.producer = 'ed561666'
    assert pilot.load_append_inputs(args) is None
    args.remaining_case = True
    with pytest.raises(ValueError, match='all closed acceptance receipts'):
        pilot.load_append_inputs(args)
    args.remaining_case = False
    args.accepted_audit = Path('not-read')
    with pytest.raises(ValueError, match='cannot be ignored'):
        pilot.load_append_inputs(args)


@pytest.mark.parametrize('producer', ['cb63305', 'ed561666'])
def test_complete_append_input_bytes_are_verified_before_dispatch(monkeypatch, tmp_path, producer):
    args = append_args()
    args.producer = producer
    args.remaining_case = True
    args.checkpoint_sha256 = 'synthetic-checkpoint-only'
    args.owned_checkpoint = tmp_path/'checkpoint.json'
    args.owned_checkpoint.write_bytes(b'SYNTHETIC_BASELINE')
    roles = ('audit', 'outcome', 'pilot', 'envelope')
    for role in roles:
        path = tmp_path/(role+'.json')
        path.write_bytes(('SYNTHETIC_'+role).encode())
        setattr(args, 'accepted_'+role, path)
    effects = []
    def accepted(blobs, **selected):
        assert selected == ({} if producer == 'cb63305' else dict(producer=producer))
        effects.append('closed_receipts')
        assert set(blobs) == set(roles)
        return {'synthetic_accepted': True}
    def checkpoint(raw, sha, acceptance, **selected):
        assert selected == ({} if producer == 'cb63305' else dict(producer=producer))
        effects.append('closed_checkpoint')
        assert raw == b'SYNTHETIC_BASELINE' and sha == args.checkpoint_sha256
        assert acceptance == {'synthetic_accepted': True}
    monkeypatch.setattr(pilot, 'helper', lambda name:
        NS(verify_first_acceptance=accepted) if name == 'owned_dataset_cb63305'
        else NS(validate_checkpoint=checkpoint))
    result = pilot.load_append_inputs(args)
    assert effects == ['closed_receipts', 'closed_checkpoint']
    assert result['checkpoint_raw'] == b'SYNTHETIC_BASELINE'
    assert result['first_receipt_blobs']['audit'] == b'SYNTHETIC_audit'


@pytest.mark.parametrize('name', ['s20', 's40', 's80'])
def test_cap32_append_preflight_explicitly_binds_mode_context_and_finite_profile(name):
    preflight, draft, kwargs = fixture()
    cases = pilot.helper('case_cb63305')
    context = pilot.helper('owned_dataset_cb63305').context('ed561666')
    profile = dict(name=name, **cases.PROFILES[name], account='gts-cs207-chemx',
                   partition='cpu-small', qos='embers', concurrency=1)
    draft.update(producer_profile='ed561666', dataset_mode='append', dataset_id=355,
        monomer_id='monomer-0006', required_first_pilot_audit_sha256=context['accepted_sha256']['audit'],
        execution_profile=profile)
    preflight.update(status='PASS_SELECTED_BANK_CASE_WORKER_PREFLIGHT_ONLY',
        producer_profile='ed561666', dataset_mode='append', monomer_id='monomer-0006',
        execution_profile=profile, task_config=dict(ncores=profile['cores'],
            memory_GiB=profile['science_memory_GiB'], internal_retries=0))
    pilot.validate_preflight(preflight, draft, producer='ed561666', remaining_case=True, **kwargs)
    with pytest.raises(ValueError):
        pilot.validate_preflight(preflight, draft, producer='ed561666', **kwargs)
    with pytest.raises(ValueError, match='owned355'):
        pilot.validate_preflight(preflight, dict(draft, dataset_id=354),
                                 producer='ed561666', remaining_case=True, **kwargs)


def test_dispatch_cannot_silently_change_the_approved_server_uri_or_owner():
    config = dict(username='awallace43', fractal_uri='http://128.61.254.96:7777',
                  password='SYNTHETIC_NOT_A_CREDENTIAL')
    pilot.validate_server_config(config)
    for changed in (dict(config, username='foreign'),
                    dict(config, fractal_uri='http://unapproved.invalid:7777')):
        with pytest.raises(ValueError, match='deployment URI'):
            pilot.validate_server_config(changed)


def test_help_is_pure_and_wrapper_is_term_safe_with_no_false_acceptance_marker():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', pilot.__file__, '--help'],
                            text=True, capture_output=True)
    assert result.returncode == 0 and '--worker-preflight' in result.stdout
    text = Path(pilot.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in text and '#SBATCH --signal=TERM@180' in text
    assert 'ccasp_wait_child "$child" 20' in text
    assert text.index('ccasp_wait_child') < text.index('EXPANDED_PILOT_DELIVERED_NOT_QUALIFIED')
    assert 'COMPLETED' not in text
