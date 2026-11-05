"""Science-launch admission seams and argument routing, never calculations."""
import copy
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_science_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['s20', 's40', 's60', 's80', 's120'])
def test_science_route_requires_the_exact_case_bound_embers_profile(name):
    launcher = load('phoenix_pilot_cb63305')
    profile = launcher.helper('grid400_bank_policy').execution_profile(name)
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs=str(profile['cores']), NumNodes='1', MinMemoryNode=str(profile['job_memory_GiB'])+'G',
        TimeLimit=profile['time_limit'])
    launcher.admit_allocation(allocation, execution_profile=profile, producer='grid400',
                             campaign_scope='eligible-bank')
    with pytest.raises(ValueError):
        launcher.admit_allocation(allocation, execution_profile=profile, producer='grid400')
    with pytest.raises(ValueError):
        launcher.admit_allocation(allocation | dict(TimeLimit='09:00:00'),
            execution_profile=profile, producer='grid400', campaign_scope='eligible-bank')


def preflight(remaining):
    fixture = load('test_grid400_dispatch_and_audit')
    record, draft, kwargs = fixture.preflight()
    profile = fixture.bootstrap.helper('grid400_bank_policy').execution_profile('s120')
    for document in (record, draft):
        document.update(campaign_scope='eligible-bank', dataset_mode='append' if remaining else 'fresh',
                        monomer_id='monomer-0002', execution_profile=copy.deepcopy(profile))
    record['status'] = ('PASS_SELECTED_BANK_CASE_WORKER_PREFLIGHT_ONLY' if remaining
                        else 'PASS_SELECTED_FRESH_WORKER_PREFLIGHT_ONLY')
    record['task_config'] = dict(ncores=24, memory_GiB=120, internal_retries=0)
    kwargs.update(campaign_scope='eligible-bank', remaining_case=remaining)
    return fixture.bootstrap, record, draft, kwargs


@pytest.mark.parametrize('remaining', [False, True])
def test_worker_preflight_matches_explicit_fresh_or_append_bank_mode(remaining):
    launcher, record, draft, kwargs = preflight(remaining)
    launcher.validate_preflight(record, draft, **kwargs)
    with pytest.raises(ValueError):
        launcher.validate_preflight(record, draft, **(kwargs | dict(campaign_scope='default')))


@pytest.mark.parametrize('change', ['scope', 'mode', 'memory', 'retries', 'descriptor', 'sources'])
def test_wrong_preflight_cannot_authorize_a_bank_science_session(change):
    launcher, record, draft, kwargs = preflight(True)
    if change == 'scope':
        record['campaign_scope'] = 'default'
    elif change == 'mode':
        record['dataset_mode'] = 'fresh'
    elif change == 'memory':
        record['task_config']['memory_GiB'] = 80
    elif change == 'retries':
        record['task_config']['internal_retries'] = 1
    elif change == 'descriptor':
        record['descriptor_verification']['status'] = 'REFUSED_DESCRIPTOR_PLAN'
    else:
        record['operational_sources_sha256'] = {}
    with pytest.raises(ValueError):
        launcher.validate_preflight(record, draft, **kwargs)


@pytest.mark.parametrize('remaining', [False, True])
def test_unsealed_bank_refuses_before_checkpoint_or_server_access(monkeypatch, remaining):
    launcher, campaign = load('phoenix_pilot_cb63305'), load('bank_campaign')
    monkeypatch.setattr(campaign, 'ALLOCATION_SHA256', None)
    calls = []
    monkeypatch.setattr(campaign, 'load_checkpoint', lambda args: calls.append('checkpoint'))
    original = launcher.helper
    monkeypatch.setattr(launcher, 'helper', lambda name: campaign if name == 'bank_campaign' else original(name))
    args = NS(producer='grid400', campaign_scope='eligible-bank', remaining_case=remaining)
    with pytest.raises(ValueError, match='has not been sealed'):
        launcher.load_append_inputs(args)
    assert calls == []


def test_historical_acceptance_inputs_cannot_enter_new_bank_namespace(monkeypatch):
    launcher = load('phoenix_pilot_cb63305')
    original = launcher.helper
    campaign = NS(authority=lambda: {'unit': 'not authority'}, load_checkpoint=lambda args: None)
    monkeypatch.setattr(launcher, 'helper', lambda name: campaign if name == 'bank_campaign' else original(name))
    args = NS(producer='grid400', campaign_scope='eligible-bank', remaining_case=True,
              accepted_envelope=Path('/unit/old-forensic-envelope.json'))
    with pytest.raises(ValueError, match='historical acceptance'):
        launcher.load_append_inputs(args)
    with pytest.raises(ValueError, match='explicit bank'):
        launcher.load_append_inputs(NS(producer='grid400', remaining_case=False,
                                       bank_context=Path('/unit/bank-context.json')))


@pytest.mark.parametrize('append', [False, True])
def test_science_wrapper_forwards_bank_context_without_executing_any_payload(append):
    script = Path(__file__).with_name('phoenix_pilot_cb63305.sbatch')
    prefix = script.read_text().split('MANAGER=/storage/project/r-cs207-0/awallace43/miniconda/envs/p4qca', 1)[0]
    case = '/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-bank001/cases/monomer-0002.json'
    args = ['/unit/source', 'a'*40, case, '/unit/preflight.json', '/unit/run',
            '--grid400-bank-append' if append else '--grid400-bank-fresh']
    if append:
        args += ['/unit/bank-context.json', 'b'*64, '/unit/bank-checkpoint.json', 'c'*64]
    result = subprocess.run(['bash', '-c', prefix+'\nprintf "%s\\n" "${EXTRA[@]}"',
                             'unit', *args], capture_output=True, text=True, check=True)
    flags = result.stdout.splitlines()
    assert flags[:4] == ['--producer', 'grid400', '--campaign-scope', 'eligible-bank']
    assert ('--remaining-case' in flags) is append
    if append:
        assert flags[flags.index('--bank-checkpoint')+1] == '/unit/bank-checkpoint.json'
