"""Explicit worker scope/path/resource routing; never installed SCI/discovery."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_worker_test_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['s20', 's40', 's60', 's80', 's120'])
def test_exact_finite_bank_worker_allocations_need_explicit_scope(name):
    worker = load('phoenix_worker_cb63305')
    policy = worker.helper('grid400_bank_policy')
    profile = policy.execution_profile(name)
    allocation = dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
        NumCPUs=str(profile['cores']), NumNodes='1',
        MinMemoryNode=str(profile['job_memory_GiB'])+'G', TimeLimit='00:45:00')
    worker.admit_allocation(allocation, execution_profile=profile, producer='grid400',
                            campaign_scope='eligible-bank')
    with pytest.raises(ValueError):
        worker.admit_allocation(allocation, execution_profile=profile, producer='grid400')
    allocation['QOS'] = 'inferno'
    with pytest.raises(ValueError):
        worker.admit_allocation(allocation, execution_profile=profile, producer='grid400',
                                campaign_scope='eligible-bank')


def test_bank_source_inventory_covers_transitive_guards_without_widening_old_mode():
    worker = load('phoenix_worker_cb63305')
    names = worker.source_names('grid400', remaining_case=True, campaign_scope='eligible-bank')
    required = worker.helper('bank_campaign').ADDITIONAL_SOURCES
    assert set(required) <= set(names) and len(names) == len(set(names))
    assert {'bank_transport.py', 'bank_append.py', 'audited_bank.py', 'bank_runtime_gate.py',
            'dataset_guard.py', 'single_record_cb63305.py'} <= set(names)
    with pytest.raises(ValueError):
        worker.source_names('grid400', remaining_case=True)
    with pytest.raises(ValueError):
        worker.source_names('ed561666', campaign_scope='eligible-bank')


@pytest.mark.parametrize('remaining', [False, True])
def test_unsealed_pool_cannot_reach_candidate_file_reads_or_worker_discovery(monkeypatch, remaining):
    worker, campaign = load('phoenix_worker_cb63305'), load('bank_campaign')
    monkeypatch.setattr(campaign, 'ALLOCATION_SHA256', None)
    reads = []
    monkeypatch.setattr(campaign, '_read', lambda *args, **kwargs: reads.append('read'))
    original = worker.helper
    monkeypatch.setattr(worker, 'helper', lambda name: campaign if name == 'bank_campaign' else original(name))
    args = NS(producer='grid400', campaign_scope='eligible-bank', remaining_case=remaining,
        inputs=campaign.INPUTS, qualification=campaign.QUALIFICATION, descriptors=campaign.DESCRIPTORS,
        pilot=campaign.CASE_ROOT/'monomer-0001.json')
    with pytest.raises(ValueError, match='has not been sealed'):
        worker.prepare_draft(args, campaign.BUILD_IDENTITY)
    assert reads == []


def test_worker_uses_the_closed_bank_reconstruction_seam(monkeypatch):
    worker = load('phoenix_worker_cb63305')
    args = NS(producer='grid400', campaign_scope='eligible-bank', remaining_case=False)
    calls, marker = [], {'unit': 'not a runtime case'}
    original = worker.helper
    monkeypatch.setattr(worker, 'helper', lambda name: NS(
        verify_prepared_from_paths=lambda a, manifest: calls.append((a, manifest)) or marker)
        if name == 'bank_campaign' else original(name))
    assert worker.prepare_draft(args, Path('/unit/build-identity.json')) is marker
    assert calls == [(args, Path('/unit/build-identity.json'))]


def test_unsealed_worker_entrypoint_refuses_before_allocation_git_or_discovery(monkeypatch):
    worker, campaign = load('phoenix_worker_cb63305'), load('bank_campaign')
    monkeypatch.setattr(campaign, 'ALLOCATION_SHA256', None)
    calls = []
    def helper(name):
        calls.append(name)
        if name != 'bank_campaign':
            raise AssertionError('unsealed worker accessed '+name)
        return campaign
    monkeypatch.setattr(worker, 'helper', helper)
    monkeypatch.setattr('sys.argv', ['worker', '--producer', 'grid400',
        '--campaign-scope', 'eligible-bank', '--commit', 'a'*40,
        *[flag for name in ('build', 'qualification', 'inputs', 'descriptors', 'pilot', 'run')
          for flag in ('--'+name, '/unit/'+name)]])
    with pytest.raises(ValueError, match='has not been sealed'):
        worker.main()
    assert calls == ['bank_campaign']


def test_shared_shell_paths_keep_science_build_but_use_new_full_pool():
    path = Path(__file__).with_name('producer_paths.sh')
    result = subprocess.run(['bash', '-c', 'source "$1"; ccasp_select_paths grid400-1aaa9e0a30-bank; '
        'printf "%s\\n" "$CCASP_BUILD" "$CCASP_QUALIFICATION" "$CCASP_INPUTS" "$CCASP_DESCRIPTORS"',
        'unit', str(path)], capture_output=True, text=True, check=True)
    rows = result.stdout.splitlines()
    assert rows[0].endswith('grid400-1aaa9e0a30-build001')
    assert rows[1].endswith('grid400-1aaa9e0a30-installed001/payload/installed-validation.json')
    assert rows[2].endswith('grid400-1aaa9e0a30-full-preparation001')
    assert rows[3].endswith('grid400-1aaa9e0a30-full-descriptors001/payload')


@pytest.mark.parametrize('append', [False, True])
def test_worker_wrapper_forwards_explicit_bank_mode_and_checkpoint_flags(append):
    path = Path(__file__).with_name('phoenix_worker_cb63305.sbatch')
    # Exercise only argument selection; no source/env/discovery/srun is executed.
    prefix = path.read_text().split('ROOT=/storage/project/r-cs207-0/awallace43/runs', 1)[0]
    case = '/storage/project/r-cs207-0/awallace43/runs/ccasp-native-s66-grid400-1aaa9e0a30-bank001/cases/monomer-0002.json'
    args = ['/unit/source', 'a'*40, case, '/unit/run',
            '--grid400-bank-append' if append else '--grid400-bank-fresh']
    if append:
        args += ['/unit/bank-context.json', 'b'*64, '/unit/bank-checkpoint.json', 'c'*64]
    result = subprocess.run(['bash', '-c', prefix+'\nprintf "%s\\n" "${EXTRA[@]}"',
                             'unit', *args], capture_output=True, text=True, check=True)
    flags = result.stdout.splitlines()
    assert flags[:4] == ['--producer', 'grid400', '--campaign-scope', 'eligible-bank']
    assert ('--remaining-case' in flags) is append
    if append:
        assert flags[flags.index('--bank-context')+1] == '/unit/bank-context.json'
        assert flags[flags.index('--bank-checkpoint-sha256')+1] == 'c'*64
