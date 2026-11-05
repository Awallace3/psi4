"""Synthetic ordinary bank audit routing; no cluster, server or SCI execution."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def load(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_bank_audit_routing_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(append):
    old = load('test_runtime_audit_cb63305')
    values = old.fixture()
    audit = load('runtime_audit_cb63305')
    pins = audit.helper('bank_campaign').pins()
    tag = pins['namespace']+'monomer-0001-123'
    values[0].update(producer_profile='grid400', campaign_scope='eligible-bank',
        dataset_mode='append' if append else 'fresh', monomer_id='monomer-0001',
        dataset_name=pins['dataset_name'], specification_name=pins['specification_name'],
        compute_tag=tag, scratch_path=str(audit.SCRATCH/tag))
    values[2]['compute_tag'] = tag
    return audit, values


@pytest.mark.parametrize('append', [False, True])
def test_explicit_bank_delivery_and_scratch_without_reusing_old_grid400_mode(append):
    audit, values = fixture(append)
    audit.validate_delivery(*values, producer='grid400', owned354=append, campaign_scope='eligible-bank')
    assert audit.scratch_for_outcome(values[0], producer='grid400', expected_monomer='monomer-0001',
        remaining_case=append, campaign_scope='eligible-bank').name == values[0]['compute_tag']
    with pytest.raises(ValueError):
        audit.validate_delivery(*values, producer='grid400', owned354=append)
    with pytest.raises(ValueError):
        audit.scratch_for_outcome(values[0], producer='grid400', expected_monomer='monomer-0001',
                                 remaining_case=append)


@pytest.mark.parametrize('field,value', [
    ('dataset_id', 353), ('dataset_id', 354), ('dataset_id', 355), ('dataset_id', 356),
    ('record_id', 5234314), ('record_id', 5234327), ('dataset_mode', 'fresh'),
    ('campaign_scope', 'default'), ('dataset_name', 'old cohort'),
    ('specification_name', 'old spec')])
def test_bank_audit_rejects_protected_ids_or_changed_mode_and_names(field, value):
    audit, values = fixture(True)
    values = copy.deepcopy(values)
    for item in values:
        if field in item:
            item[field] = value
    with pytest.raises(ValueError):
        audit.validate_delivery(*values, producer='grid400', owned354=True,
                                campaign_scope='eligible-bank')


def test_unsealed_audit_main_refuses_before_artifact_allocation_or_git_reads(monkeypatch):
    audit, campaign = load('runtime_audit_cb63305'), load('bank_campaign')
    monkeypatch.setattr(campaign, 'ALLOCATION_SHA256', None)
    original, calls = audit.helper, []
    def helper(name):
        calls.append(name)
        if name == 'bank_campaign':
            return campaign
        if name == 'phoenix_worker_cb63305':
            return original(name)
        raise AssertionError('unsealed auditor accessed '+name)
    monkeypatch.setattr(audit, 'helper', helper)
    monkeypatch.setattr('sys.argv', ['audit', '--producer', 'grid400',
        '--campaign-scope', 'eligible-bank', '--advance-bank-checkpoint', '--commit', 'a'*40,
        *[flag for name in ('build', 'case', 'worker-preflight', 'qualification',
                           'inputs', 'descriptors', 'pilot', 'run')
          for flag in ('--'+name, '/unit/'+name)]])
    with pytest.raises(ValueError, match='has not been sealed'):
        audit.main()
    assert calls == ['phoenix_worker_cb63305', 'bank_campaign']


@pytest.mark.parametrize('append', [False, True])
def test_audit_wrapper_explicit_bank_paths_and_publication_flag(append):
    path = Path(__file__).with_name('runtime_audit_cb63305.sbatch')
    prefix = path.read_text().split('MANAGER=/storage/project/', 1)[0]
    case = ('/storage/project/r-cs207-0/awallace43/runs/'
            'ccasp-native-s66-grid400-1aaa9e0a30-bank001/cases/monomer-0002.json')
    args = ['/unit/source', 'a'*40, case, '/unit/preflight', '/unit/case', '/unit/audit',
            '--grid400-bank-append' if append else '--grid400-bank-fresh']
    if append:
        args += ['/unit/context', 'b'*64, '/unit/checkpoint', 'c'*64]
    result = subprocess.run(['bash', '-c', prefix+'\nprintf "%s\\n" "${EXTRA[@]}" "$PRODUCER"',
                             'unit', *args], capture_output=True, text=True, check=True)
    flags = result.stdout.splitlines()
    assert flags[:4] == ['--producer', 'grid400', '--campaign-scope', 'eligible-bank']
    assert '--advance-bank-checkpoint' in flags
    assert ('--remaining-case' in flags) is append
    assert flags[-1] == 'grid400-1aaa9e0a30-bank'


def test_failed_report_cannot_reach_source_reads_or_authenticated_client(monkeypatch):
    audit = load('bank_audit')
    monkeypatch.setattr(audit, 'helper', lambda name: pytest.fail('unexpected helper '+name))
    with pytest.raises(ValueError, match='passing ordinary'):
        audit.advance_audit_checkpoint(args=None, raw_audit=json.dumps({'status': 'FAIL'}).encode(),
            data=None, draft=None, append_inputs=None, output=None, source=None)


def test_audit_artifacts_use_bounded_regular_descriptor_reads(monkeypatch):
    audit = load('bank_audit')
    store = audit.helper('bank_store')
    calls = []
    monkeypatch.setattr(store, '_read_regular',
        lambda path, **kwargs: calls.append((path, kwargs)) or b'unit')
    assert audit._raw('/unit/file', 32) == b'unit'
    assert calls == [(Path('/unit/file'), {'maximum': 32})]
