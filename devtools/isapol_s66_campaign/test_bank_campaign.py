"""Closed campaign paths and deliberately inactive unsealed-pool guard."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load():
    path = Path(__file__).with_name('bank_campaign.py')
    spec = importlib.util.spec_from_file_location('_bank_campaign_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unsealed_descriptor_pool_refuses_before_reading_inputs_or_git(monkeypatch):
    module = load()
    # This regression remains valid after the real completed pool is sealed:
    # an absent compiled authority can never mean "trust whatever is on disk".
    monkeypatch.setattr(module, 'ALLOCATION_SHA256', None)
    touched = []
    monkeypatch.setattr(module, '_read', lambda *args, **kwargs: touched.append('input'))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: touched.append('git'))
    with pytest.raises(ValueError, match='has not been sealed'):
        module.load_pool(module.INPUTS, module.BUILD_IDENTITY, module.QUALIFICATION, module.DESCRIPTORS)
    assert touched == []


def test_campaign_namespaces_and_protected_history_are_distinct_from_singleton():
    module = load()
    pins = module.pins()
    assert pins['namespace'] == 'ccasp-native-grid400-bank-'
    assert pins['worker_namespace'] == 'ccasp-native-worker-grid400-bank-'
    assert pins['protected_datasets'] == (353, 354, 355, 356)
    assert pins['protected_records'] == tuple(range(5234314, 5234328))
    assert pins['dataset_name'].endswith('-eligible-request3-v1')
    assert module.DESCRIPTOR_JOB == '15153484'
    assert module.INPUTS.name.endswith('-full-preparation001')
    assert module.CASE_ROOT.parent == module.CAMPAIGN_ROOT


@pytest.mark.parametrize('change', ['old_namespace', 'context_name', 'checkpoint_name', 'hash'])
def test_old_or_unbound_checkpoint_paths_refuse_before_reads(monkeypatch, change):
    module = load()
    args = NS(bank_context=module.CAMPAIGN_ROOT/'audit/payload/bank-context.json',
        bank_context_sha256='a'*64,
        bank_checkpoint=module.CAMPAIGN_ROOT/'audit/payload/bank-checkpoint.json',
        bank_checkpoint_sha256='b'*64)
    if change == 'old_namespace':
        args.bank_context = module.RUNS/'ccasp-native-s66-grid400-old/payload/bank-context.json'
    elif change == 'context_name':
        args.bank_context = args.bank_context.with_name('forensic-report.json')
    elif change == 'checkpoint_name':
        args.bank_checkpoint = args.bank_checkpoint.with_name('owned-checkpoint.json')
    else:
        args.bank_context_sha256 = 'not-a-hash'
    touched = []
    monkeypatch.setattr(module, '_read', lambda *args, **kwargs: touched.append('read'))
    with pytest.raises(ValueError, match='ordinary-audit'):
        module.load_checkpoint(args)
    assert touched == []
