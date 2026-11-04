"""Measured cohort pin/early-refusal guards; no native or QCA execution."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).with_name('grid400_cohort.py')
spec = importlib.util.spec_from_file_location('_grid400_cohort_test', path)
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


def test_measured_singleton_pins_protect_all_old_datasets_and_records():
    pins = cohort.pins()
    assert pins['producer'] == 'grid400' and pins['monomer'] == 'monomer-0008'
    assert pins['descriptor_ops'] == 'a25c3c51509b24e5a5ed4cef8eebf5e4fcbaa347'
    assert pins['descriptor_job'] == '15081893'
    assert pins['protected_datasets'] == (353, 354, 355)
    assert pins['protected_records'] == tuple(range(5234314, 5234327))
    assert pins['allocation_sha256'] == (
        '4e5eb4fd41ec151911da78bde129685fff05ae5707d5a462c622b16a4d385bce')
    assert 'dataset_id' not in pins and 'owned_checkpoint' not in pins


def test_descriptor_source_proof_uses_the_measured_git_snapshot_not_current_files():
    pins = cohort.descriptor_operation_pins()
    assert set(pins) == set(cohort.DESCRIPTOR_SOURCES)
    assert pins['qualified_grid400.py'] == (
        '940667f7779525454d716108ae0d368e7728e601a26f47d02cc26b0850f93219')
    # Shared fresh-pilot refactoring is not part of the measured descriptor closure.
    assert 'pilot_cb63305.py' not in pins


@pytest.mark.parametrize('which', ['bank', 'allocation', 'resources', 'reports', 'mutable_bank'])
def test_unpinned_or_mutable_receipts_refuse_before_any_reconstruction(monkeypatch, which):
    # Deliberately nonreceipt bytes: never synthetic descriptor/model success.
    monkeypatch.setattr(cohort, 'helper', lambda _: (_ for _ in ()).throw(
        AssertionError('unbound bytes must refuse before invoking a helper')))
    bank, allocation, resources, reports = b'not pinned', b'not pinned', b'not pinned', {}
    if which == 'mutable_bank':
        bank = bytearray(bank)
    elif which == 'reports':
        reports = {'../outside.json': b'not a report'}
    with pytest.raises(ValueError, match='exact measured singleton grid400'):
        cohort.verify_admitted_bank({}, {}, resources, bank, allocation, reports)


def test_caller_selected_helpers_are_not_supported():
    with pytest.raises(ValueError, match='closed measured'):
        cohort.helper('../caller-module')
