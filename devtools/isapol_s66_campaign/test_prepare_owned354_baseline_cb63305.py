"""Read-only fake-server baseline tests, not molecular evidence."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('owned354_baseline_fixture',
    Path(__file__).with_name('prepare_owned354_baseline_cb63305.py'))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.mark.parametrize('change', ['none', 'server', 'concurrent'])
def test_baseline_uses_two_snapshots_and_never_mutates(monkeypatch, change):
    effects = []
    before = dict(entries={'synthetic-entry': {}}, records={'9999': {}}, metadata={'id': 354})
    after = copy.deepcopy(before)
    if change == 'concurrent':
        after['records']['9999']['status'] = 'changed'
    snapshots = iter([before, after])
    def snapshot(client):
        effects.append('snapshot')
        return next(snapshots)
    def accepted(blobs):
        effects.append('verify_closed_receipts')
        assert blobs == {'synthetic': b'not actual receipts'}
        return {'unit_only': True}
    def checkpoint(value, **kwargs):
        effects.append('validate_owned_first_case')
        assert kwargs == dict(authenticated_owner='awallace43',
                              specification={'synthetic': 'spec'}, accepted={'unit_only': True})
        return dict(schema='native-owned354-checkpoint/1', status='READ_ONLY_INITIAL_BASELINE',
                    snapshot=value)
    guard = NS(verify_first_acceptance=accepted, snapshot_dataset=snapshot,
               prepare_initial_checkpoint=checkpoint)
    monkeypatch.setattr(prep, 'helper', lambda name: guard)
    class Client:
        def get_server_information(self):
            return dict(prep.SERVER, version='foreign') if change == 'server' else dict(prep.SERVER)
        def make_request(self, *args, **kwargs):
            raise AssertionError('no mutation or unreviewed endpoint available')
    kwargs = dict(blobs={'synthetic': b'not actual receipts'}, authenticated_owner='awallace43',
                  specification={'synthetic': 'spec'})
    if change == 'server':
        with pytest.raises(ValueError, match='deployment identity'):
            prep.prepare_baseline(Client(), **kwargs)
        assert effects == []
    elif change == 'concurrent':
        with pytest.raises(RuntimeError, match='changed during'):
            prep.prepare_baseline(Client(), **kwargs)
    else:
        result = prep.prepare_baseline(Client(), **kwargs)
        assert result['status'] == 'READ_ONLY_INITIAL_BASELINE'
        assert 'not mutation authorization alone' in result['qualification']
    if change != 'server':
        assert effects == ['verify_closed_receipts', 'snapshot', 'validate_owned_first_case', 'snapshot']


def test_baseline_has_no_entry_specification_or_record_write_helper():
    with pytest.raises(ValueError, match='closed'):
        prep.helper('dataset_append')


def test_cap32_baseline_selection_is_explicit_on_both_reads_and_acceptance(monkeypatch):
    effects = []
    value = dict(entries={'synthetic': {}}, records={'123': {}})
    def accepted(blobs, *, producer):
        effects.append(('acceptance', producer))
        return {'synthetic': True}
    def snapshot(client, *, producer):
        effects.append(('snapshot', producer))
        return copy.deepcopy(value)
    def checkpoint(snapshot, *, producer, **kwargs):
        effects.append(('checkpoint', producer))
        return dict(status='READ_ONLY_INITIAL_BASELINE', dataset_id=355, snapshot=snapshot)
    guard = NS(verify_first_acceptance=accepted, snapshot_dataset=snapshot,
               prepare_initial_checkpoint=checkpoint)
    monkeypatch.setattr(prep, 'helper', lambda name: guard)
    result = prep.prepare_baseline(NS(get_server_information=lambda: dict(prep.SERVER)),
        blobs={}, authenticated_owner='awallace43', specification={}, producer='ed561666')
    assert result['dataset_id'] == 355
    assert effects == [('acceptance', 'ed561666'), ('snapshot', 'ed561666'),
                       ('checkpoint', 'ed561666'), ('snapshot', 'ed561666')]
    with pytest.raises(ValueError, match='closed'):
        prep.prepare_baseline(object(), blobs={}, authenticated_owner='awallace43',
                              specification={}, producer='arbitrary')
