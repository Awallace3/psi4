"""One-shot acquisition contract, without a server or quantum execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as Obj

import pytest

spec = importlib.util.spec_from_file_location('single_claim', Path(__file__).with_name('single_claim.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeManager:
    def __init__(self):
        self.executor_programs = {'pilot': {'psi4': ['1.12']}}
        self.manager_config = Obj(executors={'pilot': Obj(compute_tags=['owned-tag'], max_workers=1)})
        self.claim_calls, self.submitted, self.updates = [], [], []
        self.tasks = [Obj(id=9, record_id=42)]
        self.client = Obj(claim=self.claim)
        self.finished = []
        self.return_meta = Obj(accepted_ids=[], rejected_info=[])
        self.stopped = False
        self.inside_update = False

    def claim(self, programs, tags, limit):
        self.claim_calls.append((programs, tags, limit))
        return self.tasks

    def _update(self, new_tasks):
        self.inside_update = True
        self.updates.append(new_tasks)
        if self.finished:
            self._return_finished(self.finished)
        self.inside_update = False

    def _submit_tasks(self, executor, tasks):
        self.submitted.append((executor, tasks))

    def _return_finished(self, results):
        return self.return_meta

    def preprocess_new_tasks(self, tasks):
        pass

    def stop(self):
        assert not self.inside_update, 'must finish processing delivery before shutdown'
        self.stopped = True


class Manager(module.SingleClaimMixin, FakeManager):
    pass


def manager():
    return Manager(target_record_id=42, compute_tag='owned-tag', acquisition_seconds=300)


def test_claim_is_one_record_once_and_failed_delivery_never_reacquires():
    m = manager()
    m._update(new_tasks=True)
    assert len(m.claim_calls) == 1 and m.claim_calls[0][1:] == (['owned-tag'], 1)
    assert m.target_task_id == 9 and len(m.submitted) == 1
    m._update(new_tasks=True)
    assert len(m.claim_calls) == 1 and not m.stopped
    # Server accepts both successful and failed results. Neither can reacquire
    # an automatically reset record on this same update.
    m.finished = {9: Obj(success=False)}
    m.return_meta.accepted_ids = [9]
    m._update(new_tasks=True)
    assert m.stopped and m.delivery_outcome == 'accepted'
    assert len(m.claim_calls) == 1
    assert m.updates == [False, False, False]


def test_empty_claim_or_local_completion_is_not_acknowledged_delivery():
    m = manager()
    m.tasks = []
    m._update(new_tasks=True)
    assert m.target_task_id is None and not m.stopped
    m.tasks = [Obj(id=9, record_id=42)]
    m._update(new_tasks=True)
    m.finished = {9: Obj(success=True)}
    m._update(new_tasks=True)
    assert not m.stopped and m.delivery_outcome is None
    # A deferred retry is observed through the same return hook.
    m.return_meta.accepted_ids = [9]
    m._update(new_tasks=True)
    assert m.stopped and m.delivery_outcome == 'accepted'
    assert len(m.claim_calls) == 2  # Empty attempt followed by the sole claim.


def test_shutdown_update_cannot_acquire_and_rejection_is_not_success():
    m = manager()
    m._update(new_tasks=False)
    assert not m.claim_calls
    m._update(new_tasks=True)
    m.finished = {9: Obj(success=True)}
    m.return_meta.rejected_info = [(9, 'not accepted')]
    m._update(new_tasks=True)
    assert m.stopped and m.delivery_outcome == 'rejected'
    assert len(m.claim_calls) == 1


@pytest.mark.parametrize('tasks', [[Obj(id=9, record_id=99)],
                                  [Obj(id=9, record_id=42), Obj(id=10, record_id=42)]])
def test_unexpected_claim_never_executes(tasks):
    m = manager()
    m.tasks = tasks
    m._update(new_tasks=True)
    assert m.stopped and not m.submitted and m.acquisition_closed
    assert m.delivery_outcome == 'unexpected_claim'


def test_acquisition_timeout_does_not_modify_or_reset_records(monkeypatch):
    m = manager()
    monkeypatch.setattr(module.time, 'monotonic', lambda: m.acquisition_deadline+1)
    m._update(new_tasks=True)
    assert m.stopped and not m.claim_calls and m.delivery_outcome == 'acquisition_timeout'


def test_ambiguous_claim_connection_failure_stops_without_retry_or_execution():
    m = manager()
    calls = []

    def timed_out(*args):
        calls.append(args)
        raise TimeoutError('Server may already have assigned the record')

    m.client.claim = timed_out
    m._update(new_tasks=True)
    assert m.stopped and m.acquisition_closed and not m.submitted
    assert m.delivery_outcome == 'claim_error' and m.claim_error_type == 'TimeoutError'
    # Normal manager shutdown calls another acquisition-disabled update.
    m._update(new_tasks=False)
    m._update(new_tasks=True)
    assert len(calls) == 1


@pytest.mark.parametrize('tag', ['*', 'native,*', '', 'Uppercase'])
def test_wildcard_or_ambiguous_tag_refused(tag):
    with pytest.raises(ValueError, match='exact record/tag'):
        Manager(target_record_id=42, compute_tag=tag, acquisition_seconds=300)
