"""Version-pinned operational adapter for a one-record compute0.63 pilot.

The underlying manager normally claims three tasks per worker. This mixin
permanently closes acquisition on its first nonempty claim and drains results
through the ordinary delivery/deferred-delivery machinery. Server acceptance
is delivery evidence, never evidence of successful molecular computation.
Private manager APIs require deployed-version/source checks before use.
"""
import time


class SingleClaimMixin:
    def __init__(self, *args, target_record_id, compute_tag, acquisition_seconds, **kwargs):
        if (type(target_record_id) is not int or target_record_id <= 0
                or type(compute_tag) is not str or not compute_tag
                or '*' in compute_tag or compute_tag != compute_tag.lower()
                or type(acquisition_seconds) is not int or not 0 < acquisition_seconds <= 1800):
            raise ValueError('one exact record/tag and bounded acquisition deadline required')
        self.target_record_id = target_record_id
        self.compute_tag = compute_tag
        self.target_task_id = None
        self.acquisition_closed = False
        self.delivery_outcome = None
        self.claim_error_type = None
        self.acquisition_deadline = time.monotonic()+acquisition_seconds
        super().__init__(*args, **kwargs)
        executors = self.manager_config.executors
        if (len(executors) != 1
                or any(e.compute_tags != [compute_tag] or e.max_workers != 1
                       for e in executors.values())):
            raise ValueError('exactly one dedicated one-worker executor required')
        self.pilot_executor = next(iter(executors))

    def _return_finished(self, results):
        metadata = super()._return_finished(results)
        if self.target_task_id is not None:
            accepted = self.target_task_id in metadata.accepted_ids
            rejected = any(task_id == self.target_task_id for task_id, _ in metadata.rejected_info)
            if accepted and rejected:
                raise RuntimeError('contradictory server delivery acknowledgment')
            if accepted:
                self.delivery_outcome = 'accepted'
            elif rejected:
                self.delivery_outcome = 'rejected'
        return metadata

    def _update(self, new_tasks):
        # Let the deployed manager collect, postprocess, deliver and retry
        # deferred returns, but never enter its three-slot acquisition path.
        super()._update(new_tasks=False)
        if self.delivery_outcome is not None:
            self.stop()
            return
        if self.acquisition_closed or not new_tasks:
            return
        if time.monotonic() >= self.acquisition_deadline:
            self.acquisition_closed = True
            self.delivery_outcome = 'acquisition_timeout'
            self.stop()
            return
        try:
            tasks = self.client.claim(
                self.executor_programs[self.pilot_executor], [self.compute_tag], 1)
        except Exception as exc:
            # A timed-out response can follow a successful server-side claim.
            # Do not retry or escape scheduler.run(), which would bypass the
            # parent's normal deactivation and executor cleanup.
            self.acquisition_closed = True
            self.delivery_outcome = 'claim_error'
            self.claim_error_type = type(exc).__name__
            self.stop()
            return
        if not tasks:
            return
        # Close before validation/submission. Even a failed result accepted and
        # automatically reset by the server must never be claimed here again.
        self.acquisition_closed = True
        if len(tasks) != 1 or tasks[0].record_id != self.target_record_id:
            self.delivery_outcome = 'unexpected_claim'
            self.stop()
            return
        self.target_task_id = tasks[0].id
        self.preprocess_new_tasks(tasks)
        self._submit_tasks(self.pilot_executor, tasks)
