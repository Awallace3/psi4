"""Owned single-record manager lifecycle; no dataset creation or retries.

The caller must bind the new dataset/record, frozen request3 and installed
worker preflight before entering this module. Result delivery is not audit.
"""
import importlib.util
from pathlib import Path
import signal


def helper(name):
    if name not in ('single_claim', 'phoenix_pilot'):
        raise ValueError('closed single-record manager helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_expanded_session_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def validate_session_config(config, compute_tag):
    executors = config.executors
    if set(executors) != {'pilot'}:
        raise ValueError('one dedicated local pilot executor required')
    executor = executors['pilot']
    if (executor.type != 'local' or executor.max_workers != 1
            or executor.cores_per_worker != 4 or executor.memory_per_worker != 20.
            or executor.compute_tags != [compute_tag]):
        raise ValueError('exact single-worker four-core20GiB tag-bound pilot required')


def run_session(config, *, record_id, compute_tag, runtime, payload, save):
    """Claim once, capture compressed local output, and stop this owned manager."""
    validate_session_config(config, compute_tag)
    if type(record_id) is not int or record_id <= 0 or record_id in (5234314, 5234315):
        raise ValueError('fresh expanded-pilot record required')
    from qcfractalcompute.compute_manager import ComputeManager
    mixin = helper('single_claim').SingleClaimMixin
    claim = helper('phoenix_pilot').validate_claim

    class PilotManager(mixin, ComputeManager):
        local_native = None

        def preprocess_new_tasks(self, tasks):
            task = tasks[0]
            values = dict(record_id=task.record_id, compute_tag=task.compute_tag,
                          function=task.function, required_programs=task.required_programs,
                          function_kwargs=task.function_kwargs)
            claim(values, runtime, record_id=record_id, compute_tag=compute_tag)
            save('claimed-input.json', values)
            save('claim.json', dict(task_id=task.id, record_id=task.record_id, manager=self.name))

        def postprocess_results(self, results):
            for task_id, result in results.items():
                with (payload/('local-result-'+str(task_id)+'.zstd')).open('xb') as stream:
                    stream.write(result.result_compressed)
                self.local_native = result.result.get('extras', {}).get('psi4:native_monomer')

    manager = PilotManager(config, target_record_id=record_id, compute_tag=compute_tag,
                           acquisition_seconds=300)
    save('manager-started.json', dict(name=manager.name, tag=compute_tag,
        executor='local', max_workers=1, cores=4, memory_gib=20))
    interrupted = False

    def on_term(signum, frame):
        nonlocal interrupted
        interrupted = True
        manager.acquisition_closed = True
        manager.stop()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, on_term)
    try:
        manager.start()
    finally:
        try:
            # Only this freshly created manager is deactivated. No other
            # manager, job, dataset or record is cancelled/reset/retagged.
            if manager.dflow_kernel is not None:
                try:
                    manager.client.deactivate(
                        active_tasks=manager.statistics.active_tasks,
                        active_cores=manager.statistics.active_cores,
                        active_memory=manager.statistics.active_memory,
                        total_cpu_hours=manager.statistics.total_cpu_hours)
                finally:
                    manager.dflow_kernel.cleanup()
            save('manager-delivery.json', dict(
                manager=manager.name, record_id=record_id, task_id=manager.target_task_id,
                outcome=manager.delivery_outcome, interrupted=interrupted,
                claim_error_type=manager.claim_error_type,
                submission_error_type=manager.submission_error_type,
                local_statistics=manager.statistics.dict()))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if interrupted or manager.delivery_outcome != 'accepted':
        raise RuntimeError('singleton result was not normally acknowledged; no retry or acceptance')
    return dict(manager=manager.name, task_id=manager.target_task_id,
                local_native=manager.local_native)
