"""Owned single-record manager lifecycle; no dataset creation or retries.

The caller must bind the new dataset/record, frozen request3 and installed
worker preflight before entering this module. Result delivery is not audit.
"""
import copy
import importlib.util
from pathlib import Path
import re
import signal


def helper(name):
    if name not in ('single_claim', 'phoenix_pilot', 'pilot_cb63305', 'dataset_fresh_cb63305',
                    'case_cb63305', 'owned_dataset_cb63305', 'grid400_policy',
                    'grid400_bank_execution'):
        raise ValueError('closed single-record manager helpers required')
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_expanded_session_'+name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def validate_session_config(config, compute_tag, *, worker_shape=(4, 20.), campaign_scope='default'):
    """Validate a finite worker shape, not molecular memory/scratch admission.

    The historical singleton keeps its exact default. Remaining-bank callers
    must separately bind their case, allocation and worker-preflight receipts.
    """
    if campaign_scope not in ('default', 'eligible-bank', 'eligible-wave'):
        raise ValueError('closed native session scope required')
    shapes = (helper('grid400_bank_execution').worker_shapes() if campaign_scope in ('eligible-bank', 'eligible-wave')
              else ((4, 20.), (8, 20.), (12, 40.), (24, 80.), (16, 60.)))
    if (type(worker_shape) is not tuple or len(worker_shape) != 2
            or type(worker_shape[0]) is not int
            or type(worker_shape[1]) not in (int, float)
            or worker_shape not in shapes):
        raise ValueError('explicit finite native worker shape required')
    cores, memory = worker_shape
    executors = config.executors
    if set(executors) != {'pilot'}:
        raise ValueError('one dedicated local pilot executor required')
    executor = executors['pilot']
    if (executor.type != 'local' or executor.max_workers != 1
            or executor.cores_per_worker != cores or executor.memory_per_worker != memory
            or executor.compute_tags != [compute_tag]):
        raise ValueError('exact single-worker shape and tag-bound pilot required')


def run_session(config, *, record_id, compute_tag, runtime, payload, save, worker_shape=(4, 20.),
                producer='cb63305', case=None, campaign_scope='default',
                wave_binding=None, allocation=None):
    """Claim once, capture compressed local output, and stop this owned manager."""
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400'):
        raise ValueError('closed singleton producer required')
    if (campaign_scope not in ('default', 'eligible-bank', 'eligible-wave')
            or campaign_scope in ('eligible-bank', 'eligible-wave') and producer != 'grid400'):
        raise ValueError('eligible bank session requires explicit grid400 producer and scope')
    if producer != 'grid400' and worker_shape == (16, 60.):
        raise ValueError('grid400 worker shape requires its explicit fresh producer')
    protected = (5234314, 5234315)
    if campaign_scope != 'eligible-wave' and (wave_binding is not None or allocation is not None):
        raise ValueError('wave authority requires explicit wave session')
    if campaign_scope in ('eligible-bank', 'eligible-wave'):
        protected = helper('grid400_bank_execution').validate_binding(case,
            runtime=runtime, compute_tag=compute_tag, worker_shape=worker_shape, record_id=record_id,
            **(dict(campaign_scope=campaign_scope, wave_binding=wave_binding, allocation=allocation)
               if campaign_scope == 'eligible-wave' else {}))
    elif producer == 'grid400':
        pins = helper('pilot_cb63305').producer_pins(producer)
        policy = helper('grid400_policy')
        profile = policy.execution_profile()
        if (case is not None or worker_shape != (profile['cores'], profile['science_memory_GiB'])
                or re.fullmatch(re.escape(pins['namespace']+pins['monomer']+'-')+
                                r'[1-9][0-9]*', compute_tag) is None):
            raise ValueError('exact fresh grid400 singleton shape and tag, never append, required')
        raw_runtime = copy.deepcopy(runtime)
        protocols = raw_runtime['specification'].pop('protocols', None)
        digest = helper('case_cb63305').json_digest
        if (digest(protocols) != digest(helper('phoenix_pilot').execution_specification({})['protocols'])
                or digest(raw_runtime) != pins['runtime_input_sha256']):
            raise ValueError('exact frozen grid400 input and no-correction execution protocols required')
        policy.validate_execution_profile(dict(monomer_id=pins['monomer'], runtime_input=raw_runtime,
            intended_memory_bytes=profile['science_memory_GiB']*1024**3))
        protected = pins['protected_records']
    elif producer == 'ed561666':
        pins = helper('pilot_cb63305').producer_pins(producer)
        protected = pins['protected_records']
        monomer = pins['monomer']
        if case is not None:
            context = helper('owned_dataset_cb63305').context(producer)
            monomer = case.get('monomer_id')
            if (case.get('producer_profile') != producer or case.get('dataset_mode') != 'append'
                    or case.get('dataset_id') != context['dataset_id']
                    or case.get('required_first_pilot_audit_sha256') != context['accepted_sha256']['audit']
                    or type(monomer) is not str or re.fullmatch(r'monomer-[0-9]{4}', monomer) is None
                    or not 1 <= int(monomer[-4:]) <= 131 or monomer == context['first_monomer']):
                raise ValueError('reconstructed owned355 singleton case required')
            cases = helper('case_cb63305')
            profile = cases.validate_execution_profile(case['execution_profile']['name'],
                dict(runtime_input=runtime, intended_memory_bytes=case['intended_memory_bytes']))
            if (cases.json_digest(profile) != cases.json_digest(case['execution_profile'])
                    or worker_shape != (profile['cores'], profile['science_memory_GiB'])):
                raise ValueError('case-bound finite cap32 singleton worker profile required')
            protected += (context['first_record'],)
        elif worker_shape != (12, 40.):
            raise ValueError('exact fresh cap32 singleton shape and namespace required')
        if re.fullmatch(re.escape(pins['namespace']+monomer+'-')+r'[1-9][0-9]*', compute_tag) is None:
            raise ValueError('exact fresh cap32 singleton shape and namespace required')
    elif case is not None:
        raise ValueError('selected singleton case binding is only supported for cap32')
    scope_args = {} if campaign_scope == 'default' else dict(campaign_scope=campaign_scope)
    validate_session_config(config, compute_tag, worker_shape=worker_shape, **scope_args)
    if type(record_id) is not int or record_id <= 0 or record_id in protected:
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
                # FailedOperation legitimately has extras=None. Preserve and
                # deliver that original failure; an ACK is not molecular success.
                extras = result.result.get('extras')
                self.local_native = extras.get('psi4:native_monomer') if isinstance(extras, dict) else None

    if producer == 'grid400':
        with helper('dataset_fresh_cb63305').grid400_manager_client_factory() as client_type:
            manager = PilotManager(config, target_record_id=record_id, compute_tag=compute_tag,
                                   acquisition_seconds=300)
            if type(manager.client) is not client_type:
                raise ValueError('owned grid400 manager did not use the one-shot activation client')
    else:
        manager = PilotManager(config, target_record_id=record_id, compute_tag=compute_tag,
                               acquisition_seconds=300)
    if producer == 'ed561666':
        helper('dataset_fresh_cb63305').guard_manager_client(manager.client)
    save('manager-started.json', dict(name=manager.name, tag=compute_tag,
        executor='local', max_workers=1, cores=worker_shape[0], memory_gib=worker_shape[1]))
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
