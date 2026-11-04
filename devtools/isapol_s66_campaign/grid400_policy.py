"""The one approved fresh pilot's finite execution policy, not admission.

Installed receipts and measured selected descriptors remain separate gates.
Historical execution profiles and dataset355 are not extended or reinterpreted.
"""
import json

MONOMER = 'monomer-0008'
PRODUCER = 'grid400'
PROTECTED_DATASETS = (353, 354, 355)
PROTECTED_RECORDS = tuple(range(5234314, 5234327))
GIB = 1024**3


def execution_profile():
    """Fresh data for the exact approved <=4h free-embers singleton shape."""
    return dict(name='grid400-s60', cores=16, science_memory_GiB=60, job_memory_GiB=96,
        time_limit='04:00:00', scratch_bytes=256*GIB, scratch_files=131072,
        account='gts-cs207-chemx', partition='cpu-small', qos='embers', concurrency=1)


def validate_execution_profile(row):
    """Bind declared resources to this shape; not a measured RSS certificate."""
    request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
    resources = request['resources']
    expected = dict(max_bytes=48*GIB, max_work=75*10**12,
                    max_io_bytes=16*GIB, capacity='large')
    if (row.get('monomer_id') != MONOMER
            or type(row.get('intended_memory_bytes')) is not int
            or row['intended_memory_bytes'] != 60*GIB
            or type(resources) is not dict or set(resources) != set(expected)
            or resources != expected
            or any(type(resources[key]) is not int for key in (
                'max_bytes', 'max_work', 'max_io_bytes'))
            or type(request.get('max_replay_work')) is not int
            or request['max_replay_work'] != 10**12
            or type(request.get('pfit_maximum_work_bytes')) is not int
            or request['pfit_maximum_work_bytes'] != 2*GIB):
        raise ValueError('exact fresh grid400 monomer0008 finite execution contract required')
    return execution_profile()


def validate_profile(profile):
    """Reject changing the allocation, duration or scratch declaration."""
    if type(profile) is not dict or json.dumps(profile, sort_keys=True, allow_nan=False) != (
            json.dumps(execution_profile(), sort_keys=True, allow_nan=False)):
        raise ValueError('exact approved grid400 execution profile required')
    return execution_profile()
