"""Finite eligible-subset execution shapes; never descriptor admission.

This is a distinct campaign from the historical grid400 singleton. A shape
matching here cannot authorize a refused row, a dataset write, or execution.
"""
import json

PRODUCER = 'grid400'
CAMPAIGN = 'grid400-eligible-bank-v1'
PROTECTED_DATASETS = (353, 354, 355, 356)
PROTECTED_RECORDS = tuple(range(5234314, 5234328))
GIB = 1024**3
_SHAPES = {
    's20': (12, 20, 32, 8, '02:00:00'),
    's40': (32, 40, 64, 12, '04:00:00'),
    's60': (48, 60, 96, 16, '04:00:00'),
    's80': (64, 80, 144, 24, '08:00:00'),
    's120': (96, 120, 186, 24, '08:00:00'),
}


def execution_profile(name):
    if type(name) is not str or name not in _SHAPES:
        raise ValueError('closed eligible grid400 execution shape required')
    _, science, job, cores, duration = _SHAPES[name]
    return dict(name='grid400-bank-'+name, cores=cores,
        science_memory_GiB=science, job_memory_GiB=job, time_limit=duration,
        scratch_bytes=256*GIB, scratch_files=131072, account='gts-cs207-chemx',
        partition='cpu-small', qos='embers', concurrency=1)


def validate_execution_profile(row):
    """Select from exact raw declarations, not forecast dimensions or promotion."""
    try:
        request = row['runtime_input']['specification']['keywords']['function_kwargs']['native_monomer']
        resources = request['resources']
        memory = row['intended_memory_bytes']
        if (request.get('schema') != 'native-monomer-request/3'
                or type(memory) is not int or type(resources) is not dict
                or set(resources) != {'max_bytes', 'max_work', 'max_io_bytes', 'capacity'}
                or any(type(resources[k]) is not int for k in ('max_bytes', 'max_work', 'max_io_bytes'))
                or resources['capacity'] not in ('large', 'extended')
                or not 0 < resources['max_work'] <= (
                    100 if resources['capacity'] == 'large' else 128)*10**12
                or not 0 < resources['max_io_bytes'] <= 64*GIB
                or type(request.get('max_replay_work')) is not int
                or not 0 < request['max_replay_work'] <= 2*10**12
                or type(request.get('pfit_maximum_work_bytes')) is not int
                or not 0 < request['pfit_maximum_work_bytes'] <= resources['max_bytes']):
            raise ValueError
        for name, (numeric, science, _, _, _) in _SHAPES.items():
            if resources['max_bytes'] == numeric*GIB and memory == science*GIB:
                return execution_profile(name)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('finite declared grid400 bank execution contract required') from exc
    raise ValueError('finite declared grid400 bank execution contract required')


def validate_profile(profile):
    if type(profile) is not dict:
        raise ValueError('exact eligible grid400 execution profile required')
    for name in _SHAPES:
        expected = execution_profile(name)
        if json.dumps(profile, sort_keys=True, allow_nan=False) == json.dumps(
                expected, sort_keys=True, allow_nan=False):
            return expected
    raise ValueError('exact eligible grid400 execution profile required')
