"""Synthetic CLI-role checks, not measured runtime evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('runtime_audit', Path(__file__).with_name('runtime_audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def execution(suffix):
    env, scratch = '/private/env', '/fresh/scratch'
    argv = [env+'/bin/python', '-I', '-B', '-X',
            'pycache_prefix='+scratch+'/cache', env+'/bin/psi4', *suffix]
    text = 'execve("'+env+'/bin/python", '+json.dumps(argv)+', 0x123 /* 3 vars */) = 0'
    return dict(pid='123', syscall=text)


@pytest.mark.parametrize('flag,role', [('--module', 'discovery_module'), ('--version', 'discovery_version')])
def test_discovery_modes_cannot_be_counted_as_molecular_execution(flag, role):
    assert audit.cli_role(execution([flag]), env='/private/env', scratch='/fresh/scratch') == role


def test_exact_owned_qcschema_call_is_the_molecular_role():
    call = execution(['--scratch', '/fresh/scratch/case', '--nthread', '4',
                      '--memory', '20.0GiB', '--qcschema', 'data.msgpack'])
    assert audit.cli_role(call, env='/private/env', scratch='/fresh/scratch') == 'molecular_qcschema'


@pytest.mark.parametrize('shape', [(8, 20.), (12, 40.), (24, 80.)])
def test_remaining_profiles_require_exact_claimed_cli_shape(shape):
    call = execution(['--scratch', '/fresh/scratch/case', '--nthread', str(shape[0]),
                      '--memory', str(float(shape[1]))+'GiB', '--qcschema', 'data.msgpack'])
    assert audit.cli_role(call, env='/private/env', scratch='/fresh/scratch',
                          worker_shape=shape) == 'molecular_qcschema'
    with pytest.raises(ValueError):
        audit.cli_role(call, env='/private/env', scratch='/fresh/scratch')
    legacy = execution(['--scratch', '/fresh/scratch/case', '--nthread', '4',
                        '--memory', '20.0GiB', '--qcschema', 'data.msgpack'])
    with pytest.raises(ValueError):
        audit.cli_role(legacy, env='/private/env', scratch='/fresh/scratch', worker_shape=shape)


@pytest.mark.parametrize('shape', [(True, 20.), (12, True), [12, 40.], (12, 80.),
                                   (24, float('inf')), (12, float('nan'))])
def test_trace_worker_shape_is_finite_explicit_and_not_a_broad_option_allowlist(shape):
    with pytest.raises(ValueError, match='explicit finite'):
        audit.cli_role(execution(['--version']), env='/private/env', scratch='/fresh/scratch',
                       worker_shape=shape)


@pytest.mark.parametrize('suffix', [
    ['-c', 'read_saved_model()'],
    ['--scratch', '/history', '--nthread', '4', '--memory', '20.0GiB', '--qcschema', 'data.msgpack'],
    ['--scratch', '/fresh/scratch/../../history', '--nthread', '4',
     '--memory', '20.0GiB', '--qcschema', 'data.msgpack'],
    ['--scratch', '/fresh/scratch/case', '--nthread', '4', '--memory', '20.0GiB',
     '--qcschema', 'data.msgpack', '--psidatadir', '/history'],
])
def test_foreign_program_modes_or_input_locations_are_refused(suffix):
    with pytest.raises(ValueError):
        audit.cli_role(execution(suffix), env='/private/env', scratch='/fresh/scratch')


def test_scratch_identity_preserves_archived_water_and_binds_new_methanol():
    water = {'allocation': {'JobId': '14948991'}}
    assert str(audit.scratch_for_outcome(water)).endswith('/ccasp-native-water-14948991')
    methanol = dict(allocation={'JobId': '123'}, append_checkpoint={'monomer_id': 'monomer-0004'},
                    scratch_path='/storage/scratch1/3/awallace43/ccasp-native-monomer-0004-123')
    assert str(audit.scratch_for_outcome(methanol)) == methanol['scratch_path']
    with pytest.raises(ValueError):
        audit.scratch_for_outcome(dict(methanol, scratch_path='/historical/scratch'))
    with pytest.raises(ValueError):
        audit.scratch_for_outcome(dict(methanol, append_checkpoint={'monomer_id': 'monomer-0006'}))
