"""Synthetic command-role fixtures, never evidence of execution or SCI."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def load():
    path = Path(__file__).with_name('bank_runtime_gate.py')
    spec = importlib.util.spec_from_file_location('_bank_runtime_gate_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(cores, memory, *, role='molecular'):
    prefix = ['/unit/env/bin/python', '-I', '-B', '-X',
              'pycache_prefix=/unit/scratch/cache', '/unit/env/bin/psi4']
    tail = (['--scratch', '/unit/scratch/task', '--nthread', str(cores),
             '--memory', str(float(memory))+'GiB', '--qcschema', 'data.msgpack']
            if role == 'molecular' else ['--'+role])
    return dict(syscall='execve("/unit/env/bin/python", '+json.dumps(prefix+tail)+', []) = 0')


@pytest.mark.parametrize('name', ['s20', 's40', 's60', 's80', 's120'])
@pytest.mark.parametrize('role,expected', [('module', 'discovery_module'),
    ('version', 'discovery_version'), ('molecular', 'molecular_qcschema')])
def test_all_bank_shapes_require_the_exact_owned_cli_and_expected_role(name, role, expected):
    gate = load()
    profile = gate.helper('grid400_bank_policy').execution_profile(name)
    observed = command(profile['cores'], profile['science_memory_GiB'], role=role)
    assert gate.cli_role(observed, env='/unit/env', scratch='/unit/scratch',
                         execution_profile=profile) == expected


@pytest.mark.parametrize('change', ['memory', 'cores', 'python', 'cache', 'scratch', 'mode', 'exit'])
def test_largest_profile_rejects_unbound_interpreter_resources_or_modes(change):
    gate = load()
    profile = gate.helper('grid400_bank_policy').execution_profile('s120')
    observed = command(24, 120)
    old, new = dict(memory=('120.0GiB', '80.0GiB'), cores=('"24"', '"16"'),
        python=('/unit/env/bin/python', '/foreign/python'),
        cache=('pycache_prefix=/unit/scratch/cache', 'pycache_prefix=/foreign/cache'),
        scratch=('/unit/scratch/task', '/foreign/scratch'), mode=('--qcschema', '--psiapi'),
        exit=('= 0', '= 1'))[change]
    observed['syscall'] = observed['syscall'].replace(old, new)
    with pytest.raises(ValueError):
        gate.cli_role(observed, env='/unit/env', scratch='/unit/scratch', execution_profile=profile)


def test_frozen_historical_classifier_is_unchanged_and_still_refuses_120():
    gate = load()
    historical = Path(__file__).with_name('runtime_audit.py')
    root = historical.resolve().parents[2]
    expected = subprocess.check_output(['git', 'show',
        '2c2c002de7e5da29ef0998e1c4236674b2a63e6b:devtools/isapol_s66_campaign/runtime_audit.py'], cwd=root)
    assert historical.read_bytes() == expected
    with pytest.raises(ValueError):
        gate.helper('runtime_audit').cli_role(command(24, 120), env='/unit/env',
            scratch='/unit/scratch', worker_shape=(24, 120))
