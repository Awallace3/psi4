"""Preflight input separation; no Psi4 import or quantum execution."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('worker_probe', Path(__file__).with_name('phoenix_worker_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_runtime_file_contains_no_entry_attributes_or_submission_controls():
    pilot = dict(
        entry=dict(molecule={'symbols': ['O', 'H', 'H'], 'geometry': [0]*9},
                   attributes={'sapt_reference': {'DISP': -2.}, 'private_label': 'metadata-only'},
                   additional_keywords={'function_kwargs': {'native_monomer': {'schema': 'native-monomer-request/2'}}}),
        specification=dict(driver='properties', program='psi4', method='pbe0',
                           basis='aug-cc-pvtz', keywords={'reference': 'rhf'}),
        submission={'find_existing': False})
    original = copy.deepcopy(pilot)
    result = probe.runtime_input(pilot)
    assert pilot == original
    assert set(result) == {'molecule', 'specification'}
    assert result['specification']['model'] == {'method': 'pbe0', 'basis': 'aug-cc-pvtz'}
    assert result['specification']['keywords']['function_kwargs'] == pilot['entry']['additional_keywords']['function_kwargs']
    text = json.dumps(result)
    assert all(token not in text for token in ('sapt_reference', 'DISP', 'metadata-only', 'find_existing'))


def test_probe_cli_help_needs_no_scientific_or_manager_packages():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', probe.__file__, '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--qualification' in result.stdout and '--scientific-probe' in result.stdout


@pytest.mark.parametrize('programs,cli,science,imported', [
    ({}, '1.12', '1.12', False),
    ({'psi4': '1.11'}, '1.12', '1.12', False),
    ({'psi4': '1.12'}, '1.12', '1.11', False),
    ({'psi4': '1.12'}, '1.12', '1.12', True),
])
def test_discovery_cannot_advertise_foreign_version_or_import(programs, cli, science, imported):
    with pytest.raises(RuntimeError):
        probe.validate_worker_discovery(programs, cli, science, scientific_imported=imported)


def test_all_three_version_observations_must_agree_without_parent_scientific_import():
    probe.validate_worker_discovery({'psi4': '1.12'}, '1.12', '1.12', scientific_imported=False)


def test_native_cli_traces_include_resolved_fd_targets_and_child_processes():
    import shlex
    text = Path(__file__).with_name('native_psi4.sh').read_text().replace('\\\n', ' ')
    command = next(line for line in text.splitlines() if line.startswith('exec /usr/bin/strace '))
    tokens = shlex.split(command)
    assert '-yy' in tokens and '-f' in tokens
    assert 'trace=%file,%process' in tokens
