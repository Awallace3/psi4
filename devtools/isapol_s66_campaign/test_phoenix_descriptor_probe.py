"""Descriptor supervisor source and exact scientific-policy checks, no SCF."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'supervisor_descriptor_probe', Path(__file__).with_name('phoenix_descriptor_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_help_is_dependency_free():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', probe.__file__, '--help'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--memory-bytes' in result.stdout and '--pilots' in result.stdout


def test_run_cannot_be_inside_or_contain_sealed_build():
    assert probe.disjoint_paths(Path('/runs/build'), Path('/runs/probe'))
    assert not probe.disjoint_paths(Path('/runs/build'), Path('/runs/build/probe'))
    assert not probe.disjoint_paths(Path('/runs/probe/build'), Path('/runs/probe'))
    assert not probe.disjoint_paths(Path('/runs/build'), Path('/runs/build'))


def test_native_policy_is_bound_to_exact_git_blobs_not_stripped_text():
    root = Path(__file__).resolve().parents[2]
    planner = probe.helper('resource_plan')
    assert probe.policy_binding(root, dict(commit=planner.SCIENTIFIC_COMMIT), planner) == planner.FILES
    with pytest.raises(RuntimeError):
        probe.policy_binding(root, dict(commit='0'*40), planner)


def test_descriptor_batch_has_explicit_free_route_and_bounded_step_teardown():
    path = Path(__file__).with_name('phoenix_descriptor_probe.sbatch')
    text = path.read_text()
    assert '#SBATCH --qos=embers' in text
    assert '#SBATCH --account=gts-cs207-chemx' in text
    assert '#SBATCH --time=00:30:00' in text
    assert '#SBATCH --signal=TERM@180' in text and '--signal=B:' not in text
    assert '/opt/slurm/current/bin/srun' in text and 'term_wait.sh' in text
    assert 'ccasp_wait_child "$child" 20' in text
    result = subprocess.run(['bash', '-n', str(path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_allocation_verifier_requires_source_build_input_and_memory_binding():
    import copy
    planner = probe.helper('resource_plan')
    identity = dict(implementation={'sha256': {'core': 'core', 'isapol_python': 'python'}},
                    core='/private/core.so', package='/private/psi4/__init__.py')
    report = dict(copy.deepcopy(identity), runtime_input_sha256='input',
                  source_policy_sha256=planner.FILES, descriptor_probe_sha256='probe',
                  planner_sha256='planner', coordinate_change_bohr=0,
                  intended_memory_bytes=20, status='PASS_DESCRIPTOR_PLAN_ONLY',
                  plan=dict(policy_scientific_commit=planner.SCIENTIFIC_COMMIT,
                            resources={'max_bytes': 12}, refusals=[],
                            status='WITHIN_PREDICTED_LEDGER_LIMITS'))
    keywords = dict(policy=planner.FILES, runtime_sha256='input',
                    probe_sha256='probe', planner_sha256='planner', intended_memory_bytes=20)
    probe.verify_child(report, identity, **keywords)
    for bad in (dict(policy={}), dict(policy={'not': 'pinned'}),
                dict(runtime_sha256='different'), dict(probe_sha256='different'),
                dict(planner_sha256='different'), dict(intended_memory_bytes=21)):
        with pytest.raises(RuntimeError):
            probe.verify_child(report, identity, **(keywords | bad))
    bad = copy.deepcopy(report)
    bad['plan']['resources']['max_bytes'] = 21
    with pytest.raises(RuntimeError):
        probe.verify_child(bad, identity, **keywords)
    bad = copy.deepcopy(report)
    bad['plan']['refusals'] = ['cumulative_work']
    with pytest.raises(RuntimeError):
        probe.verify_child(bad, identity, **keywords)
