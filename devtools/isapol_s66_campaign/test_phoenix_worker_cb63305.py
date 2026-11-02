"""Pure expanded-worker launch boundaries; never scheduler/discovery/science."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location(
    'expanded_worker_fixture', Path(__file__).with_name('phoenix_worker_cb63305.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def allocation():
    return dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                NumCPUs='4', NumNodes='1', MinMemoryNode='24G', TimeLimit='00:45:00')


def test_exact_effective_route_and_reviewed_versions():
    worker.admit_allocation(allocation())
    worker.validate_versions('v0.34.1+3.gd56a0598', '0.63')
    for key, value in (('Account', 'foreign'), ('QOS', 'inferno'),
                       ('Partition', 'cpu-large'), ('NumCPUs', '16'),
                       ('NumNodes', '2'), ('MinMemoryNode', '20G'), ('TimeLimit', '08:00:00')):
        with pytest.raises(ValueError, match='exact effective'):
            worker.admit_allocation(allocation() | {key: value})
    for versions in [('0.34', '0.63'), ('v0.34.1+3.gd56a0598', '0.64')]:
        with pytest.raises(ValueError, match='reviewed'):
            worker.validate_versions(*versions)


@pytest.mark.parametrize('name', ['s20', 's40', 's80'])
def test_remaining_case_preflight_requires_exact_finite_effective_allocation(name):
    cases = worker.helper('case_cb63305')
    profile = dict(name=name, **cases.PROFILES[name], account='gts-cs207-chemx',
                   partition='cpu-small', qos='embers', concurrency=1)
    live = allocation() | dict(NumCPUs=str(profile['cores']),
                              MinMemoryNode=str(profile['job_memory_GiB'])+'G')
    worker.admit_allocation(live, execution_profile=profile)
    for key, value in [('QOS', 'inferno'), ('NumCPUs', '4'), ('MinMemoryNode', '24G'),
                       ('TimeLimit', profile['time_limit'])]:
        with pytest.raises(ValueError):
            worker.admit_allocation(live | {key: value}, execution_profile=profile)
    changed = copy.deepcopy(profile)
    changed['science_memory_GiB'] += 1
    with pytest.raises(ValueError, match='profile changed'):
        worker.admit_allocation(live, execution_profile=changed)


def test_remaining_case_is_fully_reconstructed_not_first_pilot_or_labels(monkeypatch, tmp_path):
    seen = []
    args = NS(remaining_case=True, pilot=tmp_path/'case.json', inputs=tmp_path/'inputs',
              qualification=tmp_path/'qualification', descriptors=tmp_path/'descriptors')
    manifest = tmp_path/'manifest'
    def verify(*paths):
        seen.append(paths)
        return {'SYNTHETIC': 'fully reconstructed case'}
    def helper(name):
        assert name == 'case_cb63305'
        return NS(verify_prepared_from_paths=verify)
    monkeypatch.setattr(worker, 'helper', helper)
    assert worker.prepare_draft(args, manifest) == {'SYNTHETIC': 'fully reconstructed case'}
    assert seen == [(args.pilot, args.inputs, manifest, args.qualification, args.descriptors)]


def test_help_is_safe_without_psi4_or_manager_packages():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', worker.__file__, '--help'],
                            text=True, capture_output=True)
    assert result.returncode == 0 and '--descriptors' in result.stdout


def test_actual_discovery_call_receives_fresh_cache_prefix_not_parent_flags(monkeypatch, tmp_path):
    # Replace rather than clear the process's real environment in this test.
    monkeypatch.setattr(worker.os, 'environ', {'AMBIENT_TOKEN': 'must-not-pass'})
    cache = str(tmp_path/'fresh-cache')
    environment = dict(PYTHONPYCACHEPREFIX=cache, CCASP_NATIVE_CACHE=cache,
                       PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    seen = []
    def discovery(value):
        assert value is None
        seen.append(dict(worker.os.environ))
        return {'qcengine.compute': {'programs': {'psi4': 'synthetic-only'}}}
    assert worker.discover_manager_programs(environment, discovery) == {'psi4': 'synthetic-only'}
    assert seen == [environment] and 'AMBIENT_TOKEN' not in seen[0]
    for invalid in ({k: v for k, v in environment.items() if k != 'PYTHONPYCACHEPREFIX'},
                    environment | {'PYTHONPYCACHEPREFIX': str(tmp_path/'ambient-cache')}):
        with pytest.raises(ValueError, match='fresh explicit discovery'):
            worker.discover_manager_programs(invalid, discovery)
    assert len(seen) == 1


def test_closed_source_set_includes_shared_cli_and_term_handling():
    hashes = worker.source_hashes()
    assert set(hashes) == set(worker.SOURCES)
    assert all(len(v) == 64 for v in hashes.values())
    assert {'native_psi4.sh', 'term_wait.sh', 'dataset_fresh_cb63305.py',
            'pilot_cb63305.py', 'descriptor_probe_cb63305.py'} <= set(hashes)
    with pytest.raises(ValueError, match='closed'):
        worker.helper('arbitrary')


def test_descriptor_discovery_preflight_has_no_calculation_or_server_mutation():
    tree = ast.parse(Path(worker.__file__).read_text())
    calls = {ast.unparse(n.func).split('.')[-1] for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls.intersection({
        'energy', 'properties', 'oeprop', 'Wavefunction', 'NativeRestrictedState',
        'ComputeManager', 'register_once', 'submit_once', 'add_dataset', 'submit'})
    batch = Path(worker.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in batch and '#SBATCH --signal=TERM@180' in batch
    assert 'setsid /opt/slurm/current/bin/srun ' in batch
    assert 'ccasp_wait_child "$child" 20' in batch
    assert batch.index('ccasp_wait_child') < batch.index('EXPANDED_WORKER_PREFLIGHT_PAYLOAD_COMPLETE')
