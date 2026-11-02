"""Supervisor transaction/helper tests only; no SLURM, Psi4 or network."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'phoenix_descriptor_bank_cb63305', Path(__file__).with_name('phoenix_descriptor_bank_cb63305.py'))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def test_help_does_not_require_quantum_packages_or_scheduler():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', supervisor.__file__, '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0 and '--expected-bank-sha256' in result.stdout


def test_complete_json_publication_never_overwrites_prior_artifact(tmp_path):
    path = tmp_path/'complete.json'
    supervisor.write_json_once(path, {'status': 'original'})
    assert json.loads(path.read_text()) == {'status': 'original'}
    assert not path.with_name('.complete.json.partial').exists()
    with pytest.raises(FileExistsError):
        supervisor.write_json_once(path, {'status': 'replacement must not win'})
    assert json.loads(path.read_text()) == {'status': 'original'}


def test_inputs_are_bounded_and_nonempty(tmp_path):
    path = tmp_path/'data'
    for raw in (b'', b'12345'):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match='bounded nonempty'):
            supervisor.read_bounded(path, maximum=4)
    path.write_bytes(b'1234')
    assert supervisor.read_bounded(path, maximum=4) == b'1234'


def test_scientific_child_environment_has_no_inherited_credentials_or_manager_configuration(monkeypatch, tmp_path):
    for key in ('GITHUB_TOKEN', 'QCF_COMPUTE_PASSWORD', 'PYTHONPATH', 'PSIPATH',
                'SSH_AUTH_SOCK', 'PYTEST_ADDOPTS', 'CAMCASP'):
        monkeypatch.setenv(key, 'must-not-be-inherited')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    environment = supervisor.child_environment(tmp_path/'env', tmp_path/'private')
    assert environment['SLURM_JOB_ID'] == '123'
    assert environment['HOME'] == str(tmp_path/'private/home')
    assert environment['TMPDIR'] == environment['SCRATCH'] == environment['PSI_SCRATCH']
    assert not set(environment).intersection({
        'GITHUB_TOKEN', 'QCF_COMPUTE_PASSWORD', 'PYTHONPATH', 'PSIPATH',
        'SSH_AUTH_SOCK', 'PYTEST_ADDOPTS', 'CAMCASP'})


def test_only_fixed_helpers_and_complete_operation_sources_are_available():
    with pytest.raises(ValueError, match='closed'):
        supervisor.helper('/arbitrary/helper')
    hashes = supervisor.operation_hashes()
    assert set(hashes) == set(supervisor.OP_FILES)
    assert {'descriptor_probe_cb63305.py', 'descriptor_probe.py', 'resource_plan_cb63305.py',
            'phoenix_descriptor_bank_cb63305.sbatch'} <= set(hashes)
    assert all(len(v) == 64 for v in hashes.values())


def test_source_policy_binds_exact_committed_scientific_blobs():
    planner = supervisor.helper('resource_plan_cb63305')
    source = Path(__file__).resolve().parents[2]
    assert supervisor.policy_binding(source, {'commit': planner.SCIENTIFIC_COMMIT}, planner) == planner.FILES
    with pytest.raises(ValueError, match='exact cb63305'):
        supervisor.policy_binding(source, {'commit': 'a'*40}, planner)


def test_batch_wrapper_reuses_bounded_term_safe_wait_before_completion():
    text = Path(supervisor.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in text
    assert '#SBATCH --signal=TERM@180' in text
    assert 'source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"' in text
    assert 'setsid /opt/slurm/current/bin/srun --ntasks=1 --cpu-bind=none ' in text
    assert 'ccasp_wait_child "$child" 20' in text
    assert text.index('ccasp_wait_child') < text.index('exit 143') < text.index(
        'DESCRIPTOR_ALLOCATION_PAYLOAD_COMPLETE')
    assert 'term_wait.sh' in supervisor.OP_FILES
