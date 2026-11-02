"""Synthetic expanded-pilot audit guards; no real molecular qualification."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('expanded_audit_fixture', 'runtime_audit_cb63305.py')


def fixture():
    job, tag = '123', 'ccasp-native-cb63305-monomer-0006-123'
    outcome = dict(schema='native-expanded-pilot-outcome/1',
        status='DELIVERED_EXPANDED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION', runtime_unchanged=True,
        allocation={'JobId': job}, monomer_id='monomer-0006', dataset_id=9999, record_id=8888,
        task_id=77, manager='SYNTHETIC_UNIT_MANAGER', compute_tag=tag,
        scratch_path=str(audit.SCRATCH/'ccasp-native-cb63305-monomer-0006-123'),
        find_existing=False, maximum_records=1)
    delivery = dict(record_id=8888, task_id=77, manager=outcome['manager'], outcome='accepted',
        interrupted=False, local_statistics=dict(total_successful_tasks=1, total_failed_tasks=0,
            total_rejected_tasks=0, active_tasks=0))
    created = dict(dataset_id=9999, record_id=8888, n_inserted=1, n_existing=0, compute_tag=tag)
    claimed = dict(record_id=8888, task_id=77, manager=outcome['manager'])
    server = dict(record_id=8888, status='complete', history_count=1,
                  history_managers=[outcome['manager']])
    return outcome, delivery, created, claimed, server


def test_only_new_completed_singleton_delivery_and_exact_scratch_are_consistent():
    values = fixture()
    audit.validate_delivery(*values)
    assert audit.scratch_for_outcome(values[0]).name == 'ccasp-native-cb63305-monomer-0006-123'
    # A consistent delivery is still not a numeric/source/trace audit verdict.
    assert values[0]['status'].endswith('PENDING_QUALIFICATION')


@pytest.mark.parametrize('change', [
    'old_dataset', 'old_record', 'old_status', 'runtime_changed', 'reused_record', 'two_records',
    'wrong_created_dataset', 'wrong_record', 'wrong_task', 'wrong_manager', 'interrupted',
    'failed_task', 'rejected_task', 'active_task', 'two_histories', 'wrong_history', 'waiting'])
def test_inconsistent_or_historical_delivery_cannot_be_promoted(change):
    values = list(copy.deepcopy(fixture()))
    outcome, delivery, created, claimed, server = values
    if change == 'old_dataset':
        outcome['dataset_id'] = 353
    elif change == 'old_record':
        outcome['record_id'] = 5234314
    elif change == 'old_status':
        outcome['status'] = 'DELIVERED_NATIVE_ARTIFACTS_PENDING_QUALIFICATION'
    elif change == 'runtime_changed':
        outcome['runtime_unchanged'] = False
    elif change == 'reused_record':
        created['n_existing'] = 1
    elif change == 'two_records':
        created['n_inserted'] = 2
    elif change == 'wrong_created_dataset':
        created['dataset_id'] = 9998
    elif change == 'wrong_record':
        claimed['record_id'] += 1
    elif change == 'wrong_task':
        delivery['task_id'] += 1
    elif change == 'wrong_manager':
        delivery['manager'] = 'foreign'
    elif change == 'interrupted':
        delivery['interrupted'] = True
    elif change in ('failed_task', 'rejected_task', 'active_task'):
        key = dict(failed_task='total_failed_tasks', rejected_task='total_rejected_tasks',
                   active_task='active_tasks')[change]
        delivery['local_statistics'][key] = 1
    elif change == 'two_histories':
        server['history_count'] = 2
    elif change == 'wrong_history':
        server['history_managers'] = ['foreign']
    elif change == 'waiting':
        server['status'] = 'waiting'
    with pytest.raises(ValueError):
        audit.validate_delivery(*values)


@pytest.mark.parametrize('field,value', [
    ('monomer_id', 'monomer-0004'), ('scratch_path', '/foreign/scratch'), ('compute_tag', '*')])
def test_scratch_identity_is_not_inferred_from_old_append_or_water_defaults(field, value):
    outcome = fixture()[0]
    outcome[field] = value
    with pytest.raises(ValueError):
        audit.scratch_for_outcome(outcome)


def test_request3_uses_the_unchanged_independent_numeric_reduction():
    examples = load('numeric_examples_for_request3', 'test_validate_numeric.py')
    envelope, raw = examples.example()
    # Synthetic 3-site example only, not a methylamine molecular result.
    envelope['request'].update(schema='native-monomer-request/3',
        pfit_maximum_work_bytes=256*1024**2, max_replay_work=10**12)
    result = audit.helper('validate_numeric').validate_numeric(envelope, raw)
    assert result['status'] == 'PASS_NUMERIC_TRANSPORT_ONLY'
    assert result['maximum_cn_absolute_difference'] == 0.
    assert not result['full_C10'] and not result['independent_solver_certified']


def test_help_and_term_safe_wrapper_do_not_run_science():
    result = subprocess.run([sys.executable, '-I', '-S', '-B', audit.__file__, '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0 and '--case' in result.stdout
    text = Path(audit.__file__).with_suffix('.sbatch').read_text()
    assert '#SBATCH --qos=embers' in text and '#SBATCH --signal=TERM@180' in text
    assert 'ccasp_wait_child "$child" 20' in text
    assert text.index('ccasp_wait_child') < text.index('EXPANDED_PILOT_AUDIT_PASS_NOT_ACCURACY')
