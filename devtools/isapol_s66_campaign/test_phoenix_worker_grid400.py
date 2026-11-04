"""Pure fresh-worker shape/source guards; no actual discovery or native code."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).with_name('phoenix_worker_cb63305.py')
spec = importlib.util.spec_from_file_location('_grid400_worker_test', path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
policy = worker.helper('grid400_policy')


def allocation():
    return dict(Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
                NumCPUs='16', NumNodes='1', MinMemoryNode='96G', TimeLimit='00:45:00')


def test_closed_grid400_worker_requires_the_approved_profile_and_exact_discovery_shape():
    worker.admit_allocation(allocation(), producer='grid400', execution_profile=policy.execution_profile())
    assert worker.selected_producer(SimpleNamespace(producer='grid400')) == 'grid400'
    with pytest.raises(ValueError):
        worker.admit_allocation(allocation(), producer='grid400')
    with pytest.raises(ValueError):
        worker.admit_allocation(allocation(), execution_profile=policy.execution_profile())


@pytest.mark.parametrize('field,value', [
    ('Account', 'other'), ('QOS', 'inferno'), ('Partition', 'cpu-large'),
    ('NumCPUs', '12'), ('NumNodes', '2'), ('MinMemoryNode', '64G'),
    ('TimeLimit', '04:00:00')])
def test_worker_allocation_cannot_borrow_the_science_duration_or_another_route(field, value):
    with pytest.raises(ValueError):
        worker.admit_allocation(allocation() | {field: value}, producer='grid400',
                                execution_profile=policy.execution_profile())


def test_grid400_sources_cover_the_actual_cohort_and_transitive_descriptor_modules():
    names = set(worker.source_names('grid400'))
    assert {'grid400_policy.py', 'grid400_cohort.py', 'qualified_grid400.py',
            'resource_plan_grid400.py', 'resource_plan_ed561666.py',
            'resource_plan_cb63305.py', 'qualified_cb63305.py', 'descriptor_profiles.py'} <= names
    assert set(worker.helper('descriptor_profiles').profile('grid400')['child_sources']) <= names
    assert 'owned_dataset_cb63305.py' not in names


def test_grid400_remaining_case_is_rejected_before_inputs_or_manager_access():
    with pytest.raises(ValueError, match='fresh singleton only'):
        worker.source_names('grid400', remaining_case=True)
    with pytest.raises(ValueError, match='new fresh singleton'):
        worker.prepare_draft(SimpleNamespace(producer='grid400', remaining_case=True), None)
