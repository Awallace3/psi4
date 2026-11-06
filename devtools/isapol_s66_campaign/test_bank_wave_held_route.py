"""Task130 pending singleton display regression; no scheduler or calculation."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

# Actual selected tokenized fields retained by diagnostic bbebd4f45 for the
# unused HELD job. The original full scontrol -o row was NOT retained. Do not
# present this dictionary (or any joined reconstruction) as a raw-line capture.
CAPTURED_SELECTED_FIELDS_15228597 = dict(
    JobId='15228597', JobState='PENDING', Reason='JobHeldUser',
    Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small',
    NumCPUs='8', NumNodes='1-1', MinMemoryNode='32G', TimeLimit='02:00:00', Requeue='0')


def controller():
    path = Path(__file__).with_name('bank_wave_controller.py')
    spec = importlib.util.spec_from_file_location('_task130_held_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_route(name='s20'):
    """Other route values are synthetic exact finite declarations, not captures."""
    module = controller()
    profile = module.helper('bank_wave').helper('grid400_bank_policy').execution_profile(name)
    case = dict(execution_profile=profile)
    route = dict(JobId='999', JobState='PENDING', Reason='JobHeldUser',
        Account='gts-cs207-chemx', QOS='embers', Partition='cpu-small', NumNodes='1',
        NumCPUs=str(profile['cores']), MinMemoryNode=str(profile['job_memory_GiB'])+'G',
        TimeLimit=profile['time_limit'], Requeue='0', Priority='0', UserId='awallace43(999)')
    return module, case, route


def test_actual_selected_fields_capture_15228597_pending_singleton_range():
    module = controller()
    profile = module.helper('bank_wave').helper('grid400_bank_policy').execution_profile('s20')
    assert (profile['cores'], profile['job_memory_GiB'], profile['time_limit']) == (8, 32, '02:00:00')
    captured = copy.deepcopy(CAPTURED_SELECTED_FIELDS_15228597)
    # These two fields were NOT in the retained selected-fields capture.
    # Complete the isolated predicate harness only; do not attribute them to
    # bbebd4f45 or manufacture a full pre-cancellation scontrol observation.
    harness_only = dict(Priority='0', UserId='awallace43(SYNTHETIC_TEST_UID)')
    module.verify_held(dict(captured, **harness_only), dict(execution_profile=profile), '15228597')
    assert captured == CAPTURED_SELECTED_FIELDS_15228597


def test_actual_post_cancellation_raw_row_is_parsed_but_never_accepted_as_held(monkeypatch):
    # Exact post-cancellation stdout from diagnostic b965dba9a, NOT the lost
    # pre-cancel held row. Preserve the raw bytes/state; never rewrite them.
    raw = Path(__file__).with_name('task130_cancelled_15228597.scontrol.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'f83a19aca81e7755bcfdae5669d24f00f86c4626eb67dffcc08bd89d1507a944'
    module = controller()
    profile = module.helper('bank_wave').helper('grid400_bank_policy').execution_profile('s20')
    binding = dict(plan_path=Path('/synthetic/wave-0025-0033/plan.json'),
                   plan=dict(member_ids=['monomer-0026']))
    expected_command = ['SYNTHETIC_SUBMISSION_RECEIPT_ONLY']
    monkeypatch.setattr(module, 'read', lambda path:
        dict(command=expected_command, job_id='15228597') if path.name == 'submission.json'
        else dict(execution_profile=profile))
    monkeypatch.setattr(module, 'phase_command', lambda *args: expected_command)
    def observed(command, *, text):
        assert command == [module.SLURM+'scontrol', 'show', 'job', '-o', '15228597']
        assert text is True
        return raw.decode()
    monkeypatch.setattr(module.subprocess, 'check_output', observed)
    original_check = module.verify_held
    def check_cancellation(allocation, case, actual_job):
        assert allocation['JobState'] == 'CANCELLED'
        assert allocation['RunTime'] == '00:00:00'
        assert allocation['NumNodes'] == '1'
        assert allocation['Command'].endswith('/phoenix_pilot_cb63305.sbatch')
        original_check(allocation, case, actual_job)
    monkeypatch.setattr(module, 'verify_held', check_cancellation)
    with pytest.raises(ValueError, match='held on the exact finite route'):
        module.Production(None).held(binding)


@pytest.mark.parametrize('shape', ['s20', 's40', 's60', 's80', 's120'])
@pytest.mark.parametrize('nodes', ['1', '1-1'])
def test_exact_pending_singleton_displays_preserve_all_finite_shapes(shape, nodes):
    module, case, route = synthetic_route(shape)
    route['NumNodes'] = nodes
    original = copy.deepcopy(route)
    module.verify_held(route, case, '999')
    assert route == original  # Normalize the check only, never captured data.


@pytest.mark.parametrize('nodes', ['1-2', '2', '2-2', '0-1', '0', '01', '1-01',
    '01-01', '+1', '1-', '-1', '1 - 1', '1,1', '1-1-1', '1-1\n', ' 1', '1 ',
    '', None, 1, 1.0, True, [], {}])
def test_wider_or_malformed_pending_node_displays_remain_rejected(nodes):
    module, case, route = synthetic_route()
    route['NumNodes'] = nodes
    with pytest.raises(ValueError, match='held on the exact finite route'):
        module.verify_held(route, case, '999')


@pytest.mark.parametrize('key,value', [
    ('JobId', '1000'), ('JobState', 'RUNNING'), ('Reason', 'Priority'),
    ('Reason', 'JobHeldAdmin'), ('Account', 'other-account'), ('QOS', 'inferno'),
    ('Partition', 'cpu-large'), ('NumCPUs', '4'), ('MinMemoryNode', '24G'),
    ('TimeLimit', '00:45:00'), ('Requeue', '1'), ('Priority', '100'),
    ('UserId', 'another-user(999)'),
])
def test_singleton_range_does_not_relax_any_other_held_route_field(key, value):
    module, case, route = synthetic_route()
    route.update(NumNodes='1-1')
    route[key] = value
    with pytest.raises(ValueError, match='held on the exact finite route'):
        module.verify_held(route, case, '999')
