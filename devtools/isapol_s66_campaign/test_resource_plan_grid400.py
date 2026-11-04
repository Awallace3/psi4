"""Pure dense-grid accounting; no native imports or scientific calculation."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

path = Path(__file__).with_name('resource_plan_grid400.py')
spec = importlib.util.spec_from_file_location('_grid400_planner_test', path)
planner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner
spec.loader.exec_module(planner)
GIB = 1024**3


def dimensions():
    # Actual predecessor monomer0008 descriptors, used only as arithmetic data.
    return planner.Dimensions(sites=12, bonds=11, main_nbf=391, main_nshell=133,
        nocc=20, response_aux_width=1065, density_aux_width=1065,
        atomic_widths=(109, 49, 49, 49, 109, 109, 109, 49, 109, 49, 49, 49),
        shape_widths=(12, 8, 8, 8, 12, 12, 12, 8, 12, 8, 8, 8))


def compute(**changes):
    resources = dict(max_bytes=48*GIB, max_work=75*10**12,
                     max_io_bytes=16*GIB, capacity='large')
    resources.update(changes)
    return planner.plan(dimensions(), resources, max_replay_work=10**12,
                        pfit_maximum_work_bytes=2*GIB)


def test_dense_grid_charges_every_point_without_changing_other_science():
    new = compute()
    old = planner._legacy.plan(dimensions(), new['resources'],
        max_replay_work=10**12, pfit_maximum_work_bytes=2*GIB)
    assert old['dimensions']['integration_grid_rows'] == 2_325_912
    assert new['dimensions']['integration_grid_rows'] == 9_719_640
    assert dict(new['dimensions'], integration_grid_rows=2_325_912) == old['dimensions']
    dh = 9_719_640-2_325_912
    for current, previous in zip(new['stages'], old['stages'], strict=True):
        expected = dict(previous, reserved_bytes=previous['reserved_bytes']+32*dh,
                        numeric_bytes=previous['numeric_bytes']+32*dh)
        if current['name'] == 'ISA iteration and response-AUX Q':
            extra_bytes = 8*dh*(32*12+16)
            extra_work = 4*dh*300*1065
            expected.update(transient_bytes=previous['transient_bytes']+extra_bytes,
                numeric_bytes=expected['numeric_bytes']+extra_bytes,
                work_per_call=previous['work_per_call']+extra_work,
                charged_work=previous['charged_work']+extra_work)
        assert current == expected
    assert new['peak_numeric_bytes'] == 47_996_666_032
    assert new['charged_work'] == 63_738_054_966_384
    assert new['max_replay_work'] == old['max_replay_work'] == 10**12
    assert new['pfit_maximum_work_bytes'] == old['pfit_maximum_work_bytes'] == 2*GIB
    assert new['checkpoint_io_bytes'] == old['checkpoint_io_bytes']
    assert new['fit_plan'] == old['fit_plan']
    assert new['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    assert new['policy_scientific_commit'] == planner.SCIENTIFIC_COMMIT
    assert old['policy_scientific_commit'] == 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
    assert 'descriptor receipt' in new['qualification']
    assert 'receipt_sha256' not in new and 'implementation_sha256' not in new


def test_original_memory_refuses_without_automatic_promotion():
    result = compute(max_bytes=32*GIB)
    assert result['status'] == 'REFUSED'
    assert result['resources']['max_bytes'] == 32*GIB
    assert result['refusals'] == ['numeric_bytes']
    assert result['peak_numeric_bytes'] == compute()['peak_numeric_bytes']


@pytest.mark.parametrize('field,measured,refusal', [
    ('max_bytes', 'peak_numeric_bytes', 'numeric_bytes'),
    ('max_work', 'charged_work', 'cumulative_work'),
    ('max_io_bytes', 'checkpoint_io_bytes', 'checkpoint_io')])
def test_each_budget_accepts_exact_limit_and_refuses_one_short(field, measured, refusal):
    limit = compute()[measured]
    assert compute(**{field: limit})['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    short = compute(**{field: limit-1})
    assert short['status'] == 'REFUSED' and refusal in short['refusals']
    assert short['resources'][field] == limit-1


def test_candidate_source_binding_rejects_historical_and_changed_bytes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    binding = planner.source_binding(root)
    assert binding['scientific_commit'] == '1aaa9e0a306373b42a3decc25c7dfa106dbc8ea2'
    assert binding['policy_id'] == planner.POLICY
    with pytest.raises(ValueError, match='sources differ'):
        planner._legacy.source_binding(root)
    changed = 'psi4/driver/procrouting/isapol_molecular.py'
    for name, expected in planner.FILES.items():
        raw = subprocess.check_output(
            ['git', 'show', planner.SCIENTIFIC_COMMIT+':'+name], cwd=root)
        assert hashlib.sha256(raw).hexdigest() == expected
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    planner.source_binding(tmp_path)
    target = tmp_path/changed
    target.write_bytes(subprocess.check_output(
        ['git', 'show', planner._legacy.SCIENTIFIC_COMMIT+':'+changed], cwd=root))
    with pytest.raises(ValueError, match='sources differ'):
        planner.source_binding(tmp_path)
    target.write_bytes((root/changed).read_bytes()+b'\n# changed bytes\n')
    with pytest.raises(ValueError, match='sources differ'):
        planner.source_binding(tmp_path)


@pytest.mark.parametrize('grid', [(300, 1454), (400, 974), [400, 2030],
                                  (400.0, 2030), (True, 2030), None])
def test_private_grid_seam_is_closed_not_a_user_tunable_grid(grid):
    with pytest.raises(ValueError, match='closed integration grid'):
        planner._legacy._shared._ledger(dimensions(), compute()['resources'],
            max_replay_work=10**12, pfit_maximum_work_bytes=2*GIB,
            drho_refinement_iterations=32, scientific_commit=planner.SCIENTIFIC_COMMIT,
            policy=planner.POLICY, integration_grid=grid)
