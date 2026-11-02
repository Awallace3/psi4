"""Pure cap32 accounting and source-boundary checks; no Psi4 import."""
import importlib.util
from pathlib import Path
import sys

import pytest

path = Path(__file__).with_name('resource_plan_ed561666.py')
spec = importlib.util.spec_from_file_location('drho32_planner_test', path)
planner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner
spec.loader.exec_module(planner)


def dimensions():
    # Measured benzene dimensions, but an arithmetic-only calculation here.
    return planner.Dimensions(sites=12, bonds=12, main_nbf=414, main_nshell=138,
        nocc=21, response_aux_width=1146, density_aux_width=1146,
        atomic_widths=(109, 49)*6, shape_widths=(12, 8)*6)


def compute(resources=None):
    return planner.plan(dimensions(), resources or dict(
        max_bytes=32*1024**3, max_work=75*10**12, max_io_bytes=8*1024**3, capacity='large'),
        max_replay_work=2*10**12, pfit_maximum_work_bytes=2*1024**3)


def test_new_source_binding_does_not_relabel_historical_policy():
    root = Path(__file__).resolve().parents[2]
    binding = planner.source_binding(root)
    assert binding['scientific_commit'] == 'ed561666fcac212fffa774429fc3bd8f8cfc903d'
    assert binding['policy_id'] == 'ed561666-reference-h2h1-ledger-v3'
    with pytest.raises(ValueError, match='sources differ'):
        planner._shared.source_binding(root)
    assert planner._shared.SCIENTIFIC_COMMIT == 'cb63305f4115526417a830829a0a30c36b85b4e8'


def test_all_extra_work_is_charged_and_historical_values_are_unchanged():
    new = compute()
    old = planner._shared.plan(dimensions(), new['resources'],
        max_replay_work=2*10**12, pfit_maximum_work_bytes=2*1024**3)
    d = dimensions().density_aux_width
    delta = 22*(256*d*(d+1)+16384*d+16*d*d+64*d)
    assert delta == 8_280_024_192
    assert new['charged_work'] == old['charged_work']+delta
    assert new['dimensions']['drho_refinement_iterations'] == 32
    assert old['dimensions']['drho_refinement_iterations'] == 10
    assert old['policy_id'] == planner._shared.POLICY
    assert old['policy_scientific_commit'] == planner._shared.SCIENTIFIC_COMMIT
    assert new['peak_numeric_bytes'] == old['peak_numeric_bytes']
    assert new['checkpoint_io_bytes'] == old['checkpoint_io_bytes']
    for a, b in zip(new['stages'], old['stages']):
        if a['name'] == 'ISA iteration and response-AUX Q':
            assert a['charged_work'] == b['charged_work']+delta
        else:
            assert a == b
    assert new['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'


def test_updated_work_has_exact_one_short_boundary_and_no_automatic_promotion():
    base = compute()
    resources = dict(base['resources'], max_work=base['charged_work'])
    assert compute(resources)['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    refused = compute(dict(resources, max_work=resources['max_work']-1))
    assert refused['status'] == 'REFUSED' and 'cumulative_work' in refused['refusals']
    assert refused['charged_work'] == base['charged_work']
    assert 'descriptor receipt' in base['qualification']
    assert 'implementation_sha256' not in base and 'receipt_sha256' not in base


def test_undeclared_current_source_is_rejected(tmp_path):
    # Exact source guards run before a prediction is considered for admission.
    root = Path(__file__).resolve().parents[2]
    for name in planner.FILES:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root/name).read_bytes())
    planner.source_binding(tmp_path)
    target = tmp_path/'psi4/driver/procrouting/isapol_native_partition.py'
    target.write_bytes(target.read_bytes()+b'\n# changed bytes\n')
    with pytest.raises(ValueError, match='sources differ'):
        planner.source_binding(tmp_path)
