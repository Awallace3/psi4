"""New producer accounting only; never construct Psi4, a state or a grid."""
import importlib.util
import hashlib
import marshal
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load('legacy_resource_plan', 'resource_plan.py')
planner = load('expanded_resource_plan', 'resource_plan_cb63305.py')


def dimensions(symbols, module=planner):
    heavy, hydrogen = len(symbols)-symbols.count('H'), symbols.count('H')
    return module.Dimensions(
        sites=len(symbols), bonds=len(symbols)-1,
        main_nbf=46*heavy+23*hydrogen, main_nshell=14*heavy+9*hydrogen,
        nocc=sum({'C': 6, 'H': 1, 'N': 7, 'O': 8}[s] for s in symbols)//2,
        response_aux_width=136*heavy+55*hydrogen, density_aux_width=136*heavy+55*hydrogen,
        atomic_widths=tuple(49 if s == 'H' else 109 for s in symbols),
        shape_widths=tuple(8 if s == 'H' else 12 for s in symbols))


def resources(**changes):
    return dict(max_bytes=64*1024**3, max_work=128*10**12,
                max_io_bytes=64*1024**3, capacity='extended') | changes


def predict(symbols='OHH', *, cap=2*1024**3, replay=10**12, **changes):
    return planner.plan(dimensions(symbols), resources(**changes),
                        max_replay_work=replay, pfit_maximum_work_bytes=cap)


def test_historical_policy_binds_its_exact_qualified_source_not_current_checkout(tmp_path):
    root = Path(__file__).resolve().parents[2]
    # This policy remains cb63305 even after later scientific source changes.
    # Read the exact Git objects rather than silently accepting today's files
    # or changing the historical production hash map to bless a newer build.
    historical = tmp_path/'cb63305-source'
    for relative, expected in planner.FILES.items():
        raw = subprocess.check_output(
            ['git', 'show', 'cb63305f4115526417a830829a0a30c36b85b4e8:'+relative],
            cwd=root)
        assert hashlib.sha256(raw).hexdigest() == expected
        path = historical/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    binding = planner.source_binding(historical)
    assert binding['scientific_commit'] == 'cb63305f4115526417a830829a0a30c36b85b4e8'
    assert planner.POLICY == 'cb63305-reference-h2h1-ledger-v2'
    assert 'psi4/driver/procrouting/isapol_pfit_plan.py' in binding['files_sha256']
    assert 'psi4/driver/procrouting/isapol_schema.py' in binding['files_sha256']
    assert predict()['schema'] == 'native-campaign-ledger-prediction/2'
    with pytest.raises(ValueError, match='scientific planning sources differ'):
        planner.source_binding(root)


def test_native_cap_is_charged_once_in_the_actual_refinement_stage():
    current = predict(cap=256*1024**2)
    old = legacy.plan(dimensions('OHH', legacy),
        dict(max_bytes=64*1024**3, max_work=100*10**12,
             max_io_bytes=64*1024**3, capacity='large'), max_replay_work=10**12)
    old_fit = next(s for s in old['stages'] if s['name'] == 'complete-cloud refinement')
    fit = next(s for s in current['stages'] if s['name'] == 'complete-cloud refinement')
    assert fit['reserved_bytes'] == old_fit['reserved_bytes']
    assert fit['transient_bytes'] == old_fit['transient_bytes']+256*1024**2
    assert current['admission_numeric_bytes'] == current['peak_numeric_bytes']
    assert current['charged_work'] == old['charged_work'] == 2_114_541_737_496
    assert current['checkpoint_io_bytes'] == old['checkpoint_io_bytes'] == 46_480_064


@pytest.mark.parametrize('sites', range(3, 18))
def test_shared_plan_parity_preserves_every_scientific_dimension(sites):
    symbols = 'O'+'H'*(sites-1)  # dimension-only examples, not proposed molecular states
    current = predict(symbols)
    shared = planner.shared_pfit_plan()
    fit = shared.combined_fit_plan(2000, 9*sites, 36*sites, 11,
                                   maximum_work_bytes=2*1024**3)
    observed = next(s for s in current['stages'] if s['name'] == 'complete-cloud refinement')
    assert observed['transient_bytes'] == fit['complete_refinement_bytes']
    assert observed['charged_work'] == fit['complete_refinement_work']
    assert current['native_pfit']['base_bytes'] == fit['native_base_bytes']
    assert current['dimensions']['fit_points'] == 2000
    assert current['dimensions']['parameters'] == 36*sites
    assert current['dimensions']['nodes'] == 11
    old = legacy.plan(dimensions(symbols, legacy),
        resources(capacity='large', max_work=100*10**12), max_replay_work=10**12)
    assert current['charged_work'] == old['charged_work']
    assert current['checkpoint_io_bytes'] == old['checkpoint_io_bytes']


def test_full_pentane_envelope_uses_extended_without_discounting_work():
    current = predict('CCCCC'+'H'*12)
    assert current['charged_work'] == 105_124_176_150_908
    assert current['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    refused = predict('CCCCC'+'H'*12, capacity='large', max_work=100*10**12)
    assert 'cumulative_work' in refused['refusals']
    assert refused['charged_work'] == current['charged_work']
    assert current['native_pfit']['base_bytes'] == 1_363_037_344


@pytest.mark.parametrize('field,key,reason', [
    ('max_bytes', 'admission_numeric_bytes', 'numeric_bytes'),
    ('max_work', 'charged_work', 'cumulative_work'),
    ('max_io_bytes', 'checkpoint_io_bytes', 'checkpoint_io'),
])
def test_exact_and_one_short_whole_chain_budgets(field, key, reason):
    baseline = predict()
    assert predict(**{field: baseline[key]})['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    assert reason in predict(**{field: baseline[key]-1})['refusals']


def test_native_base_and_replay_refusals_are_not_promoted_or_retried():
    with pytest.raises(ValueError, match='native PFIT workspace budget'):
        predict('CCCCCC'+'H'*6, cap=256*1024**2)
    current = predict()
    required = current['packed_replay']['planned_work']
    assert predict(replay=required)['charged_work'] == current['charged_work']
    with pytest.raises(ValueError, match='replay work'):
        predict(replay=required-1)


@pytest.mark.parametrize('changes', [
    dict(capacity='standard'), dict(capacity='unlimited'), dict(max_work=True),
    dict(max_work=128*10**12+1), dict(max_io_bytes=64*1024**3+1),
])
def test_closed_resource_tiers_are_still_finite(changes):
    with pytest.raises(ValueError):
        predict(**changes)


def test_pure_prediction_does_not_become_a_bound_admission_receipt():
    current = predict()
    assert 'scientific_commit' not in current
    assert 'no source/build/descriptor receipt' in current['qualification']
    assert current['exclusions'] and current['execution_gates']


def test_verified_helper_never_executes_a_cached_bytecode_payload(monkeypatch, tmp_path):
    target = tmp_path/'isapol_pfit_plan.py'
    target.write_bytes((Path(__file__).resolve().parents[2]/planner.PFIT_SOURCE).read_bytes())
    cache = Path(importlib.util.cache_from_source(str(target)))
    cache.parent.mkdir()
    # Valid unchecked-hash pyc header: a SourceFileLoader can accept this even
    # with -B. The underlying .py remains byte-identical to the qualified pin.
    payload = compile("raise AssertionError('poisoned helper cache executed')", str(target), 'exec')
    cache.write_bytes(importlib.util.MAGIC_NUMBER+struct.pack('<I', 1)+b'\0'*8+marshal.dumps(payload))
    installed = ModuleType('psi4.driver.procrouting.isapol_pfit_plan')
    installed.__file__ = str(target)
    monkeypatch.setitem(sys.modules, installed.__name__, installed)
    assert planner.shared_pfit_plan().native_work_limit() == 256*1024**2
