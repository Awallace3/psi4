"""Arithmetic-only resource predictions: no Psi4 import or quantum execution."""
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'campaign_resource_plan', Path(__file__).with_name('resource_plan.py'))
planner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner
spec.loader.exec_module(planner)


def dimensions(symbols):
    heavy, hydrogen = len(symbols)-symbols.count('H'), symbols.count('H')
    electrons = sum({'C': 6, 'H': 1, 'N': 7, 'O': 8}[s] for s in symbols)
    return planner.Dimensions(
        sites=len(symbols), bonds=len(symbols)-1,
        main_nbf=46*heavy+23*hydrogen, main_nshell=14*heavy+9*hydrogen,
        nocc=electrons//2, response_aux_width=136*heavy+55*hydrogen,
        density_aux_width=136*heavy+55*hydrogen,
        atomic_widths=tuple(49 if s == 'H' else 109 for s in symbols),
        shape_widths=tuple(8 if s == 'H' else 12 for s in symbols))


def resources(**changes):
    return dict(max_bytes=12*1024**3, max_work=6_000_000_000_000,
                max_io_bytes=16*1024**3, capacity='large') | changes


def test_water_reproduces_accepted_complete_ledger_not_only_refinement():
    plan = planner.plan(dimensions('OHH'), resources(), max_replay_work=10**12)
    assert plan['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    assert plan['charged_work'] == 2_114_541_737_496
    assert plan['checkpoint_io_bytes'] == 46_480_064
    assert plan['peak_numeric_bytes'] == 1_356_480_224
    assert plan['work_by_stage']['ISA iteration and response-AUX Q'] == 1_988_699_339_376
    assert plan['work_by_stage']['complete-cloud refinement'] == 83_676_664_704
    assert plan['packed_replay']['planned_work'] == 23_834_352_000
    assert plan['packed_replay']['planned_bytes'] == 167_892_464
    assert plan['dimensions']['nvir_envelope'] == 87
    assert plan['dimensions']['isa_grid_accounting_rows'] == 120_582
    assert plan['dimensions']['response_grid_rows'] == 68_310
    assert plan['dimensions']['integration_grid_rows'] == 581_478
    assert plan['native_pfit']['maximum_work_bytes'] == 256*1024**2
    assert plan['native_pfit']['base_bytes'] > 0
    assert plan['exclusions'] and plan['execution_gates']


@pytest.mark.parametrize('symbols', ['COHHHH', 'CNHHHHH'])
def test_next_monomers_refuse_water_work_allowance_without_changing_profile(symbols):
    d = dimensions(symbols)
    plan = planner.plan(d, resources(), max_replay_work=10**12)
    assert plan['work_by_stage']['ISA iteration and response-AUX Q'] > 6*10**12
    assert plan['status'] == 'REFUSED'
    assert 'cumulative_work' in plan['refusals']
    admitted = planner.plan(d, resources(max_work=100*10**12), max_replay_work=2*10**12)
    assert admitted['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    assert admitted['dimensions']['nodes'] == 11
    assert admitted['dimensions']['isa_iterations'] == 120
    assert admitted['dimensions']['parameters'] == 36*len(symbols)


@pytest.mark.parametrize('field,total_key,refusal', [
    ('max_work', 'charged_work', 'cumulative_work'),
    ('max_io_bytes', 'checkpoint_io_bytes', 'checkpoint_io'),
    ('max_bytes', 'admission_numeric_bytes', 'numeric_bytes'),
])
def test_exact_and_one_unit_short_boundaries(field, total_key, refusal):
    d = dimensions('OHH')
    baseline = planner.plan(d, resources(), max_replay_work=10**12)
    exact = resources(**{field: baseline[total_key]})
    assert planner.plan(d, exact, max_replay_work=10**12)['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'
    short = exact | {field: baseline[total_key]-1}
    assert refusal in planner.plan(d, short, max_replay_work=10**12)['refusals']


def test_replay_is_separate_gate_not_double_charged():
    d = dimensions('OHH')
    full = planner.plan(d, resources(), max_replay_work=10**12)
    work = full['packed_replay']['planned_work']
    assert planner.plan(d, resources(), max_replay_work=work)['charged_work'] == full['charged_work']
    refused = planner.plan(d, resources(), max_replay_work=work-1)
    assert 'packed_replay_work' in refused['refusals']
    assert refused['charged_work'] == full['charged_work']


def test_large_independent_site_fit_cannot_hide_native_kernel_cap():
    # Same recipe as benzene, but independent 36 variables per atom, not COPY72.
    plan = planner.plan(dimensions('CCCC CCHHHHHH'.replace(' ', '')),
                        resources(max_work=100*10**12), max_replay_work=2*10**12)
    assert plan['native_pfit']['base_bytes'] > plan['native_pfit']['maximum_work_bytes']
    assert 'native_pfit_base_bytes' in plan['refusals']


@pytest.mark.parametrize('changes', [
    dict(sites=True), dict(main_nbf=0), dict(nocc=1000),
    dict(atomic_widths=(1,)), dict(density_aux_width=245),
    dict(shape_widths=(12, -8, 8)), dict(bonds=-1),
])
def test_invalid_dimensions_cannot_be_admitted(changes):
    from dataclasses import asdict
    with pytest.raises(ValueError):
        planner.Dimensions(**(asdict(dimensions('OHH')) | changes))


@pytest.mark.parametrize('changes', [
    dict(max_work=True), dict(max_bytes=0), dict(capacity='unlimited'),
    dict(max_work=100*10**12+1), dict(max_io_bytes=64*1024**3+1),
])
def test_closed_capacity_limits(changes):
    with pytest.raises(ValueError):
        planner.plan(dimensions('OHH'), resources(**changes), max_replay_work=10**12)


def test_source_binding_covers_python_and_native_policies():
    root = Path(__file__).resolve().parents[2]
    binding = planner.source_binding(root)
    assert binding['scientific_commit'] == 'acf54c28e50822b5b825f4d26e5d9f9a2835d773'
    assert len(binding['files_sha256']) >= 10
    assert 'psi4/src/psi4/libisapol/pfit.cc' in binding['files_sha256']


def test_standard_capacity_is_not_molecular_capacity():
    with pytest.raises(ValueError, match='large'):
        planner.plan(dimensions('OHH'), resources(capacity='standard'), max_replay_work=10**12)


def test_pure_calculator_does_not_claim_source_bound_admission():
    result = planner.plan(dimensions('OHH'), resources(), max_replay_work=10**12)
    assert 'scientific_commit' not in result
    assert 'no source/build/descriptor receipt' in result['qualification']
    assert result['status'] == 'WITHIN_PREDICTED_LEDGER_LIMITS'


def test_packed_subbudget_includes_overhead_for_entire_profile_size_range():
    for sites in range(3, 18):
        plan = planner.plan(dimensions('O'+'H'*(sites-1)),
                            resources(max_work=100*10**12), max_replay_work=2*10**12)
        packed = plan['packed_replay']
        fit = next(s for s in plan['stages'] if s['name'] == 'complete-cloud refinement')
        assert packed['overhead_bytes'] == 24*2000+8*2000*9*sites+17*4096*11
        assert fit['transient_bytes'] > packed['planned_bytes']+packed['overhead_bytes']


def test_missing_or_modified_policy_sources_cannot_bind(tmp_path):
    with pytest.raises(FileNotFoundError):
        planner.source_binding(tmp_path)
    for path in planner.FILES:
        target = tmp_path/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('not the pinned source')
    with pytest.raises(ValueError, match='differ'):
        planner.source_binding(tmp_path)
