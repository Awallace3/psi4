"""Closed descriptor scopes and synthetic extraction; no native science."""
import importlib.util
from pathlib import Path
import sys

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiles = load('_descriptor_profile_test', 'descriptor_profiles.py')
admission = load('_selected_descriptor_admission_test', 'descriptor_admission_cb63305.py')
fixtures = load('_grid400_descriptor_probe_fixtures', 'test_descriptor_probe_cb63305.py')


@pytest.mark.parametrize('producer,cap,grid,selection', [
    ('cb63305', 10, (200, 974), None),
    ('ed561666', 32, (200, 974), None),
    ('grid400', 32, (400, 2030), ('monomer-0008',))])
def test_closed_descriptor_profile_preserves_historical_scope(producer, cap, grid, selection):
    profile = profiles.profile(producer)
    assert profile['drho_refinement_iterations'] == cap
    assert profile['integration_grid'] == grid
    assert profile['selected_monomers'] == selection
    assert profile['qualifier'] == 'qualified_'+producer
    assert profile['planner'] == 'resource_plan_'+producer
    assert {'resource_plan_cb63305.py', 'descriptor_profiles.py'} <= set(profile['child_sources'])
    profile['selected_monomers'] = ('monomer-9999',)
    assert profiles.profile(producer)['selected_monomers'] == selection


@pytest.mark.parametrize('producer', [None, True, '../qualified_grid400', 'latest'])
def test_unknown_profiles_refuse_before_any_native_import(producer):
    with pytest.raises(ValueError, match='closed installed descriptor'):
        profiles.profile(producer)
    with pytest.raises(ValueError, match='closed installed descriptor'):
        admission.planner(producer)


def test_grid400_source_inventory_includes_every_transitive_planner():
    profile = profiles.profile('grid400')
    assert set(profile['child_sources']) == {
        'descriptor_probe_cb63305.py', 'descriptor_probe.py', 'descriptor_profiles.py',
        'resource_plan_cb63305.py', 'resource_plan_ed561666.py', 'resource_plan_grid400.py'}
    assert set(profile['parent_extra_sources']) == {
        'qualified_grid400.py', 'resource_plan_ed561666.py', 'resource_plan_grid400.py'}
    assert admission.planner('grid400').POLICY == '1aaa9e0a30-reference-h2h1-grid400-ledger-v4'


def test_grid400_descriptor_uses_dense_arithmetic_and_unchanged_native32(monkeypatch):
    probe = fixtures.probe
    runtime, calls, request = fixtures.mock_boundaries(monkeypatch, producer='grid400')
    request.resources.max_bytes = 48*1024**3
    request.resources.max_work = 75*10**12
    request.pfit_maximum_work_bytes = 2*1024**3
    report = probe.scientific_probe(runtime, memory_bytes=60*1024**3, producer='grid400')
    assert report['plan']['dimensions']['integration_grid_rows'] == 3*399*2030
    assert report['plan']['dimensions']['response_grid_rows'] == 3*99*230
    assert report['plan']['dimensions']['drho_refinement_iterations'] == 32
    assert 'integration400/2030' in report['grid_policy']
    assert report['max_replay_work'] == 10**12
    assert calls['plan_kwargs']['pfit_maximum_work_bytes'] == 2*1024**3
    monkeypatch.setattr(sys.modules['psi4.driver.procrouting.isapol_native_partition'],
                        'DRHO_REFINEMENT_ITERATIONS', 10)
    with pytest.raises(ValueError, match='refinement policy changed'):
        probe.scientific_probe(runtime, memory_bytes=60*1024**3, producer='grid400')


def test_selected_coverage_is_not_all131_and_refusals_stay_explicit():
    passed = [dict(monomer_id='monomer-0008', status='PASS_DESCRIPTOR_PLAN_ONLY')]
    assert admission.verify_grid400_coverage(passed) == 'PASS_SELECTED_MONOMER_DESCRIPTOR_PLAN_ONLY'
    with pytest.raises(ValueError, match='exact131'):
        admission.verify_coverage(passed, {'monomer-0008'})
    refused = [dict(monomer_id='monomer-0008', status='REFUSED_DESCRIPTOR_PLAN')]
    assert admission.verify_grid400_coverage(refused) == 'REFUSED_SELECTED_MONOMER_DESCRIPTOR_PLAN'


@pytest.mark.parametrize('reports', [
    [], [dict(monomer_id='monomer-0009', status='PASS_DESCRIPTOR_PLAN_ONLY')],
    [dict(monomer_id='monomer-0008', status='passed')],
    [dict(monomer_id='monomer-0008', status='PASS_DESCRIPTOR_PLAN_ONLY')]*2,
    [dict(monomer_id='monomer-0008', status='PASS_ALL_131_DESCRIPTOR_PLANS_ONLY')]])
def test_wrong_duplicate_or_relabelled_selected_coverage_refuses(reports):
    with pytest.raises(ValueError, match='selected monomer0008'):
        admission.verify_grid400_coverage(reports)
