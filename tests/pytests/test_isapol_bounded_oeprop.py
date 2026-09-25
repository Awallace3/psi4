"""Small contract tests for explicit bounded oeprop dispatch and preset policy."""
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_bounded_oeprop as api
from psi4.driver.procrouting.isapol_logging import StageLog


@pytest.fixture
def water():
    return core.Wavefunction.build(psi4.geometry(
        '0 1\nO 0 0 0\nH -.75 0 -.58\nH .75 0 -.58\nsymmetry c1\nno_com\nno_reorient'), 'sto-3g')


@pytest.mark.parametrize('preset', ['water', 'benzene'])
def test_presets_share_camcasp_defaults(preset):
    v = api.settings(preset, {})
    assert (v['radial_points'], v['spherical_points'], v['npoints']) == (100, 200, 2000)
    s = v['smoothing']
    assert (s.rho_epsilon, s.f_max, s.fd_delta, s.fd_alpha, v['shell_cutoff']) == (1e-8, 1000., .01, 1., 1e-8)
    assert (v['weight_type'], v['weight_coefficient'], v['cutoff']) == (3, 1e-3, 1e-4)
    assert (v['rank_limit'], v['hydrogen_rank_limit']) == (2, 2)  # localize.py hlimit=limit
    assert v['declared_variables'] is None  # cutoff-derived, as CamCASP process writes it
    assert v['auxiliary_basis'] == 'aug-cc-pVTZ-RI'
    assert v['anchor_metric_damping'] == .0005
    assert v['scf_correction'] == 'AUTO'


def test_byte_budget_follows_psi4_memory():
    assert api.settings('water', {})['resources'] is None
    psi4.set_memory(3*1024**3)
    r = api.default_resources()
    assert (r.max_bytes, r.max_work, r.max_io_bytes) == (3*1024**3, 6_000_000_000_000, 64*1024**3)


def test_auto_correction_follows_the_scf_functional(water):
    assert api.resolve_correction(water, 'AUTO', None) == ('NONE', None)
    assert api.resolve_correction(water, 'AUTO', .13) == ('FIXED_GRAC', .13)
    assert api.resolve_correction(water, 'NONE', None) == ('NONE', None)


def test_explicit_options_and_keyword_precedence():
    psi4.set_options({'atomic_refinement_points': 48, 'atomic_refinement_weight_coefficient': .0002})
    assert api.settings('benzene', {})['npoints'] == 48
    v = api.settings('benzene', {'npoints': 32})
    assert v['npoints'] == 32 and v['weight_coefficient'] == .0002


@pytest.mark.parametrize('option,value', [('atomic_multipole_distribution','ISA_A'),
    ('atomic_response_basis','DIRECT_OV'), ('atomic_response_localization','LS'), ('atomic_multipole_rank',3)])
def test_conflicting_explicit_model_is_not_silently_ignored(option, value):
    psi4.set_options({option:value})
    with pytest.raises(ValueError, match='conflicts'):
        api.settings('water', {})


def test_bad_keywords_and_preset_fail():
    with pytest.raises(TypeError, match='npointz'):
        api.settings('water', {'npointz':32})
    with pytest.raises(ValueError, match='preset'):
        api.settings('auto', {})


def test_water_axes_and_parameter_model(water):
    sites, bonds = api.preset_sites(water.molecule(), 'water')
    assert bonds == [(1,0),(2,0)]
    assert [s.rank_limit for s in sites] == [2,2,2]
    assert [s.site_type for s in sites] == ['O','H','H']  # H2 is a COPY of H1
    sites, _ = api.preset_sites(water.molecule(), 'water', 2, 1)
    assert [s.rank_limit for s in sites] == [2,1,1]
    for s in sites:
        frame = np.asarray(s.frame)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-15)
    with pytest.raises(ValueError, match='atom order'):
        api.preset_sites(water.molecule(), 'benzene')


def test_public_dispatch_owned_result_and_unchanged_return(water, monkeypatch):
    sentinel, captured = object(), {}
    def backend(wfn, recipe, **kwargs):
        captured.update(kwargs)
        assert wfn is water and recipe.representation == 'Cartesian'
        return sentinel
    monkeypatch.setattr(api, 'bounded_properties', backend)
    assert psi4.oeprop(water, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF',
        preset='water', npoints=32, radial_points=20, spherical_points=50, log=StageLog(0)) is None
    assert psi4.atomic_property_result(water) is sentinel
    assert captured['lattice_options'].npoints == 32
    assert captured['distribution'] == 'df_centre_analytic'
    assert captured['declared_variables'] is None
    assert captured['scf_correction'] == 'NONE' and captured['expected_grac_shift'] is None
    assert captured['weight_type'] == 3
    assert captured['resources'].max_bytes == core.get_memory()


@pytest.mark.parametrize('tasks,kwargs', [
    (('ATOMIC_PARTITION',), {'atomic_backend':'BOUNDED_DF', 'preset':'water'}),
    (('DIPOLE','ATOMIC_REFINED_DISPERSION'), {'atomic_backend':'BOUNDED_DF', 'preset':'water'}),
    (('ATOMIC_REFINED_DISPERSION',), {'atomic_backend':'typo', 'preset':'water'}),
    (('ATOMIC_REFINED_DISPERSION',), {'atomic_backend':'BOUNDED_DF', 'preset':'water', 'npointz':32}),
    (('ATOMIC_REFINED_DISPERSION','ATOMIC_REFINED_DISPERSION'), {'atomic_backend':'BOUNDED_DF', 'preset':'water'}),
])
def test_rejected_dispatch_invalidates_previous_result(water, tasks, kwargs):
    water._native_atomic_property_result = object()
    with pytest.raises((ValueError, TypeError, psi4.driver.p4util.exceptions.ValidationError)):
        psi4.oeprop(water, *tasks, **kwargs)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(water)


def test_missing_seal_is_still_rejected(water):
    with pytest.raises(ValueError):
        psi4.oeprop(water, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF',
                    preset='water', npoints=32, radial_points=20, spherical_points=50, log=StageLog(0))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(water)
