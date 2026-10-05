# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit bounded oeprop dispatch, preset policy, ATOMIC_* option activation and publication."""
import subprocess
import sys

import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_bounded_oeprop as api
from psi4.driver.procrouting.isapol_logging import StageLog, _report_owned

ValidationError = psi4.driver.p4util.exceptions.ValidationError
TOY = 'O 0 0 0\nH .757 0 .586\nH -.757 0 .586'

#: Registered defaults of every BOUNDED_DF option (read_options.cc), which the
#: preset ignores unless the option was explicitly set, and the preset values.
OPTION_DEFAULTS = {
    'ATOMIC_PROPERTY_AUXILIARY_BASIS': ('cc-pVDZ-JKFIT', 'aug-cc-pVTZ-RI'),
    'ATOMIC_RESPONSE_RADIAL_POINTS': (99, 100), 'ATOMIC_RESPONSE_SPHERICAL_POINTS': (590, 200),
    'ATOMIC_REFINEMENT_POINTS': (500, 2000), 'ATOMIC_REFINEMENT_SEED': (1, 1),
    'ATOMIC_REFINEMENT_LOWER_LIMIT': (2., 2.), 'ATOMIC_REFINEMENT_UPPER_LIMIT': (4., 4.),
    'ATOMIC_REFINEMENT_WEIGHT_TYPE': (4, 3), 'ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT': (1e-3, 1e-3),
    'ATOMIC_REFINEMENT_CUTOFF': (1e-4, 1e-4), 'ATOMIC_LOCALIZATION_RANK_LIMIT': (3, 2),
    'ATOMIC_REFINEMENT_RANK_LIMIT': (2, 2), 'ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT': (1, 2),
    'ATOMIC_OV_CHARGE_PENALTY': (1., 1000.), 'ATOMIC_OV_METRIC_DAMPING': (0., .0005),
    'ATOMIC_SCF_ASYMPTOTIC_CORRECTION': ('NONE', 'AUTO'), 'ATOMIC_SCF_EXPECTED_GRAC_SHIFT': (0., None),
    'ATOMIC_PROPERTY_PRINT': (1, 1),
}
CHANGED = {'ATOMIC_PROPERTY_AUXILIARY_BASIS': 'def2-universal-jkfit', 'ATOMIC_RESPONSE_RADIAL_POINTS': 30,
           'ATOMIC_RESPONSE_SPHERICAL_POINTS': 86, 'ATOMIC_REFINEMENT_POINTS': 48, 'ATOMIC_REFINEMENT_SEED': 5,
           'ATOMIC_REFINEMENT_LOWER_LIMIT': 1.5, 'ATOMIC_REFINEMENT_UPPER_LIMIT': 3.5,
           'ATOMIC_REFINEMENT_WEIGHT_TYPE': 5, 'ATOMIC_REFINEMENT_WEIGHT_COEFFICIENT': 2e-4,
           'ATOMIC_REFINEMENT_CUTOFF': 1e-3, 'ATOMIC_LOCALIZATION_RANK_LIMIT': 1,
           'ATOMIC_REFINEMENT_RANK_LIMIT': 1, 'ATOMIC_REFINEMENT_HYDROGEN_RANK_LIMIT': 3,
           'ATOMIC_OV_CHARGE_PENALTY': 10., 'ATOMIC_OV_METRIC_DAMPING': .01,
           'ATOMIC_SCF_ASYMPTOTIC_CORRECTION': 'FIXED_GRAC', 'ATOMIC_SCF_EXPECTED_GRAC_SHIFT': .1,
           'ATOMIC_PROPERTY_PRINT': 2}


@pytest.fixture
def water():
    core.clean_options()
    return core.Wavefunction.build(psi4.geometry(
        '0 1\nO 0 0 0\nH -.75 0 -.58\nH .75 0 -.58\nsymmetry c1\nno_com\nno_reorient'), 'sto-3g')


def _owned(wfn):
    return {k for k in list(wfn.scalar_variables())+list(wfn.array_variables())
            if _report_owned(k, *api.REFINED_DISPERSION_NAMESPACE)}


@pytest.mark.parametrize('preset', ['water', 'benzene'])
def test_presets_share_camcasp_defaults(preset):
    core.clean_options()
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
    assert (v['distribution'], v['response'], v['fdds']) == ('df_centre_analytic', 'reference_h2h1', None)


def test_option_defaults_never_leak_and_each_explicit_option_activates():
    """Registered defaults differ from the preset; only explicitly set options apply."""
    core.clean_options()
    assert set(OPTION_DEFAULTS) == set(api.OPTION_MAP.values())
    key_of = {option: key for key, option in api.OPTION_MAP.items()}
    preset = api.settings('water', {})
    for option, (registered, preset_value) in OPTION_DEFAULTS.items():
        assert core.get_global_option(option) == registered and not core.has_global_option_changed(option)
        assert preset[key_of[option]] == preset_value
    for option, value in CHANGED.items():
        core.clean_options()
        core.set_global_option(option, value)
        v = api.settings('water', {})
        assert v[key_of[option]] == core.get_global_option(option)
        assert core.get_global_option(option) in (value, value.upper() if isinstance(value, str) else value)
        assert all(v[k] == preset[k] for k in api.OPTION_MAP if k != key_of[option])
    core.clean_options()


def test_byte_budget_follows_psi4_memory():
    core.clean_options()
    assert api.settings('water', {})['resources'] is None
    saved = core.get_memory()
    try:
        psi4.set_memory(3*1024**3, quiet=True)
        r = api.default_resources()
        assert (r.max_bytes, r.max_work, r.max_io_bytes) == (3*1024**3, 6_000_000_000_000, 64*1024**3)
    finally:
        psi4.set_memory(saved, quiet=True)


def test_auto_correction_follows_the_scf_functional(water):
    assert api.resolve_correction(water, 'AUTO', None) == ('NONE', None)
    assert api.resolve_correction(water, 'AUTO', .13) == ('FIXED_GRAC', .13)
    assert api.resolve_correction(water, 'NONE', None) == ('NONE', None)


def test_explicit_options_and_keyword_precedence():
    core.clean_options()
    psi4.set_options({'atomic_refinement_points': 48, 'atomic_refinement_weight_coefficient': .0002})
    assert api.settings('benzene', {})['npoints'] == 48
    v = api.settings('benzene', {'npoints': 32})
    assert v['npoints'] == 32 and v['weight_coefficient'] == .0002
    core.clean_options()


@pytest.mark.parametrize('keyword', ['npointz', 'partition_recipe', 'partition_grid'])
def test_bad_keywords_and_preset_fail(keyword):
    with pytest.raises(TypeError, match=keyword):
        api.settings('water', {keyword: 32})
    with pytest.raises(ValueError, match='preset'):
        api.settings('auto', {})


def test_water_axes_and_parameter_model(water):
    sites, bonds = api.preset_sites(water.molecule(), 'water')
    assert bonds == [(1, 0), (2, 0)]
    assert [s.rank_limit for s in sites] == [2, 2, 2]
    assert [s.site_type for s in sites] == ['O', 'H', 'H']  # H2 is a COPY of H1
    sites, _ = api.preset_sites(water.molecule(), 'water', 2, 1)
    assert [s.rank_limit for s in sites] == [2, 1, 1]
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
    assert psi4.oeprop(water, 'ATOMIC_REFINED_POLARIZABILITIES', atomic_backend='BOUNDED_DF',
        preset='water', npoints=32, radial_points=20, spherical_points=50, log=StageLog(0)) is None
    assert psi4.atomic_property_result(water) is sentinel
    assert captured['lattice_options'].npoints == 32
    assert (captured['distribution'], captured['distributed_moments']) == ('df_centre_analytic', None)
    assert (captured['response'], captured['fdds']) == ('reference_h2h1', None)
    assert captured['declared_variables'] is None
    assert captured['scf_correction'] == 'NONE' and captured['expected_grac_shift'] is None
    assert captured['weight_type'] == 3
    assert captured['resources'].max_bytes == core.get_memory()
    assert captured['publish_qcvariables'] is False  # only the dispersion task publishes
    psi4.oeprop(water, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF', preset='water',
                npoints=32, radial_points=20, spherical_points=50, log=StageLog(0))
    assert captured['publish_qcvariables'] is True


@pytest.mark.parametrize('tasks,kwargs', [
    (('DIPOLE', 'ATOMIC_PARTITION'), {}),
    (('DIPOLE', 'ATOMIC_REFINED_DISPERSION'), {'atomic_backend': 'BOUNDED_DF', 'preset': 'water'}),
    (('ATOMIC_REFINED_DISPERSION',), {'atomic_backend': 'typo', 'preset': 'water'}),
    (('ATOMIC_REFINED_DISPERSION',), {'atomic_backend': 'BOUNDED_DF', 'preset': 'water', 'npointz': 32}),
    (('ATOMIC_REFINED_DISPERSION',), {'atomic_backend': 'BOUNDED_DF', 'preset': 'water',
                                      'distribution': 'isa'}),
    (('ATOMIC_REFINED_DISPERSION', 'ATOMIC_REFINED_DISPERSION'), {'atomic_backend': 'BOUNDED_DF',
                                                                  'preset': 'water'}),
])
def test_rejected_dispatch_invalidates_previous_result_and_variables(water, tasks, kwargs):
    water._native_atomic_property_result = object()
    water.set_variable('ATOMIC REFINED DISPERSION C6 TOTAL', 1.)
    water.set_variable('ATOM O C6 REFINED DISPERSION COEFFICIENT INCOMPLETE', 2.)
    water.set_variable('ATOMIC DISPERSION C6 TOTAL', 3.)  # unrefined look-alike: not owned
    water.set_variable('ATOM O CHARGE', 4.)
    with pytest.raises((ValueError, TypeError, ValidationError)):
        psi4.oeprop(water, *tasks, **dict(kwargs, npoints=32, radial_points=20, spherical_points=50,
                                          log=StageLog(0)) if 'preset' in kwargs else kwargs)
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(water)
    assert not _owned(water)
    assert water.variable('ATOMIC DISPERSION C6 TOTAL') == 3. and water.variable('ATOM O CHARGE') == 4.


def test_missing_seal_is_still_rejected(water):
    with pytest.raises(ValueError, match='functional state|convergence evidence'):
        psi4.oeprop(water, 'ATOMIC_REFINED_DISPERSION', atomic_backend='BOUNDED_DF',
                    preset='water', npoints=32, radial_points=20, spherical_points=50, log=StageLog(0))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(water)


def test_ordinary_oeprop_is_unrouted_and_leaves_atomic_state_alone():
    """No ATOMIC_* name and no backend: plain OEProp, no ISA-Pol import, nothing invalidated."""
    script = '''
import sys, psi4
psi4.core.be_quiet()
mol = psi4.geometry("0 1\\nO 0 0 0\\nH .757 0 .586\\nH -.757 0 .586\\nsymmetry c1")
e, wfn = psi4.energy('scf/sto-3g', molecule=mol, return_wfn=True)
wfn._native_atomic_property_result = sentinel = object()
wfn.set_variable('ATOMIC REFINED DISPERSION C6 TOTAL', 1.)
psi4.oeprop(wfn, 'DIPOLE', 'MULLIKEN_CHARGES', title='T')
assert wfn._native_atomic_property_result is sentinel
assert wfn.variable('ATOMIC REFINED DISPERSION C6 TOTAL') == 1.
assert wfn.has_variable('T DIPOLE') and wfn.has_array_variable('MULLIKEN CHARGES')
loaded = sorted(m for m in sys.modules if 'isapol' in m)
assert not loaded, loaded
print('OK')
'''
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith('OK'), out.stderr


@pytest.fixture(scope='module')
def sealed_water():
    core.be_quiet()
    core.clean_options()
    molecule = psi4.geometry('0 1\n'+TOY+'\nsymmetry c1\nno_com\nno_reorient')
    psi4.set_options({'basis': 'sto-3g', 'reference': 'rks', 'scf_type': 'pk',
                      'e_convergence': 1e-10, 'd_convergence': 1e-10,
                      'dft_radial_points': 50, 'dft_spherical_points': 110})
    _, wfn = psi4.energy('pbe0', molecule=molecule, return_wfn=True)
    core.clean_options()
    return wfn


def test_live_oeprop_publishes_latest_result_and_clears_it_on_refusal(sealed_water, tmp_path):
    from psi4.driver.procrouting.isapol_native import Quadrature
    wfn = sealed_water
    small = dict(atomic_backend='BOUNDED_DF', preset='water', npoints=32, radial_points=20, spherical_points=50,
                 rank_limit=1, hydrogen_rank_limit=1, localization_rank_limit=1, weight_type=4,
                 weight_coefficient=1e-5, max_order=8, quadrature=Quadrature.from_casimir(core.CasimirGrid(2, .5)),
                 scratch_directory=tmp_path, log=StageLog(0))
    wfn.set_variable('UNRELATED SENTINEL', 5.)
    assert psi4.oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES', **small) is None
    polarizabilities = psi4.atomic_property_result(wfn)
    assert len(polarizabilities.refinements) == 3 and not _owned(wfn)
    psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', 'ATOMIC_REFINED_POLARIZABILITIES', **small)
    result = psi4.atomic_property_result(wfn)
    assert result is not polarizabilities
    for a, b in zip(result.refined_tensors[1], polarizabilities.refined_tensors[1]):
        np.testing.assert_array_equal(a, b)
    published = _owned(wfn)
    # Rank-1 sites: C6 is complete, C8 needs rank 2 and is published as INCOMPLETE.
    pairs = {(p.label_a, p.label_b): p for p in result.dispersion.pairs}
    c6, c8 = pairs['O', 'H1'].coefficients
    assert c6.unrestricted_complete and not c8.unrestricted_complete
    assert wfn.variable('ATOMIC REFINED DISPERSION C6 O H1') == c6.value
    assert wfn.variable('ATOMIC REFINED DISPERSION C8 O H1 INCOMPLETE') == c8.value
    assert 'ATOMIC REFINED DISPERSION C8 O H1' not in published
    assert 'ATOMIC REFINED DISPERSION C6 TOTAL' in published
    assert 'ATOMIC REFINED DISPERSION C8 TOTAL INCOMPLETE' in published
    assert wfn.variable('ATOMIC REFINED DISPERSION MAX ORDER') == 8.
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='localization rank'):
        psi4.oeprop(wfn, 'ATOMIC_REFINED_DISPERSION', **dict(small, rank_limit=2))
    with pytest.raises(ValueError, match='No native'):
        psi4.atomic_property_result(wfn)
    assert not _owned(wfn) and wfn.variable('UNRELATED SENTINEL') == 5.
