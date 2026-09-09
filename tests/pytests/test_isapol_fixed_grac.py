"""Requires rebuilt core getters. One fresh small genuine fixed-GRAC water SCF.

No reference tables, orbital repair, inferred shift, or GRAC response derivative.
The modest real endpoint is the native ALDA producer, not a matched atomic preset.
"""
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.p4util import OptionsState
from psi4.driver.procrouting import isapol_native_correction as correction
from psi4.driver.procrouting import isapol_native as native
from psi4.driver.procrouting import isapol_native_response as response
from psi4.driver.procrouting import isapol_oeprop as api
from psi4.driver.procrouting.scf_proc.scf_iterator import _scf_state_signature

SHIFT = .06490004527520865  # caller supplied example, never an API default
DECLARATION = dict(scf_correction='FIXED_GRAC', expected_grac_shift=SHIFT)


@contextmanager
def options(values):
    saved = OptionsState(*[[k] if not isinstance(k, tuple) else list(k) for k in values])
    try:
        for key, value in values.items():
            if isinstance(key, tuple):
                core.set_local_option(*key, value)
            else:
                core.set_global_option(key, value)
        yield
    finally:
        saved.restore()


@pytest.fixture(scope='module')
def water():
    scf = dict(reference='rks', scf_type='pk', guess='core', df_scf_guess=False,
               maxiter=100, fail_on_maxiter=True, e_convergence=1e-10, d_convergence=1e-10,
               orbital_optimizer_package='internal', dft_alpha=.25,
               dft_grac_shift=SHIFT, dft_grac_alpha=.5, dft_grac_beta=40.,
               dft_grac_x_func='XC_GGA_X_LB', dft_grac_c_func='XC_LDA_C_VWN')
    with options({'BASIS': 'sto-3g', **{('SCF', k.upper()): v for k, v in scf.items()}}):
        mol = psi4.geometry('O 0 0 0\nH 1.45365196 0 -1.12168732\n'
                            'H -1.45365196 0 -1.12168732\nunits bohr\n'
                            'symmetry c1\nno_reorient\nno_com')
        _, w = psi4.energy('pbe0', molecule=mol, return_wfn=True)
    return w


@pytest.fixture(autouse=True)
def public_defaults():
    with options({'PARTITION_SCHEME': 'ISA_A', 'ATOMIC_RESPONSE_LOCALIZATION': 'LW',
                  'ATOMIC_PROPERTY_RECIPE': 'GENERATED_JKFIT_ISA_A',
                  'ATOMIC_SCF_ASYMPTOTIC_CORRECTION': 'NONE',
                  'ATOMIC_SCF_EXPECTED_GRAC_SHIFT': 0.}):
        yield


def test_defaults_reject_actual_grac_before_partition(water, monkeypatch):
    monkeypatch.setattr(api, 'generated_recipe', lambda *a: pytest.fail('partition entered'))
    monkeypatch.setattr(psi4, 'energy', lambda *a, **k: pytest.fail('hidden SCF'))
    water._native_atomic_property_result = object()
    with pytest.raises(ValueError, match='NONE.*GRAC'):
        psi4.oeprop(water, 'ATOMIC_POLARIZABILITIES')
    with pytest.raises(ValueError, match='No native'):
        api.atomic_property_result(water)
    with pytest.raises(ValueError, match='NONE.*GRAC'):
        response.native_response_from_wavefunction(water, caller_converged=True,
            kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75, grid=np.ones((1, 4)))


@pytest.mark.parametrize('shift', [None, 0., -.01, float('nan'), float('inf'), True, '0.06', SHIFT + 1e-8])
def test_expected_shift_is_explicit_and_exact(water, shift):
    with pytest.raises(ValueError, match='shift|mismatch'):
        correction.validate_correction(water, scf_correction='FIXED_GRAC', expected_grac_shift=shift)


@pytest.mark.parametrize('field,value', [('grac_alpha', .6), ('grac_beta', 41.),
                                       ('grac_shift', SHIFT + .01), ('grac_shift', float('nan')),
                                       ('grac_alpha', float('inf')), ('grac_beta', float('nan')),
                                       ('x_alpha', .4), ('x_omega', .2)])
def test_actual_controls_and_underlying_pbe0(water, field, value):
    f = water.functional()
    old = getattr(f, field)()
    try:
        f.set_lock(False)
        getattr(f, 'set_' + field)(value)
        f.set_lock(True)
        with pytest.raises(ValueError, match='mismatch|finite|canonical PBE0'):
            correction.validate_correction(water, **DECLARATION)
        assert _scf_state_signature(water) != water._scf_convergence_evidence[1]
    finally:
        f.set_lock(False)
        getattr(f, 'set_' + field)(old)
        f.set_lock(True)


@pytest.mark.parametrize('getter', ['grac_x_functional', 'grac_c_functional'])
def test_component_mutation_is_in_seal_and_admission(water, getter):
    f = getattr(water.functional(), getter)()
    old = f.alpha()
    before = correction.validate_correction(water, **DECLARATION)
    fingerprint = native._context(water)
    try:
        f.set_alpha(old + .1)  # existing Functional mutator, no new attachment setter
        assert _scf_state_signature(water) != water._scf_convergence_evidence[1]
        with pytest.raises(ValueError, match='stale'):
            correction.require_scf_seal(water)
        with pytest.raises(ValueError, match='correction mismatch'):
            correction.validate_correction(water, **DECLARATION)
        assert before.components != correction.correction_state(water.functional())[3]
        assert native._context(water) != fingerprint
    finally:
        f.set_alpha(old)
    assert correction.validate_correction(water, **DECLARATION) == before


def test_underlying_density_cutoff_mutation_invalidates_seal_and_context(water):
    functional = water.functional()
    component = (*functional.x_functionals(), *functional.c_functionals())[0]
    old = component.density_cutoff()
    before = correction.validate_correction(water, **DECLARATION)
    fingerprint = native._context(water)
    try:
        component.set_density_cutoff(old*2 if old else 1e-12)
        assert _scf_state_signature(water) != water._scf_convergence_evidence[1]
        assert native._context(water) != fingerprint
        with pytest.raises(ValueError, match='stale'):
            correction.require_scf_seal(water)
    finally:
        component.set_density_cutoff(old)
    assert native._context(water) == fingerprint
    assert correction.validate_correction(water, **DECLARATION) == before


def test_component_identity_and_underlying_tweak_helpers(water):
    class Profile:
        def __getattr__(self, key):
            return getattr(water.functional(), key)
        def grac_c_functional(self):
            return core.LibXCFunctional('XC_LDA_C_PW', True)
    with pytest.raises(ValueError, match='correction mismatch'):
        correction.validate_correction(SimpleNamespace(functional=Profile), **DECLARATION)
    changed = core.SuperFunctional.XC_build('XC_HYB_GGA_XC_PBEH', True, {'_beta': .4})
    class Underlying(Profile):
        def grac_c_functional(self):
            return water.functional().grac_c_functional()
        def c_functionals(self):
            return changed.c_functionals()
    with pytest.raises(ValueError, match='canonical PBE0'):
        correction.validate_correction(SimpleNamespace(functional=Underlying), **DECLARATION)


def test_canonical_none_and_owned_tweak_getter():
    f = core.SuperFunctional.XC_build('XC_HYB_GGA_XC_PBEH', True)
    f.set_name('PBE0')
    w = SimpleNamespace(functional=lambda: f)
    assert api._validate_pbe0(w).policy == 'NONE'
    component = f.c_functionals()[0]
    original = component.get_tweak()
    copied = component.get_tweak()
    copied['_beta'] = .4
    assert component.get_tweak() == original
    assert api._validate_pbe0(w).policy == 'NONE'
    with pytest.raises(ValueError, match='NONE'):
        correction.validate_correction(w, expected_grac_shift=SHIFT)


def test_missing_and_stale_seal(water):
    saved = water._scf_convergence_evidence
    try:
        water._scf_convergence_evidence = None
        with pytest.raises(ValueError, match='convergence evidence'):
            correction.validate_correction(water, **DECLARATION)
        water._scf_convergence_evidence = (saved[0], 'stale', saved[2])
        with pytest.raises(ValueError, match='stale'):
            correction.validate_correction(water, **DECLARATION)
    finally:
        water._scf_convergence_evidence = saved


def test_fresh_grac_native_alda_endpoint_and_ownership(water, monkeypatch):
    before = _scf_state_signature(water)
    threads = core.get_num_threads()
    monkeypatch.setattr(psi4, 'energy', lambda *a, **k: pytest.fail('hidden SCF'))
    # Independent small quadrature, not a production/matched integration claim.
    go = core.IsaGridOptions()
    go.radial_points, go.spherical_points = 12, 50
    grid = core.IsaGrid(water.molecule().clone(), go)
    points = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    result = response.native_response_from_wavefunction(water, caller_converged=True,
        kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75, grid=points, **DECLARATION)
    assert np.isfinite(result.at_frequency(.4).raw_coupled).all()
    np.testing.assert_array_equal(np.asarray(result.provider.orbitals()), np.asarray(water.Ca()))
    np.testing.assert_array_equal(np.asarray(result.provider.energies()), np.asarray(water.epsilon_a()))
    owned = result.correction_provenance
    assert owned.policy == 'FIXED_GRAC' and owned.shift == SHIFT
    assert 'no GRAC kernel derivative' in owned.response_description
    with pytest.raises(FrozenInstanceError):
        owned.shift = .1
    assert _scf_state_signature(water) == before
    assert core.get_num_threads() == threads
    # The policy declaration is independent of ambient SCF DFT settings.
    with options({('SCF', 'DFT_GRAC_SHIFT'): .2,
                  'ATOMIC_SCF_ASYMPTOTIC_CORRECTION': 'FIXED_GRAC',
                  'ATOMIC_SCF_EXPECTED_GRAC_SHIFT': SHIFT}):
        assert api.validate_request(water, ('ATOMIC_POLARIZABILITIES',)) < 2e-7
        assert correction.validate_correction(water, **DECLARATION) == owned
    # Reuse must reject even a factory-shaped context with mismatched correction
    # provenance, before partitioning. No second native response construction.
    context = native.NativeContext(replace(result, correction_provenance=replace(owned, shift=SHIFT+.01)),
        native._context(water), native._policy('alda_slater_pw92', .25, .75, points, 1e-10, owned))
    recipe = api.generated_recipe(water, 12, 50)
    monkeypatch.setattr(native, 'native_partition', lambda *a, **k: pytest.fail('partition entered'))
    with pytest.raises(ValueError, match='context mismatch'):
        native.native_properties(water, recipe, bonds=((0, 1), (0, 2)), frames=None,
            caller_converged=True, kernel='alda_slater_pw92', exact_exchange=.25,
            local_scale=.75, response_grid=points, response_context=context, **DECLARATION)
    assert native._policy('alda_slater_pw92', .25, .75, points, 1e-10, owned) != native._policy(
        'alda_slater_pw92', .25, .75, points, 1e-10, replace(owned, shift=SHIFT+.01))


def test_public_none_return_owned_result_and_invalidation(water, monkeypatch):
    # Only orchestration is doubled; real corrected SCF admission/seal is exercised.
    partition = SimpleNamespace(require_q=lambda: None,
        trajectory=SimpleNamespace(state=SimpleNamespace(iteration=1)),
        drho=SimpleNamespace(fitted_electrons=10.))
    monkeypatch.setattr(api.p, 'native_partition', lambda *a, **k: partition)
    monkeypatch.setattr(psi4, 'energy', lambda *a, **k: pytest.fail('hidden SCF'))
    with options({'ATOMIC_SCF_ASYMPTOTIC_CORRECTION': 'FIXED_GRAC',
                  'ATOMIC_SCF_EXPECTED_GRAC_SHIFT': SHIFT}):
        assert psi4.oeprop(water, 'ATOMIC_PARTITION') is None
        owned = api.atomic_property_result(water)
        assert owned.correction_provenance.shift == SHIFT
        core.set_global_option('ATOMIC_SCF_EXPECTED_GRAC_SHIFT', SHIFT + .01)
        with pytest.raises(ValueError, match='correction mismatch'):
            psi4.oeprop(water, 'ATOMIC_PARTITION')
        assert owned.correction_provenance.shift == SHIFT
        with pytest.raises(ValueError, match='No native'):
            api.atomic_property_result(water)
