"""Requires rebuilt core getters. One fresh small genuine fixed-GRAC water SCF.

No reference tables, orbital repair, inferred shift, or GRAC response derivative.
"""
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.p4util import OptionsState
from psi4.driver.procrouting import isapol_native_correction as correction
from psi4.driver.procrouting import isapol_native as native
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


def test_default_policy_rejects_actual_grac(water):
    with pytest.raises(ValueError, match='NONE.*GRAC'):
        correction.validate_correction(water)


def test_correction_options_are_closed():
    """Exercise the registered option validator rather than its C++ spelling."""
    key = 'ATOMIC_SCF_ASYMPTOTIC_CORRECTION'
    with options({key: 'NONE'}):
        assert core.get_global_option(key) == 'NONE'
        core.set_global_option(key, 'FIXED_GRAC')
        assert core.get_global_option(key) == 'FIXED_GRAC'
        with pytest.raises(RuntimeError):
            core.set_global_option(key, 'DECLARED_MULTPOLE_AC')
        with pytest.raises(RuntimeError):
            core.set_global_option('ATOMIC_AC_JOIN', 'TANH')


def test_correction_admission_does_not_mutate_scf(water, monkeypatch):
    before = _scf_state_signature(water)
    def forbidden(*args, **kwargs):
        pytest.fail('correction admission must not run SCF or set options')
    monkeypatch.setattr(psi4, 'energy', forbidden)
    monkeypatch.setattr(psi4, 'set_options', forbidden)
    correction.validate_correction(water, **DECLARATION)
    assert _scf_state_signature(water) == before


@pytest.mark.parametrize('shift', [None, 0., float('nan'), True, '0.06', SHIFT + 1e-8])
def test_expected_shift_is_explicit_and_exact(water, shift):
    with pytest.raises(ValueError, match='shift|mismatch'):
        correction.validate_correction(water, scf_correction='FIXED_GRAC', expected_grac_shift=shift)


@pytest.mark.parametrize('field,value', [('grac_alpha', .6), ('grac_beta', 41.),
                                       ('grac_shift', SHIFT + .01), ('grac_shift', float('nan')),
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
    assert correction.validate_correction(w, require_canonical=True).policy == 'NONE'
    component = f.c_functionals()[0]
    original = component.get_tweak()
    copied = component.get_tweak()
    copied['_beta'] = .4
    assert component.get_tweak() == original
    assert correction.validate_correction(w, require_canonical=True).policy == 'NONE'
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


def test_fixed_grac_provenance_is_owned_and_ignores_ambient_options(water, monkeypatch):
    before = _scf_state_signature(water)
    monkeypatch.setattr(psi4, 'energy', lambda *a, **k: pytest.fail('hidden SCF'))
    owned = correction.validate_correction(water, **DECLARATION)
    assert owned.policy == 'FIXED_GRAC' and owned.shift == SHIFT
    assert 'no GRAC kernel derivative' in owned.response_description
    with pytest.raises(FrozenInstanceError):
        owned.shift = .1
    assert _scf_state_signature(water) == before
    # The policy declaration is independent of ambient SCF DFT settings.
    with options({('SCF', 'DFT_GRAC_SHIFT'): .2,
                  'ATOMIC_SCF_ASYMPTOTIC_CORRECTION': 'FIXED_GRAC',
                  'ATOMIC_SCF_EXPECTED_GRAC_SHIFT': SHIFT}):
        assert correction.validate_correction(water, **DECLARATION) == owned
