"""Small real SCFs for native admission; never run the partition/response pipeline."""
from contextlib import contextmanager
import numpy as np
import pytest
import psi4
from psi4.driver import p4util
from psi4.driver.procrouting import isapol_oeprop as api


@contextmanager
def option_state(options):
    saved = p4util.OptionsState(*options)
    try:
        yield
    finally:
        saved.restore()


@pytest.fixture(scope='module')
def scf_cases():
    options = dict(reference='rks', scf_type='pk', guess='core', maxiter=80,
                   fail_on_maxiter=True, e_convergence=1e-10, d_convergence=1e-10,
                   dft_alpha=.25, orbital_optimizer_package='internal',
                   screening='schwarz', df_scf_guess=False)
    with option_state([['BASIS'], *(['SCF', k.upper()] for k in options)]):
        psi4.core.set_global_option('BASIS', 'sto-3g')
        for key, value in options.items():
            psi4.core.set_local_option('SCF', key.upper(), value)
        mol = psi4.geometry('O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1')
        _, converged = psi4.energy('pbe0', molecule=mol, return_wfn=True)
        psi4.core.set_local_option('SCF', 'MAXITER', 1)
        psi4.core.set_local_option('SCF', 'FAIL_ON_MAXITER', False)
        _, unfinished = psi4.energy('pbe0', molecule=mol, return_wfn=True)
        psi4.core.set_local_option('SCF', 'MAXITER', 80)
        psi4.core.set_local_option('SCF', 'FAIL_ON_MAXITER', True)
        psi4.core.set_local_option('SCF', 'DFT_ALPHA', .40)
        _, modified = psi4.energy('pbe0', molecule=mol, return_wfn=True)
    return converged, unfinished, modified


@pytest.fixture(autouse=True)
def native_policies():
    options = {'PARTITION_SCHEME': 'ISA_A', 'ATOMIC_RESPONSE_LOCALIZATION': 'LW',
               'ATOMIC_PROPERTY_RECIPE': 'GENERATED_JKFIT_ISA_A'}
    with option_state([[k] for k in options]):
        for key, value in options.items():
            psi4.core.set_global_option(key, value)
        yield


def test_actual_convergence_acceptance_and_stopping_diagnostics(scf_cases, monkeypatch):
    w, _, _ = scf_cases
    diagnostics, _, _ = w._scf_convergence_evidence
    iteration, de, dn, etol, dtol, rms = diagnostics
    assert iteration > 0 and abs(de) < etol and 0 <= dn < dtol
    assert etol == dtol == 1e-10
    assert isinstance(rms, bool)
    # Changing ambient functional/convergence options cannot redefine this wfn.
    with option_state([['SCF', 'DFT_ALPHA'], ['SCF', 'E_CONVERGENCE']]):
        psi4.core.set_local_option('SCF', 'DFT_ALPHA', .60)
        psi4.core.set_local_option('SCF', 'E_CONVERGENCE', 1e-3)
        assert api.validate_request(w, ('ATOMIC_POLARIZABILITIES',)) < 2e-7
    class Admitted(Exception):
        pass
    def stop(*a, **k):
        raise Admitted
    monkeypatch.setattr(api, 'generated_recipe', stop)
    monkeypatch.setattr(psi4, 'energy', lambda *a, **k: pytest.fail('hidden SCF'))
    with pytest.raises(Admitted):
        psi4.oeprop(w, 'ATOMIC_PARTITION')


def test_real_nonfatal_maxiter_rejected_before_partition(scf_cases, monkeypatch):
    _, w, _ = scf_cases
    assert np.isfinite(w.energy()) and w.energy() != 0
    s, d, f = map(np.asarray, (w.S(), w.Da(), w.Fa()))
    # This is precisely the misleading former admission criterion.
    assert np.max(np.abs(f @ d @ s - s @ d @ f)) < 2e-7
    assert w._scf_convergence_evidence is None
    assert w._scf_stopping_diagnostics is None
    monkeypatch.setattr(api, 'generated_recipe', lambda *a: pytest.fail('partition entered'))
    monkeypatch.setattr(psi4, 'energy', lambda *a, **k: pytest.fail('hidden SCF'))
    with pytest.raises(ValueError, match='convergence evidence'):
        psi4.oeprop(w, 'ATOMIC_PARTITION')


def test_real_dft_alpha_override_rejected_before_partition(scf_cases, monkeypatch):
    _, _, w = scf_cases
    assert w.functional().name().upper() == 'PBE0'
    assert w.functional().x_alpha() == .4
    assert w._scf_convergence_evidence is not None
    monkeypatch.setattr(api, 'generated_recipe', lambda *a: pytest.fail('partition entered'))
    with pytest.raises(ValueError, match='unmodified canonical PBE0'):
        psi4.oeprop(w, 'ATOMIC_POLARIZABILITIES')


@pytest.mark.parametrize('field', ['Ca', 'Da', 'Fa', 'epsilon_a'])
def test_changed_state_rejects_stale_convergence(scf_cases, field):
    w, _, _ = scf_cases
    array = np.asarray(getattr(w, field)())
    original = array.copy()
    try:
        array.flat[0] += .01
        with pytest.raises(ValueError, match='stale'):
            psi4.oeprop(w, 'ATOMIC_PARTITION')
    finally:
        array[...] = original
    assert api.validate_request(w, ('ATOMIC_PARTITION',)) < 2e-7


def test_external_wavefunction_has_no_inferred_seal(scf_cases):
    w, _, _ = scf_cases
    evidence = w._scf_convergence_evidence
    try:
        del w._scf_convergence_evidence
        with pytest.raises(ValueError, match='convergence evidence'):
            psi4.oeprop(w, 'ATOMIC_PARTITION')
    finally:
        w._scf_convergence_evidence = evidence


@pytest.mark.parametrize('change', ['scale', 'omega', 'grac', 'tweak', 'identity'])
def test_effective_pbe0_definition_not_just_name(change):
    f = psi4.core.SuperFunctional.XC_build('XC_HYB_GGA_XC_PBEH', True)
    f.set_name('PBE0')
    if change == 'scale':
        f.c_functionals()[0].set_alpha(.9)
    elif change == 'omega':
        f.set_x_omega(.2)
    elif change == 'grac':
        f.set_grac_alpha(f.grac_alpha() + .1)
    elif change == 'tweak':
        # XC_build copies HF metadata before applying tweaks: x_alpha remains .25.
        f = psi4.core.SuperFunctional.XC_build('XC_HYB_GGA_XC_PBEH', True, {'_beta': .4})
        f.set_name('PBE0')
        assert f.x_alpha() == .25
    else:
        f = psi4.core.SuperFunctional.XC_build('XC_HYB_GGA_XC_B3LYP', True)
        f.set_name('PBE0')
        f.set_x_alpha(.25)
    class Wfn:
        def functional(self):
            return f
    with pytest.raises(ValueError, match='PBE0'):
        api._validate_pbe0(Wfn())
