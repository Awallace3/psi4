"""Source-loaded expert native partition; no staged/source-package shadowing."""
import importlib.util
from pathlib import Path
import sys
from dataclasses import replace, FrozenInstanceError
import numpy as np
import pytest
import psi4
from psi4 import core

_PATH = Path(__file__).resolve().parents[2] / 'psi4/driver/procrouting/isapol_native_partition.py'
_SPEC = importlib.util.spec_from_file_location('_test_owned_native_isa_partition', _PATH)
native = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = native
_SPEC.loader.exec_module(native)


def controls(**kw):
    c = native.ControllerRecipe(convergence=1e-9, max_iterations=120,
        w_eps=.17, positive_lambda=.001, positive_max_alpha=.2, positive_auto=True,
        damping=0., s_block_only=True, density_cutoff=1e-36, w_eps_activation=1e-5,
        positive_activation=1e-5, tail_activation=1e-5, mixing=0., mixing_skip=20,
        tail_iteration_limit=20, fix_tails=True)
    return replace(c, **kw)


def basis(name, centre, exponents, role_rep='Spherical'):
    return native.BasisRecipe(name, 'analytic declared normalized primitive Gaussian basis; not CamCASP preset',
        role_rep, (centre,), tuple(native.ShellRecipe(0, 0, (a,), ((2*a/np.pi)**.75,)) for a in exponents))


def recipe(wfn, **kw):
    mol = wfn.molecule()
    centres = tuple((mol.x(i), mol.y(i), mol.z(i)) for i in range(mol.natom()))
    # Physically explicit even-tempered Gaussian fitting spaces, not JKFIT aliases.
    aux = native.BasisRecipe('declared molecular even-tempered s Gaussian AUX',
        'analytic recipe (.25,.5,1,2,4,8,16,32); normalized primitives, Cartesian', 'Cartesian', centres,
        tuple(native.ShellRecipe(i, 0, (a,), ((2*a/np.pi)**.75,))
              for i in range(len(centres)) for a in (.25,.5,1.,2.,4.,8.,16.,32.)))
    sites = []
    for i, c in enumerate(centres):
        atomic = basis('distinct primitive AtomAux', c, (.3,.7,1.5,3.5,9.,24.))
        shape = replace(atomic, name='matching s Shape')
        sites.append(native.SiteRecipe(f'{mol.symbol(i)}{i}', c, atomic, shape, tuple(range(6)), 2, 1.5, True))
    return native.PartitionRecipe('small explicit Gaussian Drho-C/ISA-A', 'test-authored basis definition only',
        'explicit_cartesian_drho_c_isa_a', aux, tuple(sites),
        native.GridRecipe(100, 110, 3, 1., 'native_tabulated_bragg_slater',
                          'all_sites_unscreened_full_molecular_grid'), controls(**kw), 'Drho1e-2')


@pytest.fixture(scope='module')
def helium():
    psi4.core.be_quiet()
    mol = psi4.geometry('''0 1
He .17 -.23 .31
units bohr
symmetry c1
no_com
no_reorient
''')
    psi4.set_options({'basis':'sto-3g', 'reference':'rhf', 'scf_type':'pk',
                      'e_convergence':1e-12, 'd_convergence':1e-12})
    _, wfn = psi4.energy('hf', molecule=mol, return_wfn=True)
    return wfn


def test_actual_scf_drho_controller_q(helium):
    r = native.native_partition(helium, recipe(helium), caller_converged=True)
    assert r.converged and r.require_q() is r.q
    assert r.trajectory.termination == 'converged'
    assert r.drho.charge_penalty == 1000.
    assert r.comparison_status == 'not_evaluated_no_reference'
    np.testing.assert_array_equal(r.density.evaluate(r.grid_points.tolist(), [0]), r.density_samples)
    J, A = np.asarray(r.drho.coulomb_metric), np.asarray(r.drho.metric)
    q = np.array(r.drho.charges)
    np.testing.assert_allclose(A, J+1000*np.outer(q,q), atol=1e-11, rtol=1e-15)
    np.testing.assert_allclose(r.drho.rhs, np.array(r.drho.raw_rhs)+2000*q, atol=1e-11, rtol=1e-15)
    assert r.drho.relative_residual < 1e-12
    # One site: Q00 is the unpartitioned AUX integral apart from the explicit cutoff.
    np.testing.assert_allclose(np.asarray(r.q.values)[0], q, atol=2e-7, rtol=2e-7)
    assert r.q.labels == [f'{helium.molecule().symbol(0)}0'] and r.q.ranks == [2]
    assert np.isfinite(r.q.values).all()
    with pytest.raises(ValueError):
        r.main.occupied.setflags(write=True)


def test_nonconvergence_has_no_q(helium):
    r = native.native_partition(helium, recipe(helium, max_iterations=1), caller_converged=True)
    assert not r.converged and r.q is None and r.trajectory.history
    with pytest.raises(RuntimeError, match='did not converge'):
        r.require_q()


def test_collocation_accuracy_failure_is_not_repaired(helium, monkeypatch):
    original = native._psi_samples
    def broken(basis, points):
        samples = original(basis, points)
        if len(points) == 213:  # independent shell validation set, never training
            samples[0, 0] += .01
        return samples
    monkeypatch.setattr(native, '_psi_samples', broken)
    with pytest.raises(ValueError, match='validation failure'):
        native.adapt_main(helium, caller_converged=True)


def test_declaration_and_geometry_fail_closed(helium):
    with pytest.raises(ValueError, match='caller_converged'):
        native.adapt_main(helium, caller_converged=False)
    r = recipe(helium)
    with pytest.raises(ValueError, match='modern preset'):
        replace(r, track='modern_spherical_aux')
    with pytest.raises(FrozenInstanceError):
        r.name = 'mutated'
    with pytest.raises(ValueError, match='exactly match'):
        native.native_partition(helium, replace(r, auxiliary=replace(r.auxiliary, centres=((0.,0.,0.),))),
                                caller_converged=True)


@pytest.mark.parametrize('change', [{'w_eps':np.nan}, {'convergence':0.}, {'max_iterations':0},
                                   {'mixing':1.1}, {'fix_tails':1}, {'density_cutoff':-1.}])
def test_declared_controller_rejects_invalid_values(change):
    with pytest.raises(ValueError):
        controls(**change)


def test_recipe_rejects_contracted_atomaux_and_wrong_map():
    c = (0.,0.,0.)
    a = basis('primitive', c, (.5,1.))
    contracted = replace(a, shells=(native.ShellRecipe(0,0,(.5,1.),(.7,-.1)),))
    with pytest.raises(ValueError, match='primitive'):
        native.SiteRecipe('X', c, contracted, contracted, (0,), 0, 1.5, True)
    with pytest.raises(ValueError):
        native.SiteRecipe('X', c, a, a, (1,0), 0, 1.5, True)


def test_initialization_absolute_not_log_and_first_tie():
    c = (0.,0.,0.)
    # Absolute chooses .2 rather than 2.; logarithmic would choose 2.
    a = basis('AtomAux', c, (.2, 2.))
    s = native.SiteRecipe('X', c, a, a, (0,1), 0, 1.5, True)
    initial = native.one_gto_initialization((s,))
    assert initial.shape_coefficients == [[1.,0.]]
    assert initial.atomic_coefficients == [[0.,0.]]
    a = basis('AtomAux', c, (.5,1.5))
    s = replace(s, atomic=a, shape=a)
    assert native.one_gto_initialization((s,)).shape_coefficients == [[1.,0.]]


def test_final_tail_samples_use_stored_and_eligibility():
    c = (0.,0.,0.)
    a = basis('AtomAux', c, (1.,))
    s = native.SiteRecipe('X', c, a, a, (0,), 0, 1.5, True)
    state = core.IsaAControllerState()
    coeff = core.IsaSweepState()
    coeff.atomic_coefficients = coeff.shape_coefficients = [[-1.]]
    state.coefficients = coeff
    tail = core.IsaExponentialTail()
    tail.defined, tail.amplitude, tail.exponent, tail.cutoff = True, -3., 2., 1.5
    state.tails, state.apply_tails = [tail], True
    points = [[0.,0.,0.], [0.,0.,2.]]
    x = native.final_shape_samples([a.build('Shape')], state, (s,), points)[0]
    assert x[0] < 0
    assert x[1] == pytest.approx(-3*np.exp(-4))
    y = native.final_shape_samples([a.build('Shape')], state, (replace(s, tail_allowed=False),), points)[0]
    np.testing.assert_array_equal(y, [0.,0.])


@pytest.mark.parametrize('l', range(5))
def test_independent_overlap_contracted_signed_shells(l):
    c = ((.3,-.7,1.1),)
    sh = native.ShellRecipe(0, l, (.4,1.7), (.7,-.13))
    b = native.BasisRecipe('signed contracted', 'analytic test definition', 'Spherical', c, (sh,))
    np.testing.assert_allclose(native.explicit_main_overlap(b), b.build('AtomAux').overlap(), atol=2e-13, rtol=2e-13)


def test_actual_signed_contracted_spdfg_shell_adaptation():
    psi4.basis_helper('assign isa_test_signed\n[isa_test_signed]\nspherical\n****\nHe 0\n' +
        '\n'.join(f'{l} 2 1.0\n 1.7 .7\n .4 -.13' for l in 'SPDFG') + '\n****\n',
        name='ISA_PARTITION_SIGNED_MAIN')
    mol = psi4.geometry('''0 1
He -.83 .61 -.27
units bohr
symmetry c1
no_com
no_reorient
''')
    psi4.set_options({'basis':'ISA_PARTITION_SIGNED_MAIN', 'puream':True, 'reference':'rhf',
                      'scf_type':'pk', 'e_convergence':1e-12, 'd_convergence':1e-12})
    _, wfn = psi4.energy('hf', molecule=mol, return_wfn=True)
    a = native.adapt_main(wfn, caller_converged=True)
    assert {s.l for s in a.recipe.shells} == set(range(5))
    assert all(min(s.coefficients) < 0 for s in a.recipe.shells)
    assert a.global_overlap_residual < 2e-11


def test_multicentre_main_and_honest_unconverged_small_h2_recipe():
    mol = psi4.geometry('''0 1
H .21 -.37 .59
H 1.13 .22 1.48
units bohr
symmetry c1
no_com
no_reorient
''')
    psi4.set_options({'basis':'cc-pvdz', 'puream':True, 'reference':'rhf', 'scf_type':'pk',
                      'e_convergence':1e-12, 'd_convergence':1e-12})
    _, wfn = psi4.energy('hf', molecule=mol, return_wfn=True)
    a = native.adapt_main(wfn, caller_converged=True)
    assert a.global_overlap_residual < 2e-11
    assert a.transformed_orthonormality_residual < 2e-9
    result = native.native_partition(wfn, recipe(wfn), caller_converged=True)
    # This small s-only basis recipe FAILED the original 120-iteration forward
    # attempt (native-isa-partition-tests-v3.log). This is now explicitly a
    # negative Q-gating test, NOT a recovered molecular convergence assertion.
    # No thresholds, iteration budget, tails or basis values were changed.
    assert not result.converged and result.q is None
    assert result.trajectory.termination == 'max_iterations'
    assert result.trajectory.state.iteration == 120
    assert result.trajectory.state.max_delta >= result.recipe.controller.convergence
    assert result.grid.natom() == 2
    with pytest.raises(RuntimeError, match='did not converge'):
        result.require_q()


@pytest.mark.parametrize('basis_name', ['cc-pvdz', 'cc-pvtz', 'cc-pvqz', 'cc-pv5z'])
def test_real_main_spd_and_higher_transformation(basis_name):
    # Genuine SCF, signed contractions included in standard orbital bases.
    mol = psi4.geometry('''0 1
He .31 -.47 .83
units bohr
symmetry c1
no_com
no_reorient
''')
    psi4.set_options({'basis':basis_name, 'puream':True, 'reference':'rhf', 'scf_type':'pk',
                      'e_convergence':1e-12, 'd_convergence':1e-12})
    _, wfn = psi4.energy('hf', molecule=mol, return_wfn=True)
    a = native.adapt_main(wfn, caller_converged=True)
    assert max(d.heldout_residual for d in a.diagnostics) < 2e-11
    assert a.global_overlap_residual < 2e-11
    assert a.transformed_orthonormality_residual < 2e-9
    assert max(d.rank for d in a.diagnostics) == {'cc-pvdz':1, 'cc-pvtz':2, 'cc-pvqz':3, 'cc-pv5z':4}[basis_name]
