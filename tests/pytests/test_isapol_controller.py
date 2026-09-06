"""Explicit-input ordinary-A controller tests; no production fixed-point parity claim."""
import numpy as np
import pytest
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def fixture(options=None):
    shells = []
    for a in [.4, 1.1]:
        s = core.IsaGaussianShell()
        s.exponents, s.coefficients = [a], [1.]
        shells.append(s)
    def basis(role):
        return core.IsaExplicitBasis(role, core.IsaBasisRepresentation.Spherical, [[0, 0, 0]], shells)
    atomic, shape = basis(core.IsaBasisRole.AtomAux), basis(core.IsaBasisRole.Shape)
    density = core.IsaFixedDensity(basis(core.IsaBasisRole.MolecularAux), [1.5, .7])
    x, w = np.polynomial.legendre.leggauss(160)
    r = 7.5*(x+1)
    grid = core.IsaNoTailGrid()
    grid.points = np.column_stack([r, np.zeros_like(r), np.zeros_like(r)]).tolist()
    grid.weights = (7.5*w*4*np.pi*r*r).tolist()
    grid.density_sites, grid.shape_sites = [0], [0]
    state = core.IsaSweepState()
    state.atomic_coefficients, state.shape_coefficients = [[.2, 2.]], [[.2, 2.]]
    if options is None:
        options = core.IsaAControllerOptions()
        options.fix_tails = False
    controller = core.IsaAController([atomic], [shape], [[0, 1]], density, [grid], options)
    return controller, state, shape, grid


def test_controller_fixed_density_convergence_and_restart():
    controller, state, _, _ = fixture()
    initial = controller.initialize(state)
    result = controller.run(initial)
    assert result.termination == 'converged' and result.state.converged
    assert result.state.iteration == 2
    np.testing.assert_allclose(result.state.coefficients.atomic_coefficients, [[1.5, .7]], atol=2e-13)
    one = controller.step(initial)
    assert not one.next.converged
    resumed = controller.run(one.next)
    np.testing.assert_array_equal(resumed.state.coefficients.atomic_coefficients, result.state.coefficients.atomic_coefficients)
    assert initial.iteration == 0 and state.shape_coefficients == [[.2, 2.]]
    assert resumed.state.iteration == result.state.iteration
    with pytest.raises(ValueError, match='already converged'):
        controller.step(result.state)


def test_controller_premixing_diagnostics_and_old_shape_tail_lag():
    options = core.IsaAControllerOptions()
    options.mixing, options.mixing_skip = .3, 0
    options.tail_cutoffs = [1.5]
    options.fit.w_eps, options.fit.positive_lambda = .17, .001
    options.w_eps_activation = options.positive_activation = options.tail_activation = 1.
    controller, state, shape, _ = fixture(options)
    initial = controller.initialize(state)
    assert initial.active_w_eps == initial.active_positive_lambda == 0
    assert not initial.apply_tails
    step = controller.step(initial)
    raw = np.array(step.raw_sweep.next.shape_coefficients)
    want = .7*raw+.3*np.array(state.shape_coefficients)
    np.testing.assert_allclose(step.next.coefficients.shape_coefficients, want, atol=1e-15)
    np.testing.assert_array_equal(step.next.coefficients.atomic_coefficients, step.raw_sweep.next.atomic_coefficients)
    assert step.shape_charges == step.next.saved_shape_charges
    raw_charge = core.IsaGaussianShape(shape, raw[0].tolist()).exterior_charge(0)
    mixed_charge = core.IsaGaussianShape(shape, want[0].tolist()).exterior_charge(0)
    assert step.shape_charges[0] == raw_charge and abs(raw_charge-mixed_charge) > 1
    expected_tail = core.IsaGaussianShape(shape, state.shape_coefficients[0]).fit_tail(1.5)
    assert step.next.tails[0].amplitude == expected_tail.tail.amplitude
    assert step.next.tails[0].exponent == expected_tail.tail.exponent
    assert step.next.active_w_eps == .17 and step.next.active_positive_lambda == .001
    assert step.next.apply_tails


def test_controller_activation_not_latched_and_tail_iteration_strictness():
    options = core.IsaAControllerOptions()
    options.fit.w_eps, options.fit.positive_lambda = .17, .001
    options.w_eps_activation = options.positive_activation = options.tail_activation = 1e-30
    options.tail_cutoffs = [1.5]
    options.tail_iteration_limit = 20
    controller, state, _, _ = fixture(options)
    active = controller.initialize(state)
    active.active_w_eps, active.active_positive_lambda = .17, .001
    active.iteration = 19
    at20 = controller.step(active)
    assert at20.next.max_delta > 1e-30
    assert at20.next.active_w_eps == at20.next.active_positive_lambda == 0
    assert not at20.next.apply_tails
    active.iteration = 20
    at21 = controller.step(active)
    assert at21.next.apply_tails


def test_controller_strict_convergence_and_no_forced_activation_sweep():
    controller, state, _, _ = fixture()
    delta = controller.step(controller.initialize(state)).deltas[0]
    options = core.IsaAControllerOptions()
    options.fix_tails = False
    options.convergence = delta
    same, state, _, _ = fixture(options)
    assert not same.step(same.initialize(state)).next.converged  # strict <, not <=
    options.convergence = delta*1.01
    options.fit.w_eps = .17
    options.w_eps_activation = 1.
    loose, state, _, _ = fixture(options)
    result = loose.run(loose.initialize(state))
    assert result.state.iteration == 1 and result.state.converged
    assert result.state.active_w_eps == .17  # activated next control does not force another sweep


def test_controller_max_iterations_and_mixing_skip():
    options = core.IsaAControllerOptions()
    options.fix_tails, options.max_iterations = False, 1
    options.mixing, options.mixing_skip = .5, 1
    controller, state, _, _ = fixture(options)
    result = controller.run(controller.initialize(state))
    assert result.termination == 'max_iterations' and not result.state.converged
    np.testing.assert_array_equal(result.state.coefficients.shape_coefficients,
                                  result.history[0].raw_sweep.next.shape_coefficients)
    with pytest.raises(ValueError, match='maximum'):
        controller.step(result.state)


@pytest.mark.parametrize('field,value', [('convergence', 0.), ('convergence', np.nan),
    ('mixing', -1.), ('mixing', 1.1), ('max_iterations', 0), ('mixing_skip', -1),
    ('tail_iteration_limit', -1), ('tail_cutoffs', []), ('tail_cutoffs', [0.]),
    ('tail_allowed', [True, False]), ('convergence_included', [False])])
def test_controller_rejects_options(field, value):
    options = core.IsaAControllerOptions()
    options.tail_cutoffs = [1.5]
    setattr(options, field, value)
    with pytest.raises(ValueError, match='Controller'):
        fixture(options)


@pytest.mark.parametrize('field,value', [('iteration', -1), ('active_w_eps', .33),
    ('active_positive_lambda', np.nan), ('max_delta', np.inf), ('tails', []),
    ('saved_shape_charges', [np.nan])])
def test_controller_rejects_restart_state(field, value):
    controller, state, _, _ = fixture()
    initial = controller.initialize(state)
    setattr(initial, field, value)
    with pytest.raises(ValueError, match='Controller|controller|Invalid'):
        controller.run(initial)


def test_controller_excluded_site_still_fits_but_does_not_block_convergence():
    # Synthetic coincident sites: a two-dimensional changing shape and a 1D shape
    # whose W angle is identically zero. Mimics explicit dummy-site exclusion.
    _, _, shape2, grid = fixture()
    shells = []
    for a in [.4, 1.1]:
        s = core.IsaGaussianShell()
        s.exponents, s.coefficients = [a], [1.]
        shells.append(s)
    def basis(role, selected):
        return core.IsaExplicitBasis(role, core.IsaBasisRepresentation.Spherical, [[0, 0, 0]], selected)
    atomic = [basis(core.IsaBasisRole.AtomAux, shells), basis(core.IsaBasisRole.AtomAux, shells[:1])]
    shapes = [shape2, basis(core.IsaBasisRole.Shape, shells[:1])]
    density = core.IsaFixedDensity(basis(core.IsaBasisRole.MolecularAux, shells), [1.5, .7])
    grid.shape_sites = [0, 1]
    options = core.IsaAControllerOptions()
    options.fix_tails = False
    options.convergence_included = [False, True]
    controller = core.IsaAController(atomic, shapes, [[0, 1], [0]], density, [grid, grid], options)
    state = core.IsaSweepState()
    state.atomic_coefficients = state.shape_coefficients = [[.2, 2.], [1.]]
    step = controller.step(controller.initialize(state))
    assert len(step.raw_sweep.fits) == 2
    assert step.atom_converged == [False, True]
    assert step.next.converged and step.next.max_delta < 1e-14
