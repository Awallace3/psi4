"""Synthetic synchronous/no-tail sweep checks, not production controller parity."""
import numpy as np
import pytest
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


def fixture():
    centres = [[-1., 0., 0.], [1., 0., 0.]]
    shells = []
    atomic, shape = [], []
    for site in range(2):
        s = core.IsaGaussianShell()
        s.exponents, s.coefficients = [.7], [1.]
        atomic.append(core.IsaExplicitBasis(core.IsaBasisRole.AtomAux, core.IsaBasisRepresentation.Spherical,
                                            [centres[site]], [s]))
        shape.append(core.IsaExplicitBasis(core.IsaBasisRole.Shape, core.IsaBasisRepresentation.Spherical,
                                           [centres[site]], [s]))
        s.centre = site
        shells.append(s)
    molecular = core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux, core.IsaBasisRepresentation.Spherical,
                                      centres, shells)
    density = core.IsaFixedDensity(molecular, [1.5, .6])
    state = core.IsaSweepState()
    state.atomic_coefficients, state.shape_coefficients = [[-.3], [.4]], [[.8], [1.4]]
    grids = []
    rng = np.random.default_rng(953)
    for a in range(2):
        g = core.IsaNoTailGrid()
        g.points = (rng.normal(size=(47, 3))+centres[a]).tolist()
        g.weights = rng.uniform(.01, .04, 47).tolist()
        g.shape_sites, g.density_sites = [0, 1], [0, 1]
        grids.append(g)
    options = core.IsaAFitOptions()
    options.w_eps, options.damping = .17, .1
    options.positive_lambda, options.positive_max_alpha = .001, 1.
    return atomic, shape, density, state, grids, options


def reference(state, grids, options):
    """Direct s-Gaussian formulas and scalar solve, independent of C++ providers."""
    centres = np.array([[-1., 0., 0.], [1., 0., 0.]])
    out = []
    for a, grid in enumerate(grids):
        r2 = np.sum((np.asarray(grid.points)[:, None, :]-centres[None, :, :])**2, axis=2)
        values = np.exp(-.7*r2)
        shapes = np.maximum(values*np.array(state.shape_coefficients).ravel(), 0)
        total = shapes[:, grid.shape_sites].sum(axis=1)
        rho = values[:, grid.density_sites] @ np.array([1.5, .6])[grid.density_sites]
        partition = np.divide(rho*shapes[:, a], total, out=np.zeros(len(rho)), where=np.abs(total)>options.density_cutoff)
        w = np.asarray(grid.weights)
        rhs = np.sum(w*values[:, a]*(partition+options.damping*shapes[:, a])*np.exp(options.w_eps*r2[:, a]))
        metric = (np.pi/(1.4-options.w_eps))**1.5*(1+options.damping)
        if state.atomic_coefficients[a][0] < 0:
            metric += options.positive_lambda
        out.append((rhs/metric, rhs, metric, np.sum(w*partition)))
    return np.array(out)


def test_no_tail_sweep_uses_frozen_old_state():
    atomic, shape, density, state, grids, options = fixture()
    sweep = core.IsaNoTailSweep(atomic, shape, [[0], [0]], density)
    result = sweep.run(state, grids, options)
    want = reference(state, grids, options)
    np.testing.assert_allclose(result.next.atomic_coefficients, want[:, :1], atol=3e-16, rtol=3e-15)
    np.testing.assert_array_equal(result.next.atomic_coefficients, result.next.shape_coefficients)
    for a, fit in enumerate(result.fits):
        np.testing.assert_allclose([fit.rhs.np[0, 0], fit.metric.np[0, 0], fit.population], want[a, 1:], atol=2e-15)
        assert fit.relative_residual < 1e-14
    assert state.shape_coefficients == [[.8], [1.4]]
    assert result.clipped_shape_points == [0, 0]
    # This case detects accidentally feeding the freshly fitted atom 0 into atom 1.
    async_state = core.IsaSweepState()
    async_state.atomic_coefficients = state.atomic_coefficients
    async_state.shape_coefficients = [[want[0, 0]], [1.4]]
    assert abs(reference(async_state, grids, options)[1, 0]-want[1, 0]) > 1e-3
    next_copy = result.next
    next_copy.shape_coefficients = [[99.], [99.]]
    assert result.next.shape_coefficients != next_copy.shape_coefficients
    np.testing.assert_array_equal(sweep.run(state, grids, options).next.atomic_coefficients, result.next.atomic_coefficients)


def test_no_tail_sweep_permutation_equivariance():
    atomic, shape, density, state, grids, options = fixture()
    expected = core.IsaNoTailSweep(atomic, shape, [[0], [0]], density).run(state, grids, options)
    permuted = core.IsaSweepState()
    permuted.atomic_coefficients = state.atomic_coefficients[::-1]
    permuted.shape_coefficients = state.shape_coefficients[::-1]
    # Molecular density centre numbering is NOT permuted with sweep atom numbering.
    actual = core.IsaNoTailSweep(atomic[::-1], shape[::-1], [[0], [0]], density).run(permuted, grids[::-1], options)
    np.testing.assert_allclose(actual.next.atomic_coefficients, expected.next.atomic_coefficients[::-1], atol=2e-16)


def test_no_tail_clipping_screening_and_zero_denominators():
    atomic, shape, density, state, grids, options = fixture()
    sweep = core.IsaNoTailSweep(atomic, shape, [[0], [0]], density)
    state.shape_coefficients = [[-.8], [1.4]]
    grids[1].shape_sites = [1]
    grids[1].density_sites = [0]  # deliberately distinct index domains
    result = sweep.run(state, grids, options)
    np.testing.assert_allclose(result.next.atomic_coefficients, reference(state, grids, options)[:, :1], atol=3e-16)
    assert result.clipped_shape_points == [47, 0]
    assert result.next.shape_coefficients[0] == [0.]
    grids[0].shape_sites = [0]
    result = sweep.run(state, grids, options)
    assert result.fits[0].excluded_points == 47
    assert result.fits[0].population == 0


@pytest.mark.parametrize('field,value,match', [
    ('shape_sites', [], 'include'), ('shape_sites', [1], 'include'),
    ('shape_sites', [0, 0], 'Duplicate'), ('shape_sites', [-1, 0], 'range'),
    ('shape_sites', [0, 2], 'range'), ('density_sites', [2], 'range'),
    ('points', [], 'dimension'), ('weights', [], 'dimension')])
def test_no_tail_sweep_rejects_grids(field, value, match):
    atomic, shape, density, state, grids, options = fixture()
    setattr(grids[0], field, value)
    with pytest.raises(ValueError, match=match):
        core.IsaNoTailSweep(atomic, shape, [[0], [0]], density).run(state, grids, options)


@pytest.mark.parametrize('field,value', [('shape_coefficients', []), ('atomic_coefficients', [[1.]]),
    ('shape_coefficients', [[1., 2.], [1.]]), ('atomic_coefficients', [[np.nan], [1.]])])
def test_no_tail_sweep_rejects_state(field, value):
    atomic, shape, density, state, grids, options = fixture()
    setattr(state, field, value)
    with pytest.raises(ValueError, match='count|dimension|finite'):
        core.IsaNoTailSweep(atomic, shape, [[0], [0]], density).run(state, grids, options)


def test_no_tail_sweep_constructor_and_late_failure_atomicity():
    atomic, shape, density, state, grids, options = fixture()
    with pytest.raises(ValueError, match='nonempty'):
        core.IsaNoTailSweep([], [], [], density)
    with pytest.raises(ValueError, match='count'):
        core.IsaNoTailSweep(atomic, shape[:1], [[0], [0]], density)
    sweep = core.IsaNoTailSweep(atomic, shape, [[0], [0]], density)
    old_atomic, old_shape = state.atomic_coefficients, state.shape_coefficients
    grids[1].weights = [np.nan]*47
    with pytest.raises(ValueError, match='finite'):
        sweep.run(state, grids, options)
    assert state.atomic_coefficients == old_atomic and state.shape_coefficients == old_shape


def test_supplied_tail_sweep_preserves_signed_interior():
    atomic, shapes, density, state, grids, options = fixture()
    state.shape_coefficients = [[-.8], [1.4]]
    tails = []
    for amplitude in [.4, .6]:
        t = core.IsaExponentialTail()
        t.defined, t.amplitude, t.exponent, t.cutoff = True, amplitude, 2., 1.
        tails.append(t)
    sweep = core.IsaASweep(atomic, shapes, [[0], [0]], density)
    actual = sweep.run_with_tails(state, grids, tails, [True, False], options)
    centres = np.array([[-1., 0., 0.], [1., 0., 0.]])
    for a, grid in enumerate(grids):
        r = np.linalg.norm(np.asarray(grid.points)[:, None, :]-centres[None, :, :], axis=2)
        w = np.exp(-.7*r*r)*np.array([-.8, 1.4])
        w[:, 0] = np.where(r[:, 0]>1., .4*np.exp(-2*r[:, 0]), w[:, 0])
        samples = core.IsaAFitSamples()
        samples.points, samples.weights = grid.points, grid.weights
        samples.density_sites, samples.previous = grid.density_sites, state.atomic_coefficients[a]
        samples.shape, samples.shape_sum = w[:, a].tolist(), w.sum(axis=1).tolist()
        expected = core.IsaAFitProvider(atomic[a], density).fit(samples, options)
        np.testing.assert_allclose(actual.fits[a].coefficients.np, expected.coefficients.np, atol=1e-14)
    assert actual.clipped_shape_points == [0, 0]
    with pytest.raises(ValueError, match='policy count'):
        sweep.run_with_tails(state, grids, [], [True, False], options)
