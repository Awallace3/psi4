# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Deterministic native performance paths; parent must rebuild before running.

No timing assertions or changed scientific tolerances. These fixtures are small,
explicit inputs, not a replacement for the parent's water trajectory/RSS gates.
"""
from contextlib import contextmanager

import numpy as np
import pytest
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@contextmanager
def threads(n):
    saved = core.get_num_threads()
    try:
        core.set_num_threads(n)
        yield
    finally:
        core.set_num_threads(saved)


def basis(role, centres, shells, representation=None):
    return core.IsaExplicitBasis(role, representation or core.IsaBasisRepresentation.Spherical,
                                 centres, shells)


def shell(centre, exponent, coefficient=1., l=0):
    s = core.IsaGaussianShell()
    s.centre, s.l = centre, l
    s.exponents, s.coefficients = [exponent], [coefficient]
    return s


def setup(cache_bytes, grid_kind="equal", s_block=True):
    centres = [[0., 0., 0.], [1., -.2, .3]]
    atomic, shapes, all_shells = [], [], []
    for a in range(2):
        ss = [shell(a, .4), shell(a, 1.1)]
        atomic.append(basis(core.IsaBasisRole.AtomAux, centres, ss+[shell(a, .8, l=1)]))
        shapes.append(basis(core.IsaBasisRole.Shape, centres, ss[::-1]))
        all_shells.extend(ss)
    density = core.IsaFixedDensity(basis(core.IsaBasisRole.MolecularAux, centres, all_shells),
                                   [1.5, .7, .8, .2])
    rng = np.random.default_rng(732)
    xyz = rng.normal(size=(513, 3))*1.4
    grids = []
    for a in range(2):
        g = core.IsaNoTailGrid()
        points = xyz.copy()
        if a and grid_kind == "unequal":
            points[:, 0] += .123
        if a and grid_kind == "permuted":
            points = points[::-1]
        g.points = points.tolist()
        g.weights = (np.arange(len(points))*.00001+.02+a*.003).tolist()
        g.shape_sites = [a, 1-a]  # preserve different neighbour summation orders
        g.density_sites = [1, 0] if a == 0 else [1]  # equal coordinates != equal screening
        grids.append(g)
    options = core.IsaAControllerOptions()
    options.cache_max_bytes = cache_bytes
    options.max_iterations, options.convergence = 3, 1.e-30
    options.mixing, options.mixing_skip = .2, 0
    options.tail_cutoffs, options.tail_iteration_limit = [1.5, 1.5], 0
    options.fit.w_eps, options.fit.positive_lambda = .05, .001
    options.fit.s_block_only = s_block
    options.w_eps_activation = options.positive_activation = 1.
    state = core.IsaSweepState()
    state.atomic_coefficients = [[-.2, 2., .1, -.2, .3], [.1, 1., -.1, .2, -.3]]
    state.shape_coefficients = [[2., -.2], [1., .1]]
    controller = core.IsaAController(atomic, shapes, [[1, 0], [1, 0]], density, grids, options)
    return controller, state, grids, atomic, shapes, density


def snapshot(result):
    """Flatten all numeric trajectory diagnostics, retaining their exact order."""
    out = []
    for step in result.history:
        state = step.next
        out.extend([state.iteration, state.max_delta, state.converged, state.apply_tails,
                    state.active_w_eps, state.active_positive_lambda])
        for rows in (state.coefficients.atomic_coefficients, state.coefficients.shape_coefficients):
            for row in rows:
                out.extend(row)
        for seq in (step.deltas, step.shape_charges, state.saved_shape_charges,
                    step.raw_sweep.clipped_shape_points, step.atom_converged):
            out.extend(seq)
        for tail in state.tails:
            out.extend([tail.defined, tail.amplitude, tail.exponent, tail.cutoff])
        for fit in step.raw_sweep.fits:
            out.extend([fit.population, fit.relative_residual, fit.excluded_points])
            for matrix in (fit.metric, fit.rhs, fit.coefficients):
                out.extend(matrix.np.ravel())
    return np.asarray(out)


@pytest.mark.parametrize("grid_kind", ["equal", "unequal", "permuted"])
@pytest.mark.parametrize("s_block", [True, False])
def test_controller_cache_fallback_and_thread_exact_trajectory(grid_kind, s_block):
    with threads(1):
        c, s, *_ = setup(0, grid_kind, s_block)
        reference = c.run(c.initialize(s))
    # 1 byte is a nonzero failed-admission budget, not just a disable switch.
    for cache_bytes, nt in [(1, 2), (192*1024**2, 1), (192*1024**2, 4)]:
        with threads(nt):
            c, s, *_ = setup(cache_bytes, grid_kind, s_block)
            actual = c.run(c.initialize(s))
            assert actual.termination == reference.termination
            np.testing.assert_array_equal(snapshot(actual), snapshot(reference))


def test_cache_admission_accounts_for_construction_peak():
    # The two-site fixture has nf=5 and np=513 on each site. Admission includes
    # copied preparation samples plus assembled arrays, not only persistent data.
    limit = 8*1024**2 + 2 * (2*5*5 + 3*5 + 513*(5+12)) * 8
    below, state, *_ = setup(limit-1)
    exact, other_state, *_ = setup(limit)
    assert not below.prepared_cache_enabled
    assert exact.prepared_cache_enabled
    np.testing.assert_array_equal(snapshot(below.run(below.initialize(state))),
                                  snapshot(exact.run(exact.initialize(other_state))))


def test_cacheless_snapshot_releases_preparation_without_mutating_original():
    prepared, state, *_ = setup(192*1024**2)
    light = prepared.without_prepared_cache()
    assert prepared.prepared_cache_enabled and not light.prepared_cache_enabled
    expected = prepared.run(prepared.initialize(state))
    actual = light.run(light.initialize(state))
    np.testing.assert_array_equal(snapshot(actual), snapshot(expected))
    actual.history[0].raw_sweep.fits[0].metric.np[:] = 9.
    np.testing.assert_array_equal(snapshot(light.run(light.initialize(state))), snapshot(expected))
    assert prepared.prepared_cache_enabled
    del prepared
    import gc
    gc.collect()
    np.testing.assert_array_equal(snapshot(light.run(light.initialize(state))), snapshot(expected))


def test_prepared_controller_and_expert_matrix_mutation_independence():
    with threads(2):
        c, s, grids, atomic, _, density = setup(192*1024**2)
        initial = c.initialize(s)
        expected = snapshot(c.run(initial))
        # Constructor owns grids/screens; no external getter or result can poison preparation.
        grids[0].points = [[99., 99., 99.]]
        grids[0].weights = [0.]
        grids[0].density_sites = []
        first = c.run(initial)
        first.history[0].raw_sweep.fits[0].metric.np[:] = 9.
        first.history[0].raw_sweep.fits[0].rhs.np[:] = 8.
        np.testing.assert_array_equal(snapshot(c.run(initial)), expected)
        samples = core.IsaAFitSamples()
        samples.points = grids[1].points
        samples.weights = grids[1].weights
        samples.shape = samples.shape_sum = [1.]*len(samples.points)
        samples.previous = [.1, .2, 0., 0., 0.]
        samples.density_sites = [0, 1]
        p = core.IsaAFitProvider(atomic[0], density)
        options = core.IsaAFitOptions()
        first = p.assemble(samples, options)
        expected_phi = first.basis_values.np.copy()
        first.basis_values.np[:] = 3.
        first.overlap.np[:] = 4.
        np.testing.assert_array_equal(p.assemble(samples, options).basis_values.np, expected_phi)
        np.testing.assert_array_equal(snapshot(c.run(initial)), expected)


@pytest.mark.parametrize("representation", [core.IsaBasisRepresentation.Cartesian,
                                            core.IsaBasisRepresentation.Spherical])
def test_collocation_and_streaming_density_preserve_order(representation):
    ss = [shell(1, .4, -.3, 4), shell(0, 1.1, .7, 1), shell(1, .8, 1.2)]
    ss[0].exponents, ss[0].coefficients = [.4, .9], [-.3, .2]
    b = basis(core.IsaBasisRole.MolecularAux, [[0., 0., 0.], [.2, -.3, .4]], ss, representation)
    points = np.random.default_rng(11).normal(size=(4103, 3)).tolist()  # crosses 4096 scratch block
    coeff = np.linspace(-.4, .7, b.nfunction).tolist()
    density = core.IsaFixedDensity(b, coeff)
    for screen in ([], [1], [1, 0], [0, 1]):
        with threads(1):
            values = b.evaluate_screened(points, screen).np.copy()
            expected = np.zeros(len(points))
            for k, c in enumerate(coeff):
                expected += values[:, k]*c
        with threads(4):
            np.testing.assert_array_equal(b.evaluate_screened(points, screen).np, values)
            np.testing.assert_array_equal(density.evaluate(points, screen), expected)


def test_shape_signed_interior_strict_cutoff_and_worker_errors():
    b = basis(core.IsaBasisRole.Shape, [[0., 0., 0.]], [shell(0, .7)])
    shape = core.IsaGaussianShape(b, [-1.])
    tail = core.IsaExponentialTail()
    tail.defined, tail.amplitude, tail.exponent, tail.cutoff = True, .3, 2., 1.
    points = [[0., 0., r] for r in [0., 1., np.nextafter(1., 2.)]*200]
    with threads(1):
        signed = shape.sample(points, tail, True)
        clipped = shape.sample(points, tail, False)
        tail_fit = shape.fit_tail(1.5)
    with threads(4):
        np.testing.assert_array_equal(shape.sample(points, tail, True), signed)
        np.testing.assert_array_equal(shape.sample(points, tail, False), clipped)
        assert signed[1] < 0 < signed[2] and clipped == [0.]*len(points)
        assert shape.fit_tail(1.5).tail.exponent == tail_fit.tail.exponent
        bad = [[0., 0., 0.]]*600
        bad[257] = [1.e308, 0., 0.]
        for n in (1, 4):
            with threads(n):
                with pytest.raises(ValueError, match="Nonfinite squared distance"):
                    b.evaluate(bad)
                with pytest.raises(ValueError, match="Nonfinite shape sample distance"):
                    shape.sample(bad, tail, True)
        np.testing.assert_array_equal(shape.sample(points, tail, True), signed)


def test_parallel_failure_selection_is_lowest_point_index():
    b = basis(core.IsaBasisRole.MolecularAux, [[0., 0., 0.]],
              [shell(0, .0001, 1.e308, 4)], core.IsaBasisRepresentation.Cartesian)
    for first, later, message in [([10., 0., 0.], [1.e308, 0., 0.], 'Nonfinite basis sample'),
                                  ([1.e308, 0., 0.], [10., 0., 0.], 'Nonfinite squared distance')]:
        points = [[0., 0., 0.]]*768
        points[17], points[700] = first, later
        for n in (1, 2, 4):
            with threads(n), pytest.raises(ValueError, match=message):
                b.evaluate(points)


def test_underflow_signed_zero_bits_are_thread_invariant():
    b = basis(core.IsaBasisRole.MolecularAux, [[0., 0., 0.]], [shell(0, .7, 1., 1)],
              core.IsaBasisRepresentation.Cartesian)
    points = [[-1.e150, 0., 0.], [1.e150, 0., 0.]]*300
    with threads(1):
        expected = b.evaluate(points).np.copy()
    assert np.all(expected == 0.) and np.any(np.signbit(expected))
    for n in (2, 4):
        with threads(n):
            actual = b.evaluate(points).np.copy()
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_aux_q_streaming_matches_full_collocation_point_order():
    from test_isapol_partitioned_response import basis as aux_basis, site, q
    xyz = np.random.default_rng(812).normal(size=(4103, 3))
    sample_site = site('A', rank=0, points=xyz, neighbours=[1, 0])
    auxiliary = aux_basis([[0., 0., 0.], [.2, .3, -.1]])
    samples = sample_site.samples
    with threads(1):
        phi = auxiliary.evaluate_screened(samples.points, samples.auxiliary_sites).np.copy()
        expected = np.zeros(phi.shape[1])
        weights, shapes, sums = samples.weights, samples.shape, samples.shape_sum
        for index in range(len(xyz)):
            if abs(sums[index]) > 1.e-36:
                weight = weights[index] * (shapes[index] / sums[index])
                expected += weight * phi[index]
    with threads(4):
        np.testing.assert_array_equal(q([sample_site], auxiliary).values.np[0], expected)


def test_aux_q_validates_empty_and_later_batches_before_return():
    from test_isapol_partitioned_response import basis as aux_basis, site, q
    auxiliary = aux_basis()
    with pytest.raises(ValueError, match='Neighbour site out of range'):
        q([site(points=[], neighbours=[9])], auxiliary)
    points = np.zeros((4103, 3))
    points[4098, 0] = 1.e308
    s = site(points=points)
    samples = s.samples
    samples.shape_sum = [0.]*len(points)  # exclusions cannot bypass basis validation
    s.samples = samples
    with pytest.raises(ValueError, match='Nonfinite squared distance'):
        q([s], auxiliary)


def test_aux_q_high_rank_boundary_diagnostics():
    from test_isapol_partitioned_response import basis as aux_basis, site, q
    auxiliary = aux_basis()
    s = site(rank=4, points=np.random.default_rng(741).normal(size=(4103, 3)))
    samples = s.samples
    sums = [1.]*4103
    sums[4095], sums[4096] = 0., -1.
    samples.shape_sum = sums
    s.samples = samples
    with threads(1):
        serial = q([s], auxiliary)
    with threads(4):
        parallel = q([s], auxiliary)
    np.testing.assert_array_equal(parallel.values.np, serial.values.np)
    assert parallel.excluded_denominators == serial.excluded_denominators == [1]
    assert parallel.negative_ratios == serial.negative_ratios == [1]


def test_partition_q_component_site_and_screen_order_across_threads():
    from test_isapol_partitioned_response import basis as aux_basis, site, q
    xyz = np.random.default_rng(43).normal(size=(513, 3))
    a = site("A", rank=4, points=xyz, neighbours=[1, 0])
    b = site("B", rank=1, points=xyz[::-1], neighbours=[1])
    aux = aux_basis([[0., 0., 0.], [.2, .3, -.1]])
    with threads(1):
        ab = q([a, b], aux)
        expected, components = ab.values.np.copy(), ab.components
    with threads(4):
        actual, ba = q([a, b], aux), q([b, a], aux)
        np.testing.assert_array_equal(actual.values.np, expected)
        np.testing.assert_array_equal(ba.values.np, np.concatenate([expected[25:], expected[:25]]))
        assert actual.components == components and actual.offsets == [0, 25, 29]
