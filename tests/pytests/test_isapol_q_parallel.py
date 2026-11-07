"""Frozen supplied operands and original p/t/k serial sum, NOT SCF acceptance.

Parent runs these native API tests in its freshly rebuilt compute allocation.
No saved molecular orbitals, changed scientific tolerances or timing assertions.
"""
from contextlib import contextmanager

import numpy as np
import pytest
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@contextmanager
def threads(count):
    saved = core.get_num_threads()
    try:
        core.set_num_threads(count)
        yield
    finally:
        core.set_num_threads(saved)


def shell(centre, l, exponent, coefficient):
    result = core.IsaGaussianShell()
    result.centre, result.l = centre, l
    result.exponents, result.coefficients = [exponent], [coefficient]
    return result


def auxiliary(representation, columns=None):
    if columns is not None:
        # Exact widths on both sides of the 8 MiB/4096-row collocation boundary.
        centres = [[0., 0., 0.]]
        shells = [shell(0, 0, .3 + i*.001, (-1. if i % 3 == 0 else 1.)) for i in range(columns)]
    else:
        centres = [[0., 0., 0.], [.2, -.3, .1]]
        shells = [shell(a, l, .4 + .1*l, (-.7 if l % 2 else 1.1))
                  for a in range(2) for l in range(5)]
    return core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux, representation, centres, shells)


def frozen_site(label, rank, points, screen, seed):
    rng = np.random.default_rng(seed)
    count = len(points)
    samples = core.IsaMultipoleSamples()
    samples.points = points.tolist()
    samples.weights = rng.normal(size=count).tolist()
    denominator = rng.uniform(.5, 1.5, count)
    # Exclusions on both sides of 4096 and strict cutoff-neighbour floats.
    indices = [p for p in (0, 1, 127, 255, 256, 4095, 4096, 4097) if p < count]
    boundary = [0., 1.e-36, -1.e-36, np.nextafter(1.e-36, 0.),
                np.nextafter(1.e-36, np.inf), -1., 0., -2.]
    for p, value in zip(indices, boundary):
        denominator[p] = value
    ratio = rng.uniform(-.5, 1.5, count)
    shape = denominator * ratio
    weights = np.asarray(samples.weights)
    if count:
        weights[-1] = -0.
        shape[-1] = 0.
    samples.weights = weights.tolist()
    samples.shape, samples.shape_sum = shape.tolist(), denominator.tolist()
    samples.auxiliary_sites = list(screen)
    result = core.IsaMultipoleSite()
    result.label, result.origin, result.rank = label, [.07, -.03, .09], rank
    result.samples = samples
    return result


def original_serial(aux, sites, cutoff=1.e-36):
    """Old p->component->AUX arithmetic, with native frozen basis/harmonics.

    Ufuncs act ONLY across independent AUX columns: no sum/dot/BLAS reduction
    over points, and no fused multiply-add. Each output retains every point in
    the old order. Harmonic normalization has separate independent oracle tests.
    """
    outputs, excluded, negative = [], [], []
    for site in sites:
        s = site.samples
        # pybind STL properties are owned copies: fetch once, not once per point.
        points, weights, shapes, sums = s.points, s.weights, s.shape, s.shape_sum
        origin, rank = site.origin, site.rank
        phi = aux.evaluate_screened(points, s.auxiliary_sites).np.copy()
        values = np.zeros(((rank+1)**2, aux.nfunction))
        nexc = nneg = 0
        for p in range(len(points)):
            if abs(sums[p]) <= cutoff:
                nexc += 1
                continue
            ratio = shapes[p] / sums[p]
            nneg += ratio < 0
            weight = weights[p] * ratio
            displacement = [points[p][d] - origin[d] for d in range(3)]
            harmonics = core.isa_regular_multipoles(rank, displacement)
            for component, harmonic in enumerate(harmonics):
                prefactor = weight * harmonic
                values[component] += prefactor * phi[p]
        outputs.append(values)
        excluded.append(nexc)
        negative.append(nneg)
    return np.concatenate(outputs), excluded, negative


def calculate(aux, sites, cutoff=1.e-36):
    return core.IsaPartitionedMultipoles(aux, sites, "frozen Q parallel-order fixture", cutoff)


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("representation", [
    core.IsaBasisRepresentation.Cartesian, core.IsaBasisRepresentation.Spherical])
@pytest.mark.parametrize("rank", range(5))
def test_q_original_serial_order_all_threads_and_screened_sites(representation, rank):
    aux = auxiliary(representation)
    xyz = np.random.default_rng(4103).normal(size=(4103, 3))
    sites = [frozen_site("A", rank, xyz, [1, 0], 41),
             frozen_site("B", 1, xyz[::-1].copy(), [1], 97)]
    with threads(1):
        expected, excluded, negative = original_serial(aux, sites)
    for count in (1, 2, 4, 8):
        with threads(count):
            actual = calculate(aux, sites)
            assert_bits(actual.values.np, expected)
            assert actual.excluded_denominators == excluded
            assert actual.negative_ratios == negative
            assert actual.offsets == [0, (rank+1)**2, (rank+1)**2+4]
            # Site reordering may not change any output or its ordered sum.
            reversed_q = calculate(aux, sites[::-1])
            assert_bits(reversed_q.values.np, np.concatenate((expected[-4:], expected[:-4])))


@pytest.mark.parametrize("columns", [63, 64, 65, 255, 256, 257])
def test_q_tile_and_collocation_memory_edges(columns):
    aux = auxiliary(core.IsaBasisRepresentation.Spherical, columns)
    block = min(4096, (8*1024**2)//8//columns)
    xyz = np.random.default_rng(92).normal(size=(block+7, 3))
    s = frozen_site("edge", 4, xyz, [0], columns)
    with threads(1):
        expected, excluded, negative = original_serial(aux, [s])
    for count in (1, 2, 4, 8):
        with threads(count):
            actual = calculate(aux, [s])
            assert_bits(actual.values.np, expected)
            assert actual.excluded_denominators == excluded
            assert actual.negative_ratios == negative


@pytest.mark.parametrize("count", [255, 256, 257])
def test_q_worker_admission_threshold_keeps_bits(count):
    aux = auxiliary(core.IsaBasisRepresentation.Spherical, 65)
    s = frozen_site("threshold", 4, np.random.default_rng(5).normal(size=(count, 3)), [0], 7)
    with threads(1):
        expected, excluded, negative = original_serial(aux, [s])
    for workers in (1, 2, 4, 8):
        with threads(workers):
            actual = calculate(aux, [s])
            assert_bits(actual.values.np, expected)
            assert actual.excluded_denominators == excluded and actual.negative_ratios == negative


@pytest.mark.parametrize("workers", (1, 2))
def test_q_low_thread_original_path_reselected_across_batches(workers):
    aux = auxiliary(core.IsaBasisRepresentation.Spherical, 65)
    # Three native collocation batches; bounded independent reference <4.3 MiB.
    xyz = np.random.default_rng(139).normal(size=(2*4096+7, 3))
    sites = [frozen_site("low-A", 4, xyz, [0], 41),
             frozen_site("low-B", 1, xyz[::-1].copy(), [0], 97)]
    with threads(1):
        expected, excluded, negative = original_serial(aux, sites)
    # Selection must use the current call's thread count, not cached state.
    for count in (workers, 4, workers, 8, workers):
        with threads(count):
            actual = calculate(aux, sites)
            assert_bits(actual.values.np, expected)
            assert actual.excluded_denominators == excluded
            assert actual.negative_ratios == negative
            assert actual.offsets == [0, 25, 29]


def test_q_repeated_workers_do_not_race_or_reuse_mutable_results():
    aux = auxiliary(core.IsaBasisRepresentation.Cartesian)
    xyz = np.random.default_rng(23).normal(size=(513, 3))
    sites = [frozen_site("A", 4, xyz, [1, 0], 11),
             frozen_site("B", 2, xyz[::-1].copy(), [], 12)]
    with threads(1):
        expected, excluded, negative = original_serial(aux, sites)
    for count in (1, 2, 4, 8):
        with threads(count):
            for repeat in range(6):
                actual = calculate(aux, sites)
                assert_bits(actual.values.np, expected)
                assert actual.excluded_denominators == excluded
                assert actual.negative_ratios == negative
                actual.values.np[:] = np.nan  # getter is a snapshot, never live/cache storage
                assert_bits(actual.values.np, expected)


@pytest.mark.parametrize("small", [False, True])
def test_q_cancellation_and_subnormal_products_keep_original_point_order(small):
    aux = auxiliary(core.IsaBasisRepresentation.Spherical, 65)
    points = np.zeros((4104, 3))
    s = frozen_site("ordered", 4, points, [0], 3)
    samples = s.samples
    weights = np.resize(np.array([1.e16, 1., -1.e16, 1.]), len(points))
    if small:
        weights = np.resize(np.array([1.e-320, -5.e-321, 1.e-321, -0.]), len(points))
    samples.weights = weights.tolist()
    samples.shape = samples.shape_sum = [1.]*len(points)
    s.samples = samples
    with threads(1), np.errstate(under="ignore"):
        expected, excluded, negative = original_serial(aux, [s])
    assert np.any(expected != 0.)
    if small:
        assert np.any((np.abs(expected) > 0.) & (np.abs(expected) < np.finfo(float).tiny))
    for count in (1, 2, 4, 8):
        with threads(count):
            assert_bits(calculate(aux, [s]).values.np, expected)


@pytest.mark.parametrize("screen", [[], [0]])
def test_q_signed_zero_underflow_and_excluded_harmonics(screen):
    p = shell(0, 1, .7, 1.)
    aux = core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux,
        core.IsaBasisRepresentation.Cartesian, [[0., 0., 0.]], [p])
    xyz = np.array([[-1.e150, 0., 0.], [1.e150, 0., 0.]]*257)
    s = frozen_site("zeros", 1, xyz, screen, 17)
    with threads(1):
        phi = aux.evaluate_screened(s.samples.points, screen).np.copy()
        expected, excluded, negative = original_serial(aux, [s])
    if screen:
        assert np.any(np.signbit(phi)) and np.all(phi == 0.)
    for count in (1, 2, 4, 8):
        with threads(count):
            actual = calculate(aux, [s])
            assert_bits(actual.values.np, expected)
            assert actual.excluded_denominators == excluded
            assert actual.negative_ratios == negative
    # Large coordinates would overflow rank4 harmonics if exclusions were not
    # honored, although this s-only AUX collocation is finite (underflow to0).
    aux = auxiliary(core.IsaBasisRepresentation.Spherical, 1)
    s = frozen_site("excluded", 4, np.array([[1.e80, 0., 0.]]*513), [0], 12)
    samples = s.samples
    samples.shape_sum = [0.]*513
    s.samples = samples
    for count in (1, 2, 4, 8):
        with threads(count):
            actual = calculate(aux, [s])
            assert_bits(actual.values.np, np.zeros((25, 1)))
            assert actual.excluded_denominators == [513]


def test_q_empty_grids_still_validate_screens():
    aux = auxiliary(core.IsaBasisRepresentation.Spherical, 1)
    for count in (1, 2, 4, 8):
        with threads(count):
            empty = frozen_site("empty", 4, np.empty((0, 3)), [], 1)
            actual = calculate(aux, [empty])
            assert_bits(actual.values.np, np.zeros((25, 1)))
            assert actual.excluded_denominators == actual.negative_ratios == [0]
            samples = empty.samples
            samples.auxiliary_sites = [9]
            empty.samples = samples
            with pytest.raises(ValueError, match="Neighbour site out of range"):
                calculate(aux, [empty])


@pytest.mark.parametrize("kind,message", [
    ("basis_before_sample", "Nonfinite squared distance"),
    ("sample_before_later_batch", "Nonfinite partition sample"),
    ("excluded_nonfinite_sample", "Nonfinite partition sample"),
    ("ratio", "Nonfinite partition ratio"),
    ("weight", "Nonfinite partition integration weight"),
    ("prefactor", "Nonfinite weighted multipole"),
    ("accumulation", "Nonfinite response or multipole matrix"),
])
def test_q_exception_order_and_diagnostics_across_workers(kind, message):
    aux = auxiliary(core.IsaBasisRepresentation.Spherical, 1)
    xyz = np.zeros((4103, 3))
    s = frozen_site("bad", 4, xyz, [0], 1)
    samples = s.samples
    weights, shapes, sums, points = (list(getattr(samples, name))
                                    for name in ("weights", "shape", "shape_sum", "points"))
    if kind == "basis_before_sample":
        weights[2], points[300] = np.nan, [1.e308, 0., 0.]
    elif kind == "sample_before_later_batch":
        weights[2], points[4098] = np.nan, [1.e308, 0., 0.]
    elif kind == "excluded_nonfinite_sample":
        shapes[0], sums[0] = np.nan, 0.
    elif kind == "ratio":
        shapes[0], sums[0] = 1.e308, 1.e-308
    elif kind == "weight":
        weights[0], shapes[0], sums[0] = 1.e308, 2., 1.
    elif kind == "prefactor":
        weights[0], shapes[0], sums[0], points[0] = 1.e308, 1., 1., [10., 0., 0.]
    else:
        weights[:2], shapes[:2], sums[:2] = [1.e308]*2, [1.]*2, [1.]*2
    samples.weights, samples.shape, samples.shape_sum, samples.points = weights, shapes, sums, points
    s.samples = samples
    for count in (1, 2, 4, 8):
        with threads(count), pytest.raises(ValueError, match=message):
            calculate(aux, [s], cutoff=0.)
