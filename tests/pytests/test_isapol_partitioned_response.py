"""Independent supplied-partition checks; NOT native ISA/CamCASP property parity."""
import numpy as np
import pytest
import psi4

core = psi4.core


def basis(centres=((0., 0., 0.),), role=None):
    shells = []
    for a in range(len(centres)):
        s = core.IsaGaussianShell()
        s.centre, s.l, s.exponents, s.coefficients = a, 0, [0.7], [1.2]
        shells.append(s)
    return core.IsaExplicitBasis(role or core.IsaBasisRole.MolecularAux,
                                core.IsaBasisRepresentation.Spherical, centres, shells)


def site(label="A", origin=(0., 0., 0.), rank=2, points=None, ratio=None, neighbours=(0,)):
    points = np.array([[.2, -.3, .7], [-.4, .8, -.1], [.6, .2, -.5]]) if points is None else np.asarray(points)
    samples = core.IsaMultipoleSamples()
    samples.points = points.tolist()
    samples.weights = [1.] * len(points)
    samples.shape = [1.] * len(points) if ratio is None else list(ratio)
    samples.shape_sum = [1.] * len(points)
    samples.auxiliary_sites = list(neighbours)
    s = core.IsaMultipoleSite()
    s.label, s.origin, s.rank, s.samples = label, origin, rank, samples
    return s


def q(sites, aux=None, **kwargs):
    return core.IsaPartitionedMultipoles(aux or basis(), sites, "synthetic explicit partition", **kwargs)


def response(partition, matrices, frequencies=None, representation="fitted_density_coefficients"):
    return core.IsaDistributedResponse(partition, frequencies or list(range(len(matrices))),
        [core.Matrix.from_array(np.asarray(c)) for c in matrices], representation, "synthetic coefficient response")


def poly2(points):
    x, y, z = np.asarray(points).T
    return np.array([np.ones_like(x), z, x, y, .5*(2*z*z-x*x-y*y),
                     np.sqrt(3)*x*z, np.sqrt(3)*y*z, np.sqrt(3)/2*(x*x-y*y), np.sqrt(3)*x*y])


def test_polynomial_quadrature_and_metadata():
    s = site()
    points = np.array(s.samples.points)
    out = q([s])
    expected = poly2(points) @ (1.2*np.exp(-.7*np.sum(points**2, axis=1)))
    np.testing.assert_allclose(out.values.np[:, 0], expected, rtol=2e-15, atol=2e-15)
    assert out.offsets == [0, 9] and out.ranks == [2] and out.labels == ["A"]
    assert out.components == ["00", "10", "11c", "11s", "20", "21c", "21s", "22c", "22s"]
    assert out.frame == "global_cartesian" and out.units == "atomic_units"
    assert out.excluded_denominators == [0] and out.negative_ratios == [0]


@pytest.mark.parametrize("rank", range(5))
def test_rank_addition_theorem(rank):
    # Sum_m R_lm(r)^2 = |r|^(2l); an independent normalization invariant.
    p = np.array([[.43, -.72, .29]])
    out = q([site(rank=rank, points=p)]).values.np[:, 0]
    out /= 1.2*np.exp(-.7*np.sum(p*p))
    for l in range(rank+1):
        np.testing.assert_allclose(out[l*l:(l+1)**2] @ out[l*l:(l+1)**2],
                                   np.sum(p*p)**l, rtol=3e-15, atol=1e-15)


@pytest.mark.parametrize("rank", [3, 4])
def test_high_rank_against_legendre_polynomials(rank):
    # Independent angular oracle, not the production solid-harmonic recurrence.
    import math
    from numpy.polynomial import Legendre
    points = np.array([[.43, -.72, .29], [-.4, .3, -.8]])
    radius = np.linalg.norm(points, axis=1)
    cos_theta = points[:, 2]/radius
    phi = np.arctan2(points[:, 1], points[:, 0])
    expected = []
    for l in range(rank+1):
        p = Legendre.basis(l)
        expected.append(radius**l*p(cos_theta))
        for m in range(1, l+1):
            norm = np.sqrt(2*math.factorial(l-m)/math.factorial(l+m))
            radial = norm*radius**l*(1-cos_theta**2)**(m/2)*p.deriv(m)(cos_theta)
            expected.extend([radial*np.cos(m*phi), radial*np.sin(m*phi)])
    expected = np.array(expected) @ (1.2*np.exp(-.7*radius**2))
    np.testing.assert_allclose(q([site(rank=rank, points=points)]).values.np[:, 0],
                               expected, rtol=1e-13, atol=2e-15)


def test_dipole_rotation_covariance():
    s = site(rank=1)
    angle = .71
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rotated = site(rank=1, points=np.array(s.samples.points) @ rotation.T)
    before, after = q([s]).values.np, q([rotated]).values.np
    transform = np.eye(4)
    transform[1:, 1:] = rotation[np.ix_([2, 0, 1], [2, 0, 1])]
    np.testing.assert_allclose(after, transform @ before, atol=2e-15)


def test_partition_unity_translation_and_charge_flow():
    points = np.array([[.2, -.3, .7], [-.4, .8, -.1], [.6, .2, -.5]])
    origins = np.array([[.1, .3, -.2], [-.7, .4, .1]])
    weights = np.array([.2, .6, -.1])  # signed partition retained
    sites = [site("A", origins[0], 1, points, weights),
             site("B", origins[1], 1, points, 1-weights)]
    part = q(sites)
    # Local multipoles require charge-flow translation; local dipole sums alone are wrong.
    translate = np.zeros((4, 8))
    for a in range(2):
        translate[:, 4*a:4*a+4] = np.eye(4)
        translate[1:, 4*a] = origins[a, [2, 0, 1]]
    global_q = q([site(rank=1, points=points)]).values.np
    np.testing.assert_allclose(translate @ part.values.np, global_q, atol=2e-15)
    alpha = response(part, [[[-2.]]]).at_index(0).np
    np.testing.assert_allclose(translate @ alpha @ translate.T, 2*global_q@global_q.T, atol=1e-14)
    assert not np.allclose(part.values.np[1:4]+part.values.np[5:8], global_q[1:4])
    assert part.negative_ratios == [1, 0]


def test_rigid_translation_and_site_permutation():
    shift = np.array([1.3, -.8, .2])
    s = site(rank=4)
    shifted = site(rank=4, origin=shift, points=np.array(s.samples.points)+shift)
    np.testing.assert_allclose(q([s]).values.np, q([shifted], basis([shift])).values.np, atol=2e-15)
    a, b = site("A", rank=1), site("B", origin=(.1, -.2, .3), rank=2)
    ab, ba = q([a, b]), q([b, a])
    permutation = list(range(4, 13))+list(range(4))
    np.testing.assert_array_equal(ba.values.np, ab.values.np[permutation])


def test_raw_contraction_frequency_and_ownership():
    aux = basis([(0, 0, 0), (.6, .2, -.1)])
    s = site(neighbours=(0, 1))
    part = q([s], aux)
    original_q = part.values.np.copy()
    c = np.array([[-2., .3], [.1, -1.]])  # intentionally nonreciprocal
    source = core.Matrix.from_array(c)
    out = core.IsaDistributedResponse(part, [0., .7], [source, source],
                                     "fitted_density_coefficients", "raw operator")
    expected = -original_q @ c @ original_q.T
    np.testing.assert_allclose(out.at_index(0).np, expected, atol=2e-15)
    assert out.reciprocity_errors[0] > 0
    assert out.frequencies == [0., .7]
    source.np[:] = 100
    part.values.np[:] = 50
    out.at_index(0).np[:] = 20
    out.partition.values.np[:] = 90
    s.rank = 0
    np.testing.assert_array_equal(part.values.np, original_q)
    np.testing.assert_allclose(out.at_index(1).np, expected, atol=2e-15)
    with pytest.raises(IndexError):
        out.at_index(2)


def test_gaussian_integrated_moments_through_rank_four():
    # Tensor-product Hermite quadrature exactly integrates each degree <=4
    # regular harmonic times the explicit s Gaussian. All centered nonzero
    # ranks integrate to zero; q_00 has the analytic Gaussian charge.
    from itertools import product
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    triples = np.array(list(product(range(5), repeat=3)))
    points = nodes[triples]/np.sqrt(.7)
    quadrature = np.prod(weights[triples], axis=1)/.7**1.5
    quadrature *= np.exp(.7*np.sum(points*points, axis=1))
    s = site(rank=4, points=points)
    sam = s.samples
    sam.weights = quadrature.tolist()
    s.samples = sam
    actual = q([s]).values.np[:, 0]
    expected = np.zeros(25)
    expected[0] = 1.2*(np.pi/.7)**1.5
    np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_charge_flow_sum_rule_and_frequency_limit():
    aux = basis([(0, 0, 0), (.6, .2, -.1)])
    sites = [site("A", rank=1, ratio=[.2, .4, .6], neighbours=(0, 1)),
             site("B", rank=1, ratio=[.8, .6, .4], neighbours=(0, 1))]
    part = q(sites, aux)
    total_charge_leg = part.values.np[[0, 4]].sum(axis=0)
    neutral_transition = np.array([-total_charge_leg[1], total_charge_leg[0]])
    c = -np.outer(neutral_transition, neutral_transition)
    out = response(part, [c, c/(1+1.e6)], [0., 1000.])
    alpha = out.at_index(0).np
    np.testing.assert_allclose(alpha[[0, 4]].sum(axis=0), 0., atol=3e-14)
    np.testing.assert_allclose(alpha[:, [0, 4]].sum(axis=1), 0., atol=3e-14)
    np.testing.assert_allclose(out.at_index(1).np, alpha/(1+1.e6), atol=1e-19)
    assert out.reciprocity_errors[0] < 1e-14


def test_screening_and_denominator_diagnostics():
    s = site(rank=0)
    sam = s.samples
    sam.shape_sum = [0., 1.e-36, -2.]
    sam.shape = [3., -4., 1.]
    s.samples = sam
    part = q([s])
    expected = -.5*1.2*np.exp(-.7*np.sum(np.array(sam.points[-1])**2))
    np.testing.assert_allclose(part.values.np, [[expected]])
    assert part.excluded_denominators == [2] and part.negative_ratios == [1]
    sam.auxiliary_sites = []
    s.samples = sam
    np.testing.assert_array_equal(q([s]).values.np, [[0.]])


@pytest.mark.parametrize("field,value", [("rank", -1), ("rank", 5), ("label", ""), ("origin", (np.nan, 0, 0))])
def test_invalid_site(field, value):
    s = site()
    setattr(s, field, value)
    with pytest.raises(ValueError):
        q([s])


@pytest.mark.parametrize("field,value", [("weights", []), ("shape", [np.nan]*3),
    ("shape_sum", [np.inf]*3), ("points", [[np.inf, 0, 0]]*3),
    ("auxiliary_sites", [0, 0]), ("auxiliary_sites", [1])])
def test_invalid_samples(field, value):
    s = site()
    sam = s.samples
    setattr(sam, field, value)
    s.samples = sam
    with pytest.raises(ValueError):
        q([s])


@pytest.mark.parametrize("cutoff", [-1., np.inf, np.nan])
def test_invalid_cutoff(cutoff):
    with pytest.raises(ValueError):
        q([site()], denominator_cutoff=cutoff)


def test_invalid_partition_contracts():
    with pytest.raises(ValueError):
        q([])
    with pytest.raises(ValueError):
        q([site(), site()])
    with pytest.raises(ValueError):
        q([site()], basis(role=core.IsaBasisRole.AtomAux))
    with pytest.raises(ValueError):
        core.IsaPartitionedMultipoles(basis(), [site()], "")


@pytest.mark.parametrize("frequency", [-1., np.inf, np.nan])
def test_invalid_frequency(frequency):
    with pytest.raises(ValueError):
        response(q([site()]), [[[-1.]]], [frequency])


@pytest.mark.parametrize("matrix", [[[np.nan]], [[np.inf]], np.eye(2)])
def test_invalid_response_matrix(matrix):
    with pytest.raises(ValueError):
        response(q([site()]), [matrix])


@pytest.mark.parametrize("kind", ["ratio", "weight", "harmonic", "accumulation"])
def test_finite_partition_overflow(kind):
    s = site(rank=4, points=[[0., 0., 0.], [0., 0., 0.]])
    sam = s.samples
    if kind == "ratio":
        sam.shape, sam.shape_sum = [1.e308]*2, [1.e-308]*2
    elif kind == "weight":
        sam.shape, sam.weights = [2.]*2, [1.e308]*2
    elif kind == "harmonic":
        sam.points = [[1.e80, 0., 0.]]*2
    else:
        sam.weights = [1.e308]*2
    s.samples = sam
    with pytest.raises(ValueError, match="[Nn]onfinite"):
        q([s], denominator_cutoff=0.)


@pytest.mark.parametrize("coefficient", [1.e120, 1.e250])
def test_finite_response_gemm_overflow(coefficient):
    s = site(rank=0, points=[[0., 0., 0.]])
    sam = s.samples
    sam.weights = [1.e100]
    s.samples = sam
    part = q([s])
    with pytest.raises(ValueError, match="Nonfinite"):
        response(part, [[[coefficient]]])


def test_partial_auxiliary_screening():
    centres = np.array([[0., 0., 0.], [.6, .2, -.1]])
    s = site(neighbours=(1,))
    part = q([s], basis(centres))
    p = np.array(s.samples.points)
    expected = poly2(p) @ (1.2*np.exp(-.7*np.sum((p-centres[1])**2, axis=1)))
    np.testing.assert_array_equal(part.values.np[:, 0], 0.)
    np.testing.assert_allclose(part.values.np[:, 1], expected, atol=2e-15)


def test_cartesian_p_auxiliary_quadrature():
    shell = core.IsaGaussianShell()
    shell.centre, shell.l, shell.exponents, shell.coefficients = 0, 1, [.7], [1.2]
    aux = core.IsaExplicitBasis(core.IsaBasisRole.MolecularAux,
        core.IsaBasisRepresentation.Cartesian, [[0., 0., 0.]], [shell])
    s = site()
    p = np.array(s.samples.points)
    values = p*(1.2*np.exp(-.7*np.sum(p*p, axis=1)))[:, None]
    np.testing.assert_allclose(q([s], aux).values.np, poly2(p)@values, atol=2e-15)


def test_null_symmetry_and_provenance_errors():
    part = q([site()])
    kwargs = ("fitted_density_coefficients", "explicit")
    with pytest.raises(ValueError, match="Null"):
        core.IsaDistributedResponse(part, [0.], [None], *kwargs)
    dims = core.Dimension.from_list([1, 1])
    matrix = core.Matrix("symmetry blocks", dims, dims)
    with pytest.raises(ValueError, match="symmetry block"):
        core.IsaDistributedResponse(part, [0.], [matrix], *kwargs)
    with pytest.raises(ValueError, match="provenance"):
        core.IsaDistributedResponse(part, [0.], [core.Matrix.from_array(np.ones((1, 1)))],
                                    "fitted_density_coefficients", "")


def test_wrong_representation_and_frequency_count():
    part = q([site()])
    with pytest.raises(ValueError, match="no implicit metric"):
        response(part, [[[-1.]]], representation="fdds_coulomb_auxiliary")
    with pytest.raises(ValueError):
        response(part, [[[-1.]]], [0., 1.])
    with pytest.raises(ValueError):
        core.IsaDistributedResponse(part, [], [], "fitted_density_coefficients", "empty")
