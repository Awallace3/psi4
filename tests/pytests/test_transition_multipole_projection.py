import math

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


def _project(points, weights, partition, sites, tau):
    result = psi4.core._atomic_polarizability_test_project_transition_multipoles(
        points, weights, np.asarray(partition).reshape(-1).tolist(), sites,
        psi4.core.Matrix.from_array(np.asarray(tau, dtype=float)),
    )
    return np.asarray(result["values"]), result


def _regular(point):
    x, y, z = point
    r2 = x*x + y*y + z*z
    return np.array([
        1.0, z, x, y,
        (3.0*z*z-r2)/2.0, math.sqrt(3.0)*x*z, math.sqrt(3.0)*y*z,
        math.sqrt(3.0)*(x*x-y*y)/2.0, math.sqrt(3.0)*x*y,
        (5.0*z*z*z-3.0*z*r2)/2.0,
        math.sqrt(3.0/8.0)*x*(5.0*z*z-r2),
        math.sqrt(3.0/8.0)*y*(5.0*z*z-r2),
        math.sqrt(15.0)*z*(x*x-y*y)/2.0, math.sqrt(15.0)*x*y*z,
        math.sqrt(10.0)*x*(x*x-3.0*y*y)/4.0,
        math.sqrt(10.0)*y*(3.0*x*x-y*y)/4.0,
    ])


def _independent(points, weights, partition, sites, tau):
    out = np.zeros((len(sites)*16, tau.shape[1]))
    for p, (point, weight) in enumerate(zip(points, weights)):
        for a, site in enumerate(sites):
            out[a*16:(a+1)*16] += (
                weight * partition[p, a] *
                _regular(np.asarray(point)-np.asarray(site))[:, None] * tau[p][None, :]
            )
    return out


def test_one_site_exact_and_rank_component_sentinels():
    points = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    weights = [2.0, 3.0, 5.0]
    tau = np.eye(3)
    actual, meta = _project(points, weights, [[1.0], [1.0], [1.0]], [[0, 0, 0]], tau)
    expected = _independent(points, weights, np.ones((3, 1)), [[0, 0, 0]], tau)
    assert actual == pytest.approx(expected, abs=0.0)
    assert meta["component_order"] == "00;10,11c,11s;20,21c,21s,22c,22s;30,31c,31s,32c,32s,33c,33s"
    # Reviewed rank-1 convention is z,x,y, not Cartesian x,y,z.
    np.testing.assert_array_equal(actual[1:4], [[0, 0, 15], [2, 0, 0], [0, 6, 0]])


def test_partitioned_sites_translate_back_to_molecular_common_origin():
    points = [[-0.8, 0.2, 0.1], [0.1, -0.5, 0.7], [1.1, 0.4, -0.3], [0.3, 0.8, 0.9]]
    weights = [0.7, 1.1, 0.4, 0.9]
    sites = [[-0.4, 0.1, 0.0], [0.6, -0.2, 0.3]]
    partition = np.array([[0.8, 0.2], [0.45, 0.55], [0.1, 0.9], [0.6, 0.4]])
    tau = np.array([[0.3, -0.2], [1.2, 0.5], [-0.7, 0.9], [0.4, -1.1]])
    atomic, _ = _project(points, weights, partition, sites, tau)
    origin, _ = _project(points, weights, np.ones((4, 1)), [[0, 0, 0]], tau)
    translated = np.zeros_like(origin)
    for a, site in enumerate(sites):
        for k in range(tau.shape[1]):
            translated[:, k] += psi4.core._atomic_polarizability_translate_l3(
                atomic[a*16:(a+1)*16, k].tolist(), site
            )
    assert translated == pytest.approx(origin, abs=2.0e-13)


def test_rank_zero_orthogonal_synthetic_transition_and_permutation_covariance():
    points = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
    weights = [1.0, 2.0, 1.0]
    sites = [[-0.5, 0, 0], [0.5, 0, 0]]
    partition = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    tau = np.array([[1.0, -0.5], [-1.0, 0.5], [1.0, -0.5]])
    base, _ = _project(points, weights, partition, sites, tau)
    assert base[0] + base[16] == pytest.approx([0.0, 0.0], abs=3.0e-17)

    gp = [2, 0, 1]
    sp = [1, 0]
    permuted, _ = _project(
        [points[p] for p in gp], [weights[p] for p in gp],
        partition[gp][:, sp], [sites[a] for a in sp], tau[gp],
    )
    assert permuted[:16] == pytest.approx(base[16:], abs=2.0e-14)
    assert permuted[16:] == pytest.approx(base[:16], abs=2.0e-14)


@pytest.mark.parametrize(
    "points,weights,partition,sites,tau,message",
    [
        ([[0, 0, 0]], [], [[1]], [[0, 0, 0]], [[1]], "dimensions"),
        ([[0, 0, 0]], [1], [[-0.1, 1.1]], [[0, 0, 0], [1, 0, 0]], [[1]], "nonnegative"),
        ([[0, 0, 0]], [1], [[0.2, 0.7]], [[0, 0, 0], [1, 0, 0]], [[1]], "unity"),
        ([[math.nan, 0, 0]], [1], [[1]], [[0, 0, 0]], [[1]], "finite"),
        ([[1.0e308, 0, 0]], [1], [[1]], [[0, 0, 0]], [[1]], "finite|overflow"),
        ([[0, 0, 0]], [1], [[1]], [[0, 0, 0]], [[math.inf]], "finite"),
    ],
)
def test_pure_projection_rejects_malformed_inputs(points, weights, partition, sites, tau, message):
    with pytest.raises(RuntimeError, match=message):
        _project(points, weights, partition, sites, tau)


def test_projection_planner_rejects_site_work_overflow_and_memory_envelopes():
    estimate = psi4.core._atomic_polarizability_estimate_transition_multipole_projection
    with pytest.raises(RuntimeError, match="site count"):
        estimate(10, 65, 2, 5, 0, 0, 1 << 30)
    with pytest.raises(RuntimeError, match="work bound"):
        estimate(10_000_000, 64, 512, 10, 0, 0, 1 << 62)
    with pytest.raises(RuntimeError, match="overflow"):
        estimate((1 << 64)-1, 2, 2, 1, 0, 0, (1 << 64)-1)
    with pytest.raises(RuntimeError, match="memory"):
        estimate(100, 4, 20, 100, 0, 0, 1024)


def _cartesian_to_real_l3(cart):
    """Explicit Mints Cartesian powers -> reviewed regular harmonics through rank 3."""
    x, y, z = cart[0:3]
    xx, xy, xz, yy, yz, zz = cart[3:9]
    xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz = cart[9:19]
    return np.array([
        (2*zz-xx-yy)/2,
        math.sqrt(3)*xz, math.sqrt(3)*yz, math.sqrt(3)*(xx-yy)/2, math.sqrt(3)*xy,
        (2*zzz-3*xxz-3*yyz)/2,
        math.sqrt(3/8)*(4*xzz-xxx-xyy),
        math.sqrt(3/8)*(4*yzz-xxy-yyy),
        math.sqrt(15)*(xxz-yyz)/2, math.sqrt(15)*xyz,
        math.sqrt(10)*(xxx-3*xyy)/4,
        math.sqrt(10)*(3*xxy-yyy)/4,
    ])


def test_cartesian_to_real_transform_is_explicit_through_rank3():
    p = np.array([0.7, -0.4, 1.2])
    x, y, z = p
    cart = np.array([x, y, z, x*x, x*y, x*z, y*y, y*z, z*z,
                     x**3, x*x*y, x*x*z, x*y*y, x*y*z, x*z*z,
                     y**3, y*y*z, y*z*z, z**3])
    transformed = np.concatenate(([1.0, z, x, y], _cartesian_to_real_l3(cart)))
    assert transformed == pytest.approx(_regular(p), abs=3.0e-15)


@pytest.fixture(scope="module")
def frozen_h2o_projection_context():
    psi4.core.be_quiet()
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk", "reference": "rhf",
                      "dft_spherical_points": 302, "dft_radial_points": 50,
                      "dft_density_tolerance": 1.0e-12, "dft_grac_shift": 0.0})
    neutral = psi4.geometry("""
        0 1
        O 0 0 0
        H 0 0.757160 0.586260
        H 0 -0.757160 0.586260
        symmetry c1
        units angstrom
    """)
    _, precursor = psi4.energy("pbe0", molecule=neutral, return_wfn=True)
    cation = psi4.geometry("""
        1 2
        O 0 0 0
        H 0 0.757160 0.586260
        H 0 -0.757160 0.586260
        symmetry c1
        units angstrom
    """)
    psi4.set_options({"reference": "uhf", "dft_grac_shift": 0.0})
    _, cation_wfn = psi4.energy("pbe0", molecule=cation, return_wfn=True)
    homo = max(precursor.epsilon_a_subset("SO", "OCC").to_array().ravel())
    shift = cation_wfn.energy() - precursor.energy() + homo
    psi4.set_options({"reference": "rhf", "dft_grac_shift": shift})
    _, grac = psi4.energy("pbe0", molecule=neutral, return_wfn=True)
    psi4.set_options({"dft_grac_shift": 0.0})
    context = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation_wfn)
    other_context = psi4.core._atomic_polarizability_make_frozen_response_context(
        grac, precursor, cation_wfn)
    return context, grac, other_context


@pytest.mark.scf
def test_real_sto3g_h2o_common_origin_matches_independent_ao_multipoles(frozen_h2o_projection_context):
    context, grac, _ = frozen_h2o_projection_context
    nsite = context.summary()["site_count"]
    npoint = context.summary()["grid_point_count"]
    result = psi4.core._atomic_polarizability_test_project_transition_multipoles_context(
        context, context, [1.0/nsite] * (npoint*nsite))
    assert result["plan"]["algorithm"] == "SEALED_BLOCK_TAU_STREAM"
    assert "transition_values" not in result  # production never publishes or retains full-grid T
    atomic = np.asarray(result["values"])
    common = np.zeros((16, atomic.shape[1]))
    sites = np.asarray(grac.molecule().geometry())
    for a, site in enumerate(sites):
        for k in range(atomic.shape[1]):
            common[:, k] += psi4.core._atomic_polarizability_translate_l3(
                atomic[a*16:(a+1)*16, k].tolist(), site.tolist())

    occ = np.flatnonzero(np.asarray(grac.occupation_a()).ravel() == 1.0)
    vir = np.flatnonzero(np.asarray(grac.occupation_a()).ravel() == 0.0)
    C = np.asarray(grac.Ca())
    mints = psi4.core.MintsHelper(grac.basisset())
    overlap = np.asarray(mints.ao_overlap())
    cart = [np.asarray(m) for m in mints.ao_multipoles(3, [0.0, 0.0, 0.0])]
    # Mints electric multipoles include the electron charge (-1); B uses bare
    # regular-harmonic coordinate moments, so remove that sign at every rank.
    ao_real = [overlap, -cart[2], -cart[0], -cart[1]]
    # Mints order is x,y,z; xx,xy,xz,yy,yz,zz; then the ten degree-3 monomials.
    stacked = -np.stack(cart)
    ao_real.extend(_cartesian_to_real_l3(stacked))
    oracle = np.array([
        [C[:, i] @ operator @ C[:, a] for i in occ for a in vir]
        for operator in ao_real
    ])
    assert common == pytest.approx(oracle, abs=1.0e-6)
    # Quadrature approximates MO orthogonality; it is intentionally not forced to exact zero.
    assert np.max(np.abs(common[0])) < 1.0e-7


@pytest.mark.scf
def test_context_projection_rejects_isa_from_different_context(frozen_h2o_projection_context):
    context, _, other_context = frozen_h2o_projection_context
    nsite = context.summary()["site_count"]
    npoint = context.summary()["grid_point_count"]
    with pytest.raises(RuntimeError, match="ISA weights.*same frozen response context"):
        psi4.core._atomic_polarizability_test_project_transition_multipoles_context(
            context, other_context, [1.0/nsite] * (npoint*nsite))
