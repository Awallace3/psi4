"""Independent analytic supplied-local tests; no native/localization acceptance.

No reference tables, finite-difference derivatives, generated production targets,
binding skips, or direction-only substitutes for independent molecular averaging.
Interaction tolerances allow degree-eight polynomial cancellation and double output;
rotation tolerances additionally cover several harmonic transforms and signed sums.
They are fixed a priori, not fitted to an executed result.
"""
import itertools
import math

import numpy as np
import pytest
import psi4

c = psi4.core
DECL = "supplied_local_response"


def site(label="A", ranks=(1,), origin=(0., 0., 0.), frame=None, responses=None):
    s = c.IsaAnisotropicSite()
    s.label, s.ranks, s.origin = label, list(ranks), list(origin)
    s.frame = np.eye(3) if frame is None else frame
    n = sum(2*l+1 for l in ranks)
    s.responses = [c.Matrix.from_array(np.asarray(x, dtype=float))
                   for x in ([np.eye(n)] if responses is None else responses)]
    return s


def model(sites, frequencies=(1.,), provenance="analytic supplied test"):
    return c.IsaAnisotropicModel(list(frequencies), sites, DECL, provenance)


def evaluate(a, b, weights=(1.,), order=12):
    return c.isa_anisotropic_dispersion(a, b, list(weights), order)


def values(result, pair=0):
    return np.array([x.value for x in result.pairs[pair].coefficients])


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


def dipole_t(r):
    r = np.asarray(r, dtype=float)
    distance = np.linalg.norm(r)
    u = r[[2, 0, 1]]/distance
    return (np.eye(3)-3*np.outer(u, u))/distance**3


@pytest.mark.parametrize("r", [[0., 0., 2.], [.7, -.9, 1.3], [-2., .4, -.3]])
def test_general_dipole_closed_form_and_offdiagonal(r):
    aa = np.array([[2., .7, -.4], [.7, -1., .2], [-.4, .2, 3.]])
    bb = np.array([[1., -.3, .8], [-.3, 4., -.6], [.8, -.6, 2.]])
    t = dipole_t(r)
    np.testing.assert_allclose(c.isa_anisotropic_interaction(1, 1, r).np, t, rtol=2e-14, atol=2e-15)
    result = evaluate(model([site(responses=[aa])]), model([site("B", origin=r, responses=[bb])]))
    expected = np.trace(aa@t@bb@t.T)
    np.testing.assert_allclose(values(result)[0], expected*np.linalg.norm(r)**6, rtol=2e-14)
    np.testing.assert_allclose(result.truncated_energy, -expected, rtol=2e-14)
    np.testing.assert_array_equal(values(result)[1:], 0.)


@pytest.mark.parametrize("l,k", itertools.product(range(1, 5), repeat=2))
def test_axial_binomial_sign_exchange_and_physical_scaling(l, k):
    # H_l0(grad)H_k0(grad) on the axis gives (-1)^k choose(l+k,l).
    radius = 1.7
    actual = c.isa_anisotropic_interaction(l, k, [0., 0., radius]).np
    expected = (-1)**k*math.comb(l+k, l)/radius**(l+k+1)
    np.testing.assert_allclose(actual[0, 0], expected, rtol=3e-13, atol=2e-15)
    np.testing.assert_allclose(c.isa_anisotropic_interaction(k, l, [0., 0., -radius]).np.T,
                               actual, rtol=3e-13, atol=3e-14)
    r = np.array([.4, -.8, .6])
    first = c.isa_anisotropic_interaction(l, k, r).np
    second = c.isa_anisotropic_interaction(l, k, 3*r).np
    np.testing.assert_allclose(second*3**(l+k+1), first, rtol=4e-12, atol=3e-12)


def test_offaxis_dipole_quadrupole_cartesian_third_derivative():
    # d_i d_j d_k (1/r) = [-15 u_i u_j u_k +
    # 3(delta_ij u_k + delta_ik u_j + delta_jk u_i)] / r^4.
    # H_20=(2z^2-x^2-y^2)/2; H_21c=sqrt(3)xz, etc.
    r = np.array([.6, -.4, .9])
    radius = np.linalg.norm(r)
    u = r/radius
    third = np.empty((3, 3, 3))
    for i, j, k in itertools.product(range(3), repeat=3):
        third[i, j, k] = (-15*u[i]*u[j]*u[k] +
                          3*((i == j)*u[k]+(i == k)*u[j]+(j == k)*u[i]))/radius**4
    expected = []
    for i in [2, 0, 1]:
        expected.append(-np.array([third[i, 2, 2]-(third[i, 0, 0]+third[i, 1, 1])/2,
                                  np.sqrt(3)*third[i, 0, 2], np.sqrt(3)*third[i, 1, 2],
                                  np.sqrt(3)/2*(third[i, 0, 0]-third[i, 1, 1]),
                                  np.sqrt(3)*third[i, 0, 1]])/3)
    np.testing.assert_allclose(c.isa_anisotropic_interaction(1, 2, r).np, expected,
                               rtol=3e-14, atol=3e-15)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_axial_odd_orders_multiplicity_and_reversal(l):
    # Only A_l0,(l+1)0 = A_(l+1)0,l0 = x and B_10,10 = b survive.
    # Two ordered terms, tau_l0,10=-(l+1), tau_(l+1)0,10=-(l+2).
    x, b = .7, 1.3
    n = (2*l+1)+(2*l+3)
    aa = np.zeros((n, n)); aa[0, 2*l+1] = aa[2*l+1, 0] = x
    bb = np.diag([b, 0., 0.])
    a = model([site(ranks=(l, l+1), responses=[aa])])
    for sign in [1, -1]:
        result = evaluate(a, model([site("B", origin=(0., 0., sign*2.), responses=[bb])]))
        expected = np.zeros(7); expected[2*l-1] = sign*2*x*b*(l+1)*(l+2)
        np.testing.assert_allclose(values(result), expected, rtol=3e-13, atol=2e-14)
        assert result.pairs[0].coefficients[2*l-1].order == 2*l+5


def test_explicit_c7_both_mixed_blocks_and_exchange():
    aa = np.zeros((8, 8)); bb = np.zeros((8, 8))
    aa[0, 0], bb[0, 0] = 2., 3.
    aa[0, 3] = aa[3, 0] = .4
    bb[0, 3] = bb[3, 0] = -.7
    a = model([site(ranks=(1, 2), responses=[aa])])
    b = model([site("B", ranks=(1, 2), origin=(0., 0., 2.), responses=[bb])])
    # tau11=-2, tau12=+3, tau21=-3. Each mixed block occurs twice.
    expected_c7 = 12*(.4*3-2*(-.7))
    np.testing.assert_allclose(values(evaluate(a, b))[1], expected_c7, rtol=2e-14)
    np.testing.assert_allclose(values(evaluate(a, b)), values(evaluate(b, a)), rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("l,k", itertools.product(range(1, 5), repeat=2))
def test_frobenius_binomial_identity(l, k):
    # Addition theorem plus Coulomb derivatives: ||tau_lk||_F^2=choose(2l+2k,2l).
    u = np.array([.31, -.57, .76]); u /= np.linalg.norm(u)
    tau = c.isa_anisotropic_interaction(l, k, u).np
    np.testing.assert_allclose(np.sum(tau*tau), math.comb(2*l+2*k, 2*l), rtol=5e-13)


def test_scalar_rank_blocks_existing_isotropic_c6_through_c12():
    avals, bvals = [2., 3., 5., 7.], [11., 13., 17., 19.]
    def dense(v):
        return np.diag(np.repeat(v, [3, 5, 7, 9]))
    a = model([site(ranks=(1, 2, 3, 4), responses=[dense(avals)])])
    b = model([site("B", ranks=(1, 2, 3, 4), origin=(.7, -.3, 1.1), responses=[dense(bvals)])])
    result = evaluate(a, b, [.25])
    expected = [.25*6*2*11, 0., .25*15*(2*13+3*11), 0.,
                .25*(28*(2*17+5*11)+70*3*13), 0.,
                .25*(45*(2*19+7*11)+210*(3*17+5*13))]
    np.testing.assert_allclose(values(result), expected, rtol=5e-13, atol=2e-12)
    def isotropic(v, label):
        s = c.IsaIsotropicSite()
        s.label, s.ranks = label, [1, 2, 3, 4]
        s.polarizabilities = c.Matrix.from_array(np.array([v]))
        return c.IsaIsotropicModel([1.], [s], "independent scalar comparison")
    iso = c.isa_isotropic_dispersion(isotropic(avals, "A"), isotropic(bvals, "B"), [.25])
    np.testing.assert_allclose(values(result)[::2], [x.value for x in iso.pairs[0].coefficients], rtol=5e-13)


def test_noncommuting_frames_global_rotation_all_pairs_exchange():
    rng = np.random.default_rng(907)
    aa = rng.normal(size=(24, 24)); aa = aa+aa.T  # exactly symmetric, indefinite
    bb = rng.normal(size=(8, 8)); bb = bb+bb.T
    f = rotation([1, 2, -1], .73); g = rotation([-2, 1, 3], -.39)
    h = rotation([2, -1, 1], .52)
    assert not np.allclose(f@g, g@f)
    def models(global_frame):
        sa = [site("A0", (1, 2, 3, 4), global_frame@np.array([.1, .2, -.3]), global_frame@f, [aa]),
              site("A1", (1, 2, 3, 4), global_frame@np.array([-.4, .7, -.1]), global_frame@g, [aa*.3])]
        sb = [site("B0", (1, 2), global_frame@np.array([1.2, -.7, .9]), global_frame@g, [bb]),
              site("B1", (1, 2), global_frame@np.array([.6, -.4, 1.8]), global_frame@f, [bb*.7])]
        return model(sa), model(sb)
    a, b = models(np.eye(3))
    ab = evaluate(a, b); rotated = evaluate(*models(h)); ba = evaluate(b, a)
    reverse = {(p.site_b, p.site_a): p for p in ba.pairs}
    assert [(p.site_a, p.site_b) for p in ab.pairs] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for p, q in zip(ab.pairs, rotated.pairs):
        r = reverse[p.site_a, p.site_b]
        np.testing.assert_allclose([x.value for x in p.coefficients], [x.value for x in q.coefficients],
                                   rtol=3e-11, atol=3e-10)
        np.testing.assert_allclose([x.value for x in p.coefficients], [x.value for x in r.coefficients],
                                   rtol=3e-12, atol=3e-11)
        np.testing.assert_allclose(q.displacement, h@p.displacement, atol=5e-16)
        for x, y in zip(p.coefficients, r.coefficients):
            assert set(map(tuple, x.included_rank_quadruples)) == {(k, kp, l, lp) for l, lp, k, kp in y.included_rank_quadruples}
    np.testing.assert_allclose(ab.truncated_energy, rotated.truncated_energy, rtol=3e-11)


def test_dipole_frames_against_independent_cartesian_rotation():
    aa = np.array([[2., .3, -.7], [.3, 1., .8], [-.7, .8, -1.]])
    bb = np.array([[1., -.6, .2], [-.6, 4., -.3], [.2, -.3, 2.]])
    f = rotation([1, 2, 3], .61); g = rotation([3, -1, 2], -.32)
    r = np.array([.3, -.6, 1.2])
    df = f[np.ix_([2, 0, 1], [2, 0, 1])]; dg = g[np.ix_([2, 0, 1], [2, 0, 1])]
    tau = dipole_t(r)*np.linalg.norm(r)**3
    expected = np.trace((df@aa@df.T)@tau@(dg@bb@dg.T)@tau.T)
    actual = evaluate(model([site(frame=f, responses=[aa])]), model([site("B", origin=r, frame=g, responses=[bb])]))
    np.testing.assert_allclose(values(actual)[0], expected, rtol=3e-14, atol=3e-14)


def test_analytic_lorentz_cp_normalization():
    nodes, weights = np.polynomial.legendre.leggauss(80)
    omega = (1+nodes)/(1-nodes)
    cp = weights/(np.pi*(1-nodes)**2)  # Jacobian AND 1/(2*pi), not 1/pi
    wa, wb = .8, 1.7
    aa = np.array([[2., .4, -.2], [.4, 3., .7], [-.2, .7, 1.]])
    bb = np.array([[4., -.3, .1], [-.3, 1., .2], [.1, .2, 2.]])
    a = model([site(responses=[aa*wa**2/(wa**2+w*w) for w in omega])], omega)
    b = model([site("B", origin=(0., 0., 2.), responses=[bb*wb**2/(wb**2+w*w) for w in omega])], omega)
    tau = np.diag([-2., 1., 1.])
    # Integral with normalized CP weights is wa*wb/[4(wa+wb)].
    expected = wa*wb/(4*(wa+wb))*np.trace(aa@tau@bb@tau)
    np.testing.assert_allclose(values(evaluate(a, b, cp))[0], expected, rtol=2e-14)


def test_independent_a_b_orientation_average_anisotropic_and_mixed_blocks():
    # Exact Haar cubature for A: 5 equally spaced alpha and gamma angles
    # annihilate every nonzero Fourier mode |m|<=4; GL3 in cos(beta) integrates
    # surviving P_J through J=4. D_l alpha_ll' D_l'^T has J<=l+l'<=4.
    # B is a genuinely anisotropic DIAGONAL dipole tensor. Three cyclic axis
    # permutations have exactly its Haar mean trace(B)/3 I (state-specific
    # exact rule, not a general SO(3) quadrature). A and B choices are independent.
    # Bilinearity makes this 75x3 product rule exact for these tensors. Geometry
    # is fixed; this is NOT separation-direction averaging.
    rng = np.random.default_rng(37)
    aa = rng.normal(size=(8, 8)); aa = aa+aa.T+np.diag(np.arange(1., 9.))
    bb = np.diag([2., 5., 11.])
    z, w = np.polynomial.legendre.leggauss(3)
    a_sites, a_weights = [], []
    for i, j, k in itertools.product(range(5), range(3), range(5)):
        f = rotation([0, 0, 1], 2*np.pi*i/5)@rotation([0, 1, 0], np.arccos(z[j]))@rotation([0, 0, 1], 2*np.pi*k/5)
        a_sites.append(site(f"A{i}{j}{k}", (1, 2), frame=f, responses=[aa]))
        a_weights.append(w[j]/50)
    cycle = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    b_sites = [site(f"B{i}", origin=(.3, -.7, 1.1), frame=np.linalg.matrix_power(cycle, i), responses=[bb]) for i in range(3)]
    result = evaluate(model(a_sites), model(b_sites), order=8)
    mean = sum(a_weights[p.site_a]/3*np.array([x.value for x in p.coefficients]) for p in result.pairs)
    expected = [6*np.trace(aa[:3, :3])/3*np.trace(bb)/3, 0.,
                15*np.trace(aa[3:, 3:])/5*np.trace(bb)/3]
    # Absolute tolerance covers cancellation of mixed C7; relative tolerance
    # covers accumulation of 225 weighted pairs, not a stochastic error budget.
    np.testing.assert_allclose(mean, expected, rtol=2e-12, atol=2e-11)
    assert np.ptp([p.coefficients[0].value for p in result.pairs]) > 1.


@pytest.mark.parametrize("ranks", [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 4)])
def test_ordered_coverage_explicit_zeros_and_missing_theoretical_ranks(ranks):
    n = sum(2*l+1 for l in ranks)
    zero = np.zeros((n, n))
    result = evaluate(model([site(ranks=ranks, responses=[zero])]),
                      model([site("B", ranks=ranks, origin=(0., 0., 1.), responses=[zero])]))
    for x in result.pairs[0].coefficients:
        required = {q for q in itertools.product(range(1, x.order-4), repeat=4) if sum(q)+2 == x.order}
        included = {q for q in required if all(l in ranks for l in q)}
        assert set(map(tuple, x.included_rank_quadruples)) == included
        assert set(map(tuple, x.missing_rank_quadruples)) == required-included
        assert x.declared_model_complete and x.unrestricted_complete == (required == included)
        assert x.value == x.energy == 0.
    if ranks == (1, 2, 3, 4):
        assert [x.unrestricted_complete for x in result.pairs[0].coefficients] == [True]*4+[False]*3
        assert [7, 1, 1, 1] in result.pairs[0].coefficients[-1].missing_rank_quadruples
        assert [5, 1, 1, 1] in result.pairs[0].coefficients[4].missing_rank_quadruples


@pytest.mark.parametrize("order", range(6, 13))
def test_truncated_energy_and_requested_orders(order):
    aa = np.eye(8); aa[0, 3] = aa[3, 0] = .2
    a = model([site(ranks=(1, 2), responses=[aa])])
    b = model([site("B", origin=(0., 0., 2.))])
    result = evaluate(a, b, order=order)
    assert result.max_order == order and result.energy_is_truncated
    assert [x.order for x in result.pairs[0].coefficients] == list(range(6, order+1))
    expected = {6: 6., 7: 2*.2*2*3, 8: 15.}
    for x in result.pairs[0].coefficients:
        np.testing.assert_allclose(x.energy, -expected.get(x.order, 0.)/2**x.order, atol=1e-16)
    np.testing.assert_allclose(result.truncated_energy, sum(-v/2**n for n, v in expected.items() if n <= order), atol=1e-16)


def test_ownership_metadata_and_static_exclusion():
    f = rotation([1, 2, 3], .7)
    dynamic = np.diag([-2., 3., 1.])
    s = site(frame=f, responses=[np.full((3, 3), 1e308), dynamic])
    frequencies, weights = [0., 1.], [0., 1.]
    a = model([s], frequencies, "owned A provenance")
    b = model([site("B", origin=(0., 0., 2.), responses=[np.eye(3), np.eye(3)])], frequencies, "owned B provenance")
    result = evaluate(a, b, weights)
    expected = values(result).copy()
    s.responses[1].np[:] = 99.
    s.label, s.origin, s.ranks, s.frame = "changed", [1., 1., 1.], [4], np.zeros((3, 3))
    a.sites[0].responses[1].np[:] = 88.
    frequencies[1], weights[1] = 2., 3.
    result.model_a.sites[0].responses[1].np[:] = 77.
    result.model_a.sites[0].ranks = [2]
    result.pairs[0].direction[0] = 99.
    del a, b, s
    snap = result.model_a.sites[0]
    assert snap.label == "A" and snap.ranks == [1] and snap.components == ["10", "11c", "11s"]
    np.testing.assert_array_equal(snap.responses[1].np, dynamic)
    np.testing.assert_array_equal(snap.frame, f)
    assert result.model_a.provenance == "owned A provenance"
    assert result.model_b.provenance == "owned B provenance"
    assert result.model_a.declaration == DECL and result.model_a.units == "atomic_units"
    assert result.frequencies == [0., 1.] and result.cp_weights == [0., 1.]
    assert result.method == "supplied_local_anisotropic"
    assert result.coefficient_representation == "orientation_resolved_scalar"
    assert result.units == "atomic_units" and result.pairs[0].direction == [0., 0., 1.]
    np.testing.assert_array_equal(values(result), expected)
    # Getters really are isolated from the retained result, not merely original input.
    repeated = evaluate(result.model_a, result.model_b, result.cp_weights)
    np.testing.assert_array_equal(values(repeated), expected)


@pytest.mark.parametrize("frequencies", [[], [-1.], [np.nan], [np.inf], [1., 1.], [2., 1.]])
def test_invalid_frequencies(frequencies):
    with pytest.raises(ValueError):
        model([site()], frequencies)


@pytest.mark.parametrize("ranks", [[], [0], [5], [1, 1], [2, 1]])
def test_invalid_ranks(ranks):
    s = site(); s.ranks = ranks
    with pytest.raises(ValueError):
        model([s])


@pytest.mark.parametrize("frame", [np.zeros((3, 3)), np.diag([-1., 1., 1.]), np.eye(3)*2,
                                    [[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]],
                                    np.full((3, 3), np.nan), np.full((3, 3), np.inf)])
def test_invalid_frames(frame):
    with pytest.raises(ValueError):
        model([site(frame=frame)])


def test_frame_must_be_explicit_and_frame_shape():
    s = c.IsaAnisotropicSite()
    s.label, s.ranks, s.responses = "A", [1], [c.Matrix.from_array(np.eye(3))]
    with pytest.raises(ValueError):
        model([s])
    with pytest.raises((TypeError, ValueError)):
        s.frame = np.eye(2)


@pytest.mark.parametrize("kind", ["null", "nonsquare", "partial", "frequency_count", "symmetry_blocks",
                                  "asymmetric_ulp", "nan", "inf", "label", "origin", "duplicate", "empty"])
def test_invalid_models(kind):
    s = site()
    if kind == "null": s.responses = [None]
    elif kind == "nonsquare": s.responses = [c.Matrix.from_array(np.ones((3, 2)))]
    elif kind == "partial": s.ranks = [1, 2]  # cannot imply unsupplied blocks
    elif kind == "frequency_count": s.responses = []
    elif kind == "symmetry_blocks":
        dims = c.Dimension.from_list([1, 2]); s.responses = [c.Matrix("blocked", dims, dims)]
    elif kind == "asymmetric_ulp":
        m = np.eye(3); m[0, 1] = 1.; m[1, 0] = np.nextafter(1., 2.)
        s.responses = [c.Matrix.from_array(m)]
    elif kind in ["nan", "inf"]: s.responses[0].np[0, 0] = getattr(np, kind)
    elif kind == "label": s.label = " \t"
    elif kind == "origin": s.origin = [np.inf, 0., 0.]
    with pytest.raises(ValueError):
        model([] if kind == "empty" else [s, s] if kind == "duplicate" else [s])


@pytest.mark.parametrize("declaration,provenance", [("", "test"), ("distributed_response", "test"),
                                                    (DECL, ""), (DECL, " \n\t")])
def test_required_declaration_provenance(declaration, provenance):
    with pytest.raises(ValueError):
        c.IsaAnisotropicModel([1.], [site()], declaration, provenance)


@pytest.mark.parametrize("weights", [[], [0.], [-1.], [np.nan], [np.inf], [1., 2.]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        evaluate(model([site()]), model([site("B", origin=(0., 0., 1.))]), weights)


def test_static_weight_and_frequency_matching():
    a = model([site()], [0.]); b = model([site("B", origin=(0., 0., 1.))], [0.])
    for weights in [[0.], [1.]]:
        with pytest.raises(ValueError): evaluate(a, b, weights)
    with pytest.raises(ValueError, match="match exactly"):
        evaluate(model([site()]), model([site("B", origin=(0., 0., 1.))], [np.nextafter(1., 2.)]))


@pytest.mark.parametrize("order", [5, 13, -1])
def test_invalid_max_order(order):
    with pytest.raises(ValueError):
        evaluate(model([site()]), model([site("B", origin=(0., 0., 1.))]), order=order)


@pytest.mark.parametrize("l,k,r", [(0, 1, [0., 0., 1.]), (1, 5, [0., 0., 1.]),
                                   (1, 1, [0., 0., 0.]), (2, 3, [np.nan, 0., 1.]),
                                   (4, 4, [np.inf, 0., 1.]), (4, 4, [0., 0., 1e-40])])
def test_invalid_interaction(l, k, r):
    with pytest.raises(ValueError): c.isa_anisotropic_interaction(l, k, r)


def test_interaction_ownership_and_large_distance_conditioning():
    r = [0., 0., 1e30]
    t = c.isa_anisotropic_interaction(4, 4, r)
    expected = 70e-270
    np.testing.assert_allclose(t.np[0, 0], expected, rtol=3e-13, atol=0.)
    t.np[:] = 99.
    np.testing.assert_allclose(c.isa_anisotropic_interaction(4, 4, r).np[0, 0], expected, rtol=3e-13, atol=0.)


@pytest.mark.parametrize("kind", ["coincident", "displacement", "norm", "rotation", "product", "sum", "energy", "total"])
def test_nonfinite_intermediates_and_overflow(kind):
    sa, sb = site(), site("B", origin=(0., 0., 1.))
    weights, frequencies = [1.], [1.]
    if kind == "coincident": sb.origin = sa.origin
    elif kind == "displacement": sa.origin = [-1e308, 0., 0.]; sb.origin = [1e308, 0., 0.]
    elif kind == "norm": sb.origin = [1.1e308]*3
    elif kind == "rotation":
        sa.responses = [c.Matrix.from_array(np.full((3, 3), 1e308))]
        sa.frame = rotation([1, 2, 3], .7)
    elif kind == "product":
        sa.responses = [c.Matrix.from_array(np.eye(3)*1e200)]
        sb.responses = [c.Matrix.from_array(np.eye(3)*1e200)]
    elif kind == "sum":
        sa.responses = [c.Matrix.from_array(np.eye(3)*2e307)]*2
        sb.responses = [c.Matrix.from_array(np.eye(3))]*2
        weights, frequencies = [1., 1.], [1., 2.]
    elif kind == "energy": sb.origin = [0., 0., 1e-60]
    elif kind == "total": sa.responses = [c.Matrix.from_array(np.eye(3)*2e307)]
    aa = [sa]
    if kind == "total":
        second = site("A2", responses=[np.eye(3)*2e307]); aa.append(second)
    with pytest.raises(ValueError):
        evaluate(model(aa, frequencies), model([sb], frequencies), weights)


def test_sparse_rank_axes_and_nonstatic_zero_weight():
    # Only scalar rank2 blocks contribute below order12 for ranks (2,4).
    aa = np.diag([2.]*5+[7.]*9)
    bb = np.diag([3.]*5+[11.]*9)
    a = model([site(ranks=(2, 4), responses=[np.full((14, 14), 1e308), aa])], [1., 2.])
    b = model([site("B", ranks=(2, 4), origin=(.4, -.3, 1.), responses=[bb, bb])], [1., 2.])
    result = evaluate(a, b, [0., 1.])
    expected = np.zeros(7); expected[4] = 70*2*3
    np.testing.assert_allclose(values(result), expected, rtol=5e-13, atol=2e-12)
    assert result.model_a.sites[0].components == ["20", "21c", "21s", "22c", "22s",
        "40", "41c", "41s", "42c", "42s", "43c", "43s", "44c", "44s"]


def test_small_distance_representable_energy_and_norm():
    # Forming R^-6 first would overflow, although the actual energy is finite.
    a = model([site(responses=[np.eye(3)*1e-150])])
    b = model([site("B", origin=(0., 0., 1e-50), responses=[np.eye(3)*1e-150])])
    result = evaluate(a, b, order=6)
    np.testing.assert_allclose(result.truncated_energy, -6., rtol=2e-14)
    # Squaring the displacement in double would underflow; hypot must not.
    zero = model([site(responses=[np.zeros((3, 3))])])
    tiny = model([site("B", origin=(0., 0., 1e-200))])
    out = evaluate(zero, tiny)
    assert out.pairs[0].distance == 1e-200 and out.truncated_energy == 0.


def test_resource_limits_without_large_tensor_allocation():
    # Reuse one small input Matrix; the model must reject aggregate size before
    # cloning. The 24x24 rank4 model would exceed its 8M element ceiling.
    s = site(ranks=(1, 2, 3, 4))
    s.responses = [s.responses[0]]*13889
    with pytest.raises(ValueError, match="resource limit"):
        model([s], np.arange(1., 13890.))
    # 65x64 pairs exceed 4096, while individual owned inputs are tiny.
    a = model([site(f"A{i}") for i in range(65)])
    b = model([site(f"B{i}", origin=(0., 0., 1.)) for i in range(64)])
    with pytest.raises(ValueError, match="pair resource limit"):
        evaluate(a, b)
