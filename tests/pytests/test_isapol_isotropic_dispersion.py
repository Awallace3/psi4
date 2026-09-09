"""Supplied-model isotropic identities; no anisotropic/native/localization claim."""
import numpy as np
import pytest
import psi4

c = psi4.core


def site(label="A", ranks=(1, 2, 3, 4), values=None):
    s = c.IsaIsotropicSite()
    s.label, s.origin, s.ranks = label, [0., 0., 0.], list(ranks)
    s.polarizabilities = c.Matrix.from_array(np.ones((2, len(ranks))) if values is None else np.asarray(values, dtype=float))
    return s


def model(sites=None, frequencies=(0., 1.)):
    return c.IsaIsotropicModel(list(frequencies), [site()] if sites is None else sites, "supplied test model")


def coefficients(a=None, b=None, weights=(0., 1.), order=12):
    result = c.isa_isotropic_dispersion(a or model(), b or model(), list(weights), order)
    return result.pairs[0].coefficients


def test_independent_rank_factors_and_coverage():
    # Independent low-rank isotropic identities, cp includes 1/(2*pi).
    expected = {6: 6., 8: 15.+15., 10: 28.+70.+28., 12: 45.+210.+210.+45.}
    for x in coefficients():
        assert x.value == expected[x.order]
        assert x.complete and x.missing_rank_pairs == []
        assert len(x.included_rank_pairs) == x.order//2-2


@pytest.mark.parametrize("ranks,complete,values", [
    ((1,), [True, False, False, False], [6., 0., 0., 0.]),
    ((1, 2), [True, True, False, False], [6., 30., 70., 0.]),
    ((1, 2, 3), [True, True, True, False], [6., 30., 126., 420.]),
    ((1, 2, 3, 4), [True]*4, [6., 30., 126., 510.]),
])
def test_truncated_models(ranks, complete, values):
    m = model([site(ranks=ranks)])
    result = coefficients(m, m)
    assert [x.complete for x in result] == complete
    assert [x.value for x in result] == values
    for x in result:
        required = {(la, x.order//2-1-la) for la in range(1, x.order//2-1)}
        included = set(map(tuple, x.included_rank_pairs))
        missing = set(map(tuple, x.missing_rank_pairs))
        assert included | missing == required and not included & missing


def test_different_models_exchange_and_all_site_pairs():
    a = model([site("O", (1, 2), [[10., 20.], [2., 3.]]), site("H", (1,), [[5.], [4.]])])
    b = model([site("X", (1, 3), [[1., 2.], [7., 11.]])])
    ab = c.isa_isotropic_dispersion(a, b, [0., .5])
    ba = c.isa_isotropic_dispersion(b, a, [0., .5])
    assert [(p.site_a, p.site_b) for p in ab.pairs] == [(0, 0), (1, 0)]
    assert ab.labels_a == ["O", "H"] and ab.labels_b == ["X"]
    assert ab.method == "supplied_local_isotropic" and ab.units == "atomic_units"
    for p, reverse in zip(ab.pairs, ba.pairs):
        for x, y in zip(p.coefficients, reverse.coefficients):
            assert x.value == y.value
            assert set(map(tuple, x.included_rank_pairs)) == {(lb, la) for la, lb in y.included_rank_pairs}
    assert ab.pairs[0].coefficients[0].value == 6*.5*2*7
    assert ab.pairs[0].coefficients[1].value == 15*.5*3*7
    assert not ab.pairs[0].coefficients[1].complete


def test_zero_available_rank_is_not_missing():
    a = model([site(ranks=(1, 2), values=np.zeros((2, 2)))])
    result = coefficients(a, a)
    assert result[1].value == 0 and result[1].complete
    assert result[2].value == 0 and not result[2].complete
    assert result[2].included_rank_pairs == [[2, 2]]


def test_lorentz_oscillator_analytic_integral():
    # Integral_0^inf [a wa^2/(wa^2+w^2)] [b wb^2/(wb^2+w^2)] dw
    # = pi*a*b*wa*wb/(2*(wa+wb)). Use independent high-order mapped quadrature.
    nodes, weights = np.polynomial.legendre.leggauss(80)
    omega = (1+nodes)/(1-nodes)
    cp = weights/(np.pi*(1-nodes)**2)
    aa, bb, wa, wb = 2.3, 4.1, .8, 1.7
    ranks = np.arange(1, 5)
    avals = aa*wa**2/(wa**2+omega[:, None]**2)*ranks[None, :]
    bvals = bb*wb**2/(wb**2+omega[:, None]**2)*(ranks[None, :]+1)
    a = model([site(values=avals)], omega)
    b = model([site(values=bvals)], omega)
    integral = aa*bb*wa*wb/(4*(wa+wb))  # includes 1/(2*pi)
    factors = [[(1, 1, 6)], [(1, 2, 15), (2, 1, 15)],
               [(1, 3, 28), (2, 2, 70), (3, 1, 28)],
               [(1, 4, 45), (2, 3, 210), (3, 2, 210), (4, 1, 45)]]
    for x, terms in zip(coefficients(a, b, cp), factors):
        np.testing.assert_allclose(x.value, integral*sum(la*(lb+1)*f for la, lb, f in terms), rtol=1e-14)


def test_model_ownership_static_exclusion_and_signed_values():
    s = site(ranks=(1,), values=[[1.e308], [-2.]])
    a = model([s])
    s.polarizabilities.np[:] = 99.
    a.sites[0].polarizabilities.np[:] = 88.
    result = coefficients(a, model([site(ranks=(1,))]), order=6)
    assert result[0].value == -12.
    assert result[0].complete


@pytest.mark.parametrize("frequencies", [[], [-1., 1.], [0., np.nan], [0., np.inf], [1., 0.], [1., 1.]])
def test_invalid_frequencies(frequencies):
    with pytest.raises(ValueError):
        model(frequencies=frequencies)


@pytest.mark.parametrize("ranks", [[], [0], [5], [1, 1], [2, 1]])
def test_invalid_ranks(ranks):
    with pytest.raises(ValueError):
        model([site(ranks=ranks)])


@pytest.mark.parametrize("kind", ["null", "shape", "nan", "symmetry", "label", "origin", "duplicate", "empty", "provenance"])
def test_invalid_model(kind):
    s = site()
    if kind == "null": s.polarizabilities = None
    elif kind == "shape": s.polarizabilities = c.Matrix.from_array(np.ones((3, 4)))
    elif kind == "nan": s.polarizabilities.np[0, 0] = np.nan
    elif kind == "symmetry":
        dims = c.Dimension.from_list([2, 2])
        s.polarizabilities = c.Matrix("symmetry", dims, dims)
    elif kind == "label": s.label = ""
    elif kind == "origin": s.origin = [np.inf, 0., 0.]
    with pytest.raises(ValueError):
        c.IsaIsotropicModel([0., 1.], [] if kind == "empty" else [s, s] if kind == "duplicate" else [s],
                            "" if kind == "provenance" else "invalid test")


@pytest.mark.parametrize("weights", [[], [0.], [0., -1.], [1., 1.], [0., np.nan], [0., np.inf], [0., 0.]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        coefficients(weights=weights)


@pytest.mark.parametrize("order", [4, 7, 14])
def test_invalid_orders(order):
    with pytest.raises(ValueError):
        coefficients(order=order)


def test_mismatched_frequencies():
    with pytest.raises(ValueError, match="match exactly"):
        coefficients(b=model(frequencies=[0., 1.00000001]))


@pytest.mark.parametrize("kind", ["quadrature_sum", "rank_factor"])
def test_finite_sum_and_coefficient_overflow(kind):
    if kind == "quadrature_sum":
        a = model([site(ranks=(1,), values=[[1.e308], [1.e308]])], [1., 2.])
        b = model([site(ranks=(1,))], [1., 2.])
        weights = [1., 1.]
    else:
        a = model([site(ranks=(1,), values=[[1.], [4.e307]])])
        b = model([site(ranks=(1,))])
        weights = [0., 1.]
    with pytest.raises(ValueError, match="quadrature sum" if kind == "quadrature_sum" else "dispersion coefficient"):
        coefficients(a, b, weights)


def test_two_by_two_pairs_and_exchange_indices():
    a = model([site("A0", (1,), [[0.], [2.]]), site("A1", (1,), [[0.], [3.]])])
    b = model([site("B0", (1,), [[0.], [5.]]), site("B1", (1,), [[0.], [7.]])])
    ab = c.isa_isotropic_dispersion(a, b, [0., 1.], 6)
    ba = c.isa_isotropic_dispersion(b, a, [0., 1.], 6)
    reverse = {(p.site_a, p.site_b): p.coefficients[0].value for p in ba.pairs}
    assert len(ab.pairs) == len(reverse) == 4
    for p in ab.pairs:
        expected = 6*[2., 3.][p.site_a]*[5., 7.][p.site_b]
        assert p.coefficients[0].value == expected == reverse[p.site_b, p.site_a]


def test_metadata_and_result_ownership():
    frequencies, weights = [0., 1.], [0., 1.]
    s = site()
    s.origin = [1., 2., 3.]
    a = c.IsaIsotropicModel(frequencies, [s], "snapshot A")
    b = model()
    result = c.isa_isotropic_dispersion(a, b, weights)
    frequencies[1] = 2.
    weights[1] = 100.
    s.label, s.origin, s.ranks = "changed", [4., 5., 6.], [1]
    a.sites[0].label = "changed snapshot"
    result.labels_a[0] = "changed list"
    result.origins_a[0][0] = -100.
    del a, b, s
    assert result.frequencies == [0., 1.] and result.cp_weights == [0., 1.]
    assert result.labels_a == ["A"] and result.origins_a == [[1., 2., 3.]]
    assert result.provenance_a == "snapshot A"
    assert result.pairs[0].coefficients[0].value == 6.


@pytest.mark.parametrize("order", [8, 10])
def test_requested_order_truncation(order):
    result = coefficients(order=order)
    assert [x.order for x in result] == list(range(6, order+1, 2))


def test_additional_invalid_models():
    with pytest.raises(ValueError, match="dimensions"):
        model([site(values=np.ones((2, 3)))])
    with pytest.raises(ValueError, match="Nonfinite"):
        model([site(values=np.full((2, 4), np.inf))])
    a = model([site(values=np.ones((1, 4)))], [0.])
    with pytest.raises(ValueError):
        coefficients(a, a, [0.])


def test_scalar_trace_normalization_rank_blocks():
    # Supplied isotropic Racah block alpha_ll=alpha_l I_(2l+1) gives
    # alpha_l on trace averaging, independently of block dimension.
    scalars = [2., 3., 5., 7.]
    blocks = [np.eye(2*l+1)*scalars[l-1] for l in range(1, 5)]
    extracted = [np.trace(block)/(2*l+1) for l, block in enumerate(blocks, 1)]
    a = model([site(values=[extracted, extracted])])
    result = coefficients(a, a)
    expected = [6*2**2, 30*2*3, 56*2*5+70*3**2, 90*2*7+420*3*5]
    np.testing.assert_allclose([x.value for x in result], expected, rtol=0., atol=0.)


def test_finite_overflow():
    a = model([site(values=np.full((2, 4), 1.e200))])
    with pytest.raises(ValueError, match="Nonfinite"):
        coefficients(a, a)
