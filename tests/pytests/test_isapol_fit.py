"""Frozen ISA-A kernel checks; these do NOT establish converged ISA water parity.

The water oracle runs extracted CamCASP arithmetic on identical supplied samples.
Its manifest explicitly distinguishes the diagnostic basis/AO density from the
production AtomAux/Drho-C protocol. No CamCASP installation is needed by pytest.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]
DATA = Path(__file__).parent / "data_isapol"


def matrix(array):
    return psi4.core.Matrix.from_array(np.asarray(array, dtype=float))


def small_data():
    d = psi4.core.IsaAFitData()
    d.weights = [0.2, 0.4, 0.3, 0.1, 0.5]
    d.density = [2., 1., -0.01, 0.5, 0.7]  # fitted densities can be signed
    d.shape = [0.5, 0.2, -0.1, 0.3, 0.4]  # active-tail branch need not clamp
    d.shape_sum = [1., 0., -0.5, 1e-36, 2.]
    d.radius_squared = [0., 1., 2., 3., 4.]
    d.basis_values = matrix([[1., 0.5, 0.3], [.4, .8, .2], [.2, .1, .5], [.1, .2, .3], [.5, .6, .2]])
    d.overlap = matrix([[2., 0., .4], [0., 3., 0.], [.4, 0., 1.]])
    d.previous = [1., -.2, -.1]
    d.angular_momenta = [0, 1, 0]  # noncontiguous s block
    d.exponents = [1., .1, .2]
    return d


def numpy_reference(d, o):
    weights, rho, wa, wsum, r2 = map(np.asarray, [d.weights, d.density, d.shape, d.shape_sum, d.radius_squared])
    selected = np.abs(wsum) > o.density_cutoff
    rhoa = np.zeros_like(rho)
    rhoa[selected] = rho[selected] * wa[selected] / wsum[selected]
    exp_weight = np.exp(np.minimum(o.w_eps * r2, 230.))
    s = np.asarray(d.angular_momenta) == 0
    factors = np.broadcast_to(rhoa[:, None], (len(weights), len(s))).copy()
    factors[:, s] += o.damping * wa[:, None]
    factors[:, s if o.s_block_only else np.ones_like(s)] *= exp_weight[:, None]
    rhs = np.einsum("p,pk,pk->k", weights, d.basis_values.np, factors)
    metric = d.overlap.np.copy()
    metric[np.ix_(s, s)] *= 1. + o.damping
    eligible = s & (np.asarray(d.exponents) <= o.positive_max_alpha)
    if o.positive_auto:
        eligible &= np.asarray(d.previous) < 0.
    metric[np.diag_indices_from(metric)] += o.positive_lambda * eligible
    return metric, rhs, np.linalg.solve(metric, rhs), weights @ rhoa


@pytest.mark.parametrize("s_only", [False, True])
@pytest.mark.parametrize("eps", [0., .17])
@pytest.mark.parametrize("damping", [0., .03])
@pytest.mark.parametrize("auto", [False, True])
def test_isa_fit_numpy(s_only, eps, damping, auto):
    d, o = small_data(), psi4.core.IsaAFitOptions()
    o.s_block_only, o.w_eps, o.damping, o.positive_auto = s_only, eps, damping, auto
    o.positive_lambda = .001
    o.positive_max_alpha = 1.
    before = d.overlap.np.copy()
    want_metric, want_rhs, want_coeff, want_population = numpy_reference(d, o)
    got = psi4.core.isa_a_fit_step(d, o)
    np.testing.assert_allclose(got.metric.np, want_metric, rtol=0, atol=1e-15)
    np.testing.assert_allclose(got.rhs.np[:, 0], want_rhs, rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(got.coefficients.np[:, 0], want_coeff, rtol=2e-14, atol=1e-15)
    assert got.population == pytest.approx(want_population, abs=1e-15)
    assert got.excluded_points == 2  # zero and exactly the cutoff are excluded
    assert got.relative_residual < 2e-15
    np.testing.assert_array_equal(d.overlap.np, before)
    got.metric.np[:] = 999.  # result access returns an owned copy
    np.testing.assert_allclose(got.metric.np, want_metric)


def test_isa_fit_exponent_cap_and_cutoff_damping():
    d, o = small_data(), psi4.core.IsaAFitOptions()
    d.weights = [1e-100] * 5
    d.radius_squared = [1e8] * 5
    d.shape_sum = [0.] * 5
    o.w_eps, o.damping = .17, .1
    got = psi4.core.isa_a_fit_step(d, o)
    _, rhs, _, _ = numpy_reference(d, o)
    assert got.excluded_points == 5
    assert got.population == 0.
    assert got.rhs.np[1, 0] == 0.  # non-s has no damping contribution
    assert got.rhs.np[0, 0] > 0.  # s damping is not suppressed by denominator cutoff
    np.testing.assert_allclose(got.rhs.np[:, 0], rhs, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("field,value,message", [
    ("density", [1.], "wrong size"),
    ("shape", [np.nan] * 5, "nonfinite"),
    ("radius_squared", [-1.] * 5, "nonnegative"),
    ("exponents", [1., 0., .2], "positive"),
    ("angular_momenta", [0, -1, 0], "nonnegative"),
    ("previous", [], "count"),
])
def test_isa_fit_rejects_invalid_samples(field, value, message):
    d = small_data()
    setattr(d, field, value)
    with pytest.raises(RuntimeError, match=message):
        psi4.core.isa_a_fit_step(d)


@pytest.mark.parametrize("field", ["w_eps", "damping", "positive_lambda", "positive_max_alpha", "density_cutoff"])
@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_isa_fit_rejects_invalid_options(field, value):
    o = psi4.core.IsaAFitOptions()
    setattr(o, field, value)
    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        psi4.core.isa_a_fit_step(small_data(), o)


def test_isa_fit_rejects_bad_matrices():
    for bad, message in [(None, "C1"), (matrix(np.eye(2)), "dimensions"),
                         (matrix([[1., .2, 0.], [0., 1., 0.], [0., 0., 1.]]), "symmetric"),
                         (matrix(1e-20 * np.array([[1., .9, 0.], [0., 1., 0.], [0., 0., 1.]])), "symmetric"),
                         (matrix(np.ones((3, 3))), "LU solve failed")]:
        d = small_data()
        d.overlap = bad
        with pytest.raises(RuntimeError, match=message):
            psi4.core.isa_a_fit_step(d)
    d = small_data()
    d.basis_values = matrix(np.ones((4, 3)))
    with pytest.raises(RuntimeError, match="basis_values.*dimensions"):
        psi4.core.isa_a_fit_step(d)


def test_isa_overlap_change():
    overlap = matrix([[2., .3], [.3, 1.]])
    current, old = np.array([.8, .3]), np.array([1., -.1])
    want = abs(1. - abs(current @ overlap.np @ old) /
               np.sqrt((current @ overlap.np @ current) * (old @ overlap.np @ old)))
    assert psi4.core.isa_overlap_change(current, old, overlap) == pytest.approx(want, abs=3e-16)
    # Important source convention: amplitude and sign changes are invisible to W convergence.
    for scale in [2., -1., 1e200, 1e-200]:
        assert psi4.core.isa_overlap_change(scale * old, old, overlap) < 5e-16
    for scale in [1e-308, 1., 1e308]:
        assert psi4.core.isa_overlap_change([1., 1.], [1., 1.], matrix(scale * np.eye(2))) < 5e-16
    with pytest.raises(RuntimeError, match="zero expansion"):
        psi4.core.isa_overlap_change([0., 0.], old, overlap)
    with pytest.raises(RuntimeError, match="wrong size"):
        psi4.core.isa_overlap_change([1.], old, overlap)


@pytest.fixture(scope="module")
def water():
    with np.load(DATA / "camcasp_isa_fit_water.npz") as archive:
        return {key: archive[key] for key in archive.files}


def test_water_fixture_provenance():
    manifest = json.loads((DATA / "camcasp_isa_fit_water.json").read_text())
    assert manifest["schema_version"] == 1
    assert "not converged ISA" in manifest["claim"]
    assert "not CamCASP Drho-C" in manifest["density"]
    assert hashlib.sha256((DATA / "camcasp_isa_fit_water.npz").read_bytes()).hexdigest() == manifest["fixture_sha256"]
    assert hashlib.sha256((DATA / "camcasp_isa_fit_edges.npz").read_bytes()).hexdigest() == manifest["edge_fixture_sha256"]


@pytest.mark.parametrize("case", range(3), ids=["inactive", "active-s-only", "active-all"])
@pytest.mark.parametrize("atom", range(3), ids=["O", "H1", "H2"])
def test_water_frozen_fit_camcasp(water, case, atom):
    """Compare a source-extracted update, not a self-generated NumPy oracle."""
    d = psi4.core.IsaAFitData()
    for field in ["weights", "density", "shape_sum", "exponents", "angular_momenta", "previous"]:
        setattr(d, field, water[field].tolist())
    d.shape = water["shape"][atom].tolist()
    d.radius_squared = water["radius_squared"][atom].tolist()
    d.basis_values = matrix(water["basis_values"][atom])
    d.overlap = matrix(water["overlap"][case])
    o = psi4.core.IsaAFitOptions()
    values = water["options"][case]
    o.w_eps, o.damping, o.positive_lambda, o.positive_max_alpha, o.density_cutoff = values[:5]
    o.s_block_only, o.positive_auto = map(bool, values[5:])
    got = psi4.core.isa_a_fit_step(d, o)
    reference = water["result"][case, atom]
    np.testing.assert_allclose(got.metric.np, reference[:, :5], rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(got.rhs.np[:, 0], reference[:, 5], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(got.coefficients.np[:, 0], reference[:, 6], rtol=2e-12, atol=2e-13)
    assert got.population == pytest.approx(water["population"][case, atom], rel=2e-13, abs=1e-14)
    assert got.excluded_points == np.count_nonzero(np.abs(water["shape_sum"]) <= o.density_cutoff)
    assert got.relative_residual < 2e-15


@pytest.mark.parametrize("case", range(4))
def test_isa_fit_source_edge_cases(case):
    """Source-backed branches not reached by the nonnegative water samples."""
    with np.load(DATA / "camcasp_isa_fit_edges.npz") as edge:
        d = psi4.core.IsaAFitData()
        for field in ["weights", "density", "shape", "shape_sum", "radius_squared", "previous",
                      "angular_momenta", "exponents"]:
            setattr(d, field, edge[field].tolist())
        d.basis_values, d.overlap = matrix(edge["basis_values"]), matrix(edge["overlap"])
        o = psi4.core.IsaAFitOptions()
        values = edge["options"][case]
        o.w_eps, o.damping, o.positive_lambda, o.positive_max_alpha, o.density_cutoff = values[:5]
        o.s_block_only, o.positive_auto = map(bool, values[5:])
        got = psi4.core.isa_a_fit_step(d, o)
        reference = edge["result"][case]
        np.testing.assert_allclose(got.metric.np, reference[:, :3], rtol=2e-15, atol=1e-15)
        np.testing.assert_allclose(got.rhs.np[:, 0], reference[:, 3], rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(got.coefficients.np[:, 0], reference[:, 4], rtol=2e-13, atol=1e-14)
        assert got.population == pytest.approx(edge["population"][case], rel=2e-14, abs=1e-15)
        assert got.excluded_points == 2
        assert got.relative_residual < 2e-15
        assert got.metric.np[0, 0] == pytest.approx(2. * 1.03 + (0. if o.positive_auto else .5))
