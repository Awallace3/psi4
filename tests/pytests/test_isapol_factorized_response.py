# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Small independent contractions; no archived molecular fixtures."""
import numpy as np
import pytest


def test_exchange_work_guard_counts_both_matrix_products(monkeypatch):
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    # Lower the test-only ceiling: old estimate=480, exchange alone=576 flops.
    monkeypatch.setattr(FactorizedDFOperators, "MAX_WORK", 500)
    ops = FactorizedDFOperators(np.ones(6), np.zeros((4, 2, 2)),
                                np.zeros((4, 2, 3)), np.zeros((4, 2, 3)),
                                np.zeros((4, 3, 3)), exact_exchange=1.)
    with pytest.raises(ValueError, match="work resource"):
        ops.apply_h1(np.ones((6, 1)))


@pytest.mark.parametrize("exchange", [0., .25, 1.])
@pytest.mark.parametrize("independent", [False, True])
def test_factorized_actions_match_explicit_four_index_contractions(exchange, independent):
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators

    rng = np.random.default_rng(17)
    nocc, nvir, naux = 2, 3, 4
    raw = rng.normal(size=(naux, nocc+nvir, nocc+nvir))
    mo = (raw+raw.transpose(0, 2, 1))/2
    metric = np.diag([1., 2., 3., 4.])
    dual = np.linalg.solve(metric, mo.reshape(naux, -1)).reshape(mo.shape)
    if independent:
        mo = raw[:, :, ::-1]  # noncontiguous and not constrained to symmetry
        dual = rng.normal(size=mo.shape)
    gaps = np.arange(1., 7.)
    operators = FactorizedDFOperators(
        gaps, mo[:, :nocc, :nocc], mo[:, :nocc, nocc:],
        dual[:, :nocc, nocc:], dual[:, nocc:, nocc:],
        exact_exchange=exchange)
    h1, h2 = np.diag(gaps), np.diag(gaps)
    # Explicit orbital loops and scalar auxiliary sums are the independent oracle.
    for a in range(nvir):
        for i in range(nocc):
            for b in range(nvir):
                for j in range(nocc):
                    v = sum(mo[p, i, nocc+a]*dual[p, j, nocc+b] for p in range(naux))
                    x = sum(mo[p, i, j]*dual[p, nocc+a, nocc+b] for p in range(naux))
                    y = sum(mo[p, i, nocc+b]*dual[p, j, nocc+a] for p in range(naux))
                    h1[a*nocc+i, b*nocc+j] += 4*v-exchange*(x+y)
                    h2[a*nocc+i, b*nocc+j] -= exchange*(x-y)
    rhs = np.column_stack((np.eye(6), rng.normal(size=(6, 3))))[:, ::-1]
    # Mutate every caller-owned factor after forming the independent oracle.
    mo[:] = 0.
    dual[:] = 0.
    gaps[:] = 0.
    np.testing.assert_allclose(operators.apply_h1(rhs), h1 @ rhs, atol=1e-12)
    np.testing.assert_allclose(operators.apply_h2(rhs), h2 @ rhs, atol=1e-12)


def test_action_byte_budget_is_checked_separately_from_factor_snapshots():
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    ops = FactorizedDFOperators(np.ones(2), np.ones((1, 1, 1)),
                                np.ones((1, 1, 2)), np.ones((1, 1, 2)),
                                np.ones((1, 2, 2)), exact_exchange=1., max_bytes=300)
    with pytest.raises(ValueError, match="action byte resource"):
        ops.apply_h1(np.ones((2, 1)))


@pytest.mark.parametrize("overflow", ["gap", "exchange", "kernel"])
def test_finite_operands_that_overflow_fail_closed(overflow):
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    huge = np.finfo(float).max
    gap = np.array([huge if overflow == "gap" else 1.])
    oo = np.array([[[huge if overflow == "exchange" else 0.]]])
    ov, vv = np.zeros((1, 1, 1)), np.full((1, 1, 1), 2.)
    kwargs = dict(exact_exchange=1.)
    if overflow == "kernel":
        kwargs.update(local_scale=1., kernel_legs=np.array([[2.]]),
                      auxiliary_kernel=np.array([[huge]]))
    ops = FactorizedDFOperators(gap, oo, ov, ov, vv, **kwargs)
    with pytest.raises((FloatingPointError, ValueError)):
        ops.apply_h1(np.array([[2.]]))


def test_auxiliary_kernel_action_preserves_h2_and_owned_operands():
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    gaps = np.array([1., 2., 3.])
    oo, ov, vv = np.zeros((2, 1, 1)), np.zeros((2, 1, 3)), np.zeros((2, 3, 3))
    legs = np.array([[1., 2.], [3., -1.], [.5, 4.]])
    kernel = np.array([[-.2, .03], [.03, -.1]])
    expected = np.diag(gaps) + 4*.75*legs @ kernel @ legs.T
    operators = FactorizedDFOperators(
        gaps, oo, ov, ov, vv, exact_exchange=.25,
        local_scale=.75, kernel_legs=legs, auxiliary_kernel=kernel)
    gaps[:] = 9.
    legs[:] = 0.
    kernel[:] = 0.
    np.testing.assert_allclose(operators.apply_h1(np.eye(3)), expected, atol=1e-14)
    np.testing.assert_array_equal(operators.apply_h2(np.eye(3)), np.diag([1., 2., 3.]))


@pytest.mark.parametrize("fault, message", [
    ("gaps", "positive"), ("shape", "dimensions"), ("finite", "finite"),
    ("budget", "byte resource"), ("exchange", "exact_exchange"),
    ("missing_kernel", "requires kernel"), ("half_kernel", "both kernel"),
])
def test_factorized_input_refusals(fault, message):
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    gaps = np.array([1., 2.])
    oo, ov, vv = np.ones((1, 1, 1)), np.ones((1, 1, 2)), np.ones((1, 2, 2))
    kwargs = dict(exact_exchange=.25)
    if fault == "gaps":
        gaps[0] = 0.
    elif fault == "shape":
        vv = np.ones((1, 3, 3))
    elif fault == "finite":
        ov[0, 0, 0] = np.nan
    elif fault == "budget":
        kwargs["max_bytes"] = 1
    elif fault == "exchange":
        kwargs["exact_exchange"] = np.inf
    elif fault == "missing_kernel":
        kwargs["local_scale"] = .75
    elif fault == "half_kernel":
        kwargs["kernel_legs"] = np.ones((2, 1))
    with pytest.raises(ValueError, match=message):
        FactorizedDFOperators(gaps, oo, ov, ov, vv, **kwargs)


def test_factorized_actions_refuse_invalid_rhs_and_return_owned_results():
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    ops = FactorizedDFOperators(np.array([1., 2.]), np.zeros((1, 1, 1)),
                                np.zeros((1, 1, 2)), np.zeros((1, 1, 2)),
                                np.zeros((1, 2, 2)), exact_exchange=0.)
    for rhs in (np.ones(2), np.ones((3, 1)), np.ones((2, 65)),
                np.full((2, 1), np.nan)):
        with pytest.raises(ValueError, match="rhs"):
            ops.apply_h1(rhs)
    result = ops.apply_h1(np.eye(2))
    result[:] = 0.
    np.testing.assert_array_equal(ops.apply_h1(np.eye(2)), np.diag([1., 2.]))
