# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Small independent contractions; no archived molecular fixtures."""
import numpy as np
import pytest


def test_action_kernel_composition_preserves_target_hessian_for_anchor_solves():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, KernelCorrectedDFOperators, factorized_projected_response)
    gaps = np.array([1., 2., 3.])
    zeros = np.zeros((1, 1, 1))
    plain = FactorizedDFOperators(
        gaps, zeros, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
        np.zeros((1, 3, 3)), exact_exchange=0.)
    target = np.array([[1., .1], [.2, .8], [.3, -.1]])
    kernel = np.array([[-.03, .01], [.01, -.02]])
    corrected = KernelCorrectedDFOperators(
        plain, target, kernel.__matmul__, local_scale=.75,
        kernel_storage_bytes=kernel.nbytes, kernel_workspace_bytes=128,
        kernel_work_per_rhs=8)
    expected_h1 = np.diag(gaps)+3*target @ kernel @ target.T
    np.testing.assert_allclose(corrected.apply_h1(np.eye(3)), expected_h1)
    np.testing.assert_array_equal(corrected.apply_h2(np.eye(3)), np.diag(gaps))
    np.testing.assert_array_equal(plain.apply_h1(np.eye(3)), np.diag(gaps))
    anchor = target + np.array([[.2, -.1], [.1, .2], [0., .3]])
    for frequency in (0., .7):
        for legs in (target, anchor):
            expected = legs.T @ np.linalg.solve(
                np.diag(gaps) @ expected_h1+frequency**2*np.eye(3),
                -4*gaps[:, None]*legs)
            result = factorized_projected_response(corrected, legs, frequency)
            np.testing.assert_allclose(result.response, expected, rtol=1e-10, atol=1e-10)
    # The kernel legs are frozen, even when callers reuse the target array.
    target[:] = 0.
    np.testing.assert_allclose(corrected.apply_h1(np.eye(3)), expected_h1)


def test_action_kernel_composition_noncommuting_exchange_and_readonly_input():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, KernelCorrectedDFOperators, factorized_projected_response)
    rng = np.random.default_rng(182)
    mo = rng.normal(scale=.12, size=(3, 5, 5))
    mo = (mo+mo.transpose(0, 2, 1))/2
    plain = FactorizedDFOperators(
        np.arange(1., 7.), mo[:, :2, :2], mo[:, :2, 2:],
        mo[:, :2, 2:], mo[:, 2:, 2:], exact_exchange=.25)
    kernel = np.array([[-.03, .01], [.01, -.02]])
    target = rng.normal(size=(6, 4))[:, ::2]
    calls = []

    def action(rhs):
        assert not rhs.flags.writeable
        calls.append(rhs.shape)
        return kernel @ rhs

    ops = KernelCorrectedDFOperators(
        plain, target, action, local_scale=.75, kernel_storage_bytes=32,
        kernel_workspace_bytes=2048, kernel_work_per_rhs=8)
    h1 = plain.apply_h1(np.eye(6))+3*target @ kernel @ target.T
    h2 = plain.apply_h2(np.eye(6))
    assert np.linalg.norm(h2 @ h1-h1 @ h2) > .01
    np.testing.assert_allclose(ops.apply_h2(np.eye(6)), h2)
    assert not calls
    legs = rng.normal(size=(6, 4))[:, ::2]
    expected = legs.T @ np.linalg.solve(h2 @ h1+.49*np.eye(6), -4*h2 @ legs)
    result = factorized_projected_response(ops, legs, .7)
    np.testing.assert_allclose(result.response, expected, rtol=1e-10, atol=1e-10)
    assert calls


@pytest.mark.parametrize("value", [np.zeros((2, 1)), np.zeros((1, 1), dtype=np.float32),
                                  np.array([[np.nan]]), np.array([[np.inf]])])
def test_composed_kernel_rejects_malformed_callback_results(value):
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, KernelCorrectedDFOperators)
    zero = np.zeros((1, 1, 1))
    plain = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    ops = KernelCorrectedDFOperators(
        plain, np.ones((1, 1)), lambda rhs: value, local_scale=.75,
        kernel_storage_bytes=value.nbytes, kernel_workspace_bytes=1024, kernel_work_per_rhs=1)
    with pytest.raises(ValueError, match="kernel action must return"):
        ops.apply_h1(np.ones((1, 1)))


def test_composed_kernel_refuses_aggregate_bytes_and_work_before_callback(monkeypatch):
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, KernelCorrectedDFOperators, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    plain = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    calls = []

    def action(rhs):
        calls.append(rhs.shape)
        return rhs.copy()

    # Storage: 40 plain + 8 legs + 8 callback + 64 callback workspace = 120.
    # One-column action workspace: 8*(4+16+4+4+4) = 256.
    kwargs = dict(local_scale=.75, kernel_storage_bytes=8, kernel_workspace_bytes=64,
                  kernel_work_per_rhs=100)
    ops = KernelCorrectedDFOperators(plain, np.ones((1, 1)), action, max_bytes=376, **kwargs)
    np.testing.assert_allclose(ops.apply_h1(np.ones((1, 1))), [[4.]])
    calls.clear()
    limited = KernelCorrectedDFOperators(
        plain, np.ones((1, 1)), action, max_bytes=375, **kwargs)
    with pytest.raises(ValueError, match="byte resource"):
        limited.apply_h1(np.ones((1, 1)))
    with pytest.raises(ValueError, match="byte resource"):
        factorized_projected_response(ops, np.ones((1, 1)), .4)
    assert not calls
    generous = KernelCorrectedDFOperators(plain, np.ones((1, 1)), action, **kwargs)
    # Plain work 22 + projection/scaling 6 + supplied kernel 100.
    monkeypatch.setattr(KernelCorrectedDFOperators, "MAX_WORK", 128)
    generous.apply_h1(np.ones((1, 1)))
    calls.clear()
    with pytest.raises(ValueError, match="work resource"):
        generous.apply_h1(np.ones((1, 2)))
    assert not calls


def test_shared_budget_preserves_failed_dispatch_charge_and_local_limit():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, KernelCorrectedDFOperators,
        FactorizedResponseBudget, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    plain = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)

    def failure(rhs):
        raise RuntimeError("kernel failure")

    ops = KernelCorrectedDFOperators(
        plain, np.ones((1, 1)), failure, local_scale=.75, kernel_storage_bytes=0,
        kernel_workspace_bytes=1024, kernel_work_per_rhs=1)
    budget = FactorizedResponseBudget(max_actions=20)
    with pytest.raises(RuntimeError, match="kernel failure"):
        factorized_projected_response(ops, np.ones((1, 1)), .4, budget=budget)
    assert budget.operator_actions == 2  # RHS H2 and failed H1.
    with pytest.raises(RuntimeError, match="budget exhausted"):
        factorized_projected_response(plain, np.ones((1, 1)), .4,
                                      budget=budget, max_actions=1)
    assert budget.operator_actions == 3


def test_static_h1_formulation_checks_both_noncommuting_equations():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, KernelCorrectedDFOperators, factorized_projected_response)
    rng = np.random.default_rng(991)
    mo = rng.normal(scale=.1, size=(3, 5, 5))
    mo = (mo+mo.transpose(0, 2, 1))/2
    plain = FactorizedDFOperators(
        np.arange(1., 7.), mo[:, :2, :2], mo[:, :2, 2:],
        mo[:, :2, 2:], mo[:, 2:, 2:], exact_exchange=.25)
    target = rng.normal(size=(6, 2))
    kernel = -.02*np.eye(2)
    ops = KernelCorrectedDFOperators(
        plain, target, kernel.__matmul__, local_scale=.75,
        kernel_storage_bytes=kernel.nbytes, kernel_workspace_bytes=2048,
        kernel_work_per_rhs=8)
    h1 = plain.apply_h1(np.eye(6))+3*target @ kernel @ target.T
    h2 = plain.apply_h2(np.eye(6))
    assert np.linalg.norm(h2 @ h1-h1 @ h2) > .01
    for legs in (target, rng.normal(size=(6, 2))):
        result = factorized_projected_response(ops, legs, 0., static_h1=True)
        np.testing.assert_allclose(result.response, legs.T @ np.linalg.solve(h1, -4*legs),
                                   rtol=1e-10, atol=1e-10)
        assert max(result.relative_residuals) <= 1e-10
        assert max(result.static_relative_residuals) <= 1e-10


def test_static_h1_zero_and_singular_equations_have_explicit_semantics():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, FactorizedResponseBudget, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    # X=2, Y=0, gap=1, exchange=.5: H2=0, plain H1=0.
    # Add a unit local term to make H1=1. H2D=0 must not erase nonzero D.
    ops = FactorizedDFOperators(
        np.ones(1), 2*np.ones((1, 1, 1)), zero, zero, np.ones((1, 1, 1)),
        exact_exchange=.5, local_scale=1., kernel_legs=np.ones((1, 1)),
        auxiliary_kernel=np.array([[.25]]))
    budget = FactorizedResponseBudget(max_actions=20)
    empty = factorized_projected_response(ops, np.zeros((1, 2)), 0.,
                                         static_h1=True, budget=budget)
    np.testing.assert_array_equal(empty.response, np.zeros((2, 2)))
    assert empty.operator_actions == budget.operator_actions == 0
    assert empty.relative_residuals == empty.static_relative_residuals == (0., 0.)
    mixed = factorized_projected_response(ops, np.array([[0., 1.]]), 0., static_h1=True)
    np.testing.assert_allclose(mixed.response, [[0., 0.], [0., -4.]])
    assert mixed.relative_residuals == (0., 0.)
    old = factorized_projected_response(ops, np.ones((1, 1)), 0.)
    np.testing.assert_array_equal(old.response, [[0.]])
    assert old.static_relative_residuals == ()
    inconsistent = FactorizedDFOperators(
        np.ones(1), zero, zero, zero, zero, exact_exchange=0., local_scale=1.,
        kernel_legs=np.ones((1, 1)), auxiliary_kernel=np.array([[-.25]]))
    with pytest.raises(RuntimeError, match="dual residual/convergence"):
        factorized_projected_response(inconsistent, np.ones((1, 1)), 0., static_h1=True)


@pytest.mark.parametrize("flag, frequency", [(1, 0.), ("yes", 0.), (True, .01)])
def test_static_h1_requires_explicit_bool_and_exact_zero(flag, frequency):
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    with pytest.raises(ValueError, match="static_h1"):
        factorized_projected_response(ops, np.ones((1, 1)), frequency, static_h1=flag)


def test_static_h1_reserves_workspace_and_charges_final_certification():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, FactorizedResponseBudget, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    legs = np.ones((1, 1))
    with pytest.raises(ValueError, match="byte resource"):
        factorized_projected_response(ops, legs, 0., static_h1=True, max_bytes=751)
    result = factorized_projected_response(ops, legs, 0., static_h1=True, max_bytes=752)
    np.testing.assert_allclose(result.response, [[-4.]])
    budget = FactorizedResponseBudget(max_actions=result.operator_actions-1)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        factorized_projected_response(ops, legs, 0., static_h1=True, budget=budget)
    assert budget.operator_actions == result.operator_actions-1
    with pytest.raises(RuntimeError, match="budget exhausted"):
        factorized_projected_response(ops, legs, 0., static_h1=True,
                                      max_actions=result.operator_actions-1)


@pytest.mark.parametrize("frequency", [0., .4, 3.])
def test_factorized_frequency_response_independent_particle_limit(frequency):
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, factorized_projected_response)
    gaps = np.array([.5, 1., 2., 3., 4., 6.])
    operators = FactorizedDFOperators(
        gaps, np.zeros((2, 2, 2)), np.zeros((2, 2, 3)),
        np.zeros((2, 2, 3)), np.zeros((2, 3, 3)), exact_exchange=0.)
    legs = np.array([[1., 2.], [0., 1.], [3., -1.], [.5, .1], [0., 2.], [1., 1.]])
    result = factorized_projected_response(operators, legs, frequency, restart=6)
    expected = legs.T @ ((-4*gaps/(gaps*gaps+frequency**2))[:, None]*legs)
    np.testing.assert_allclose(result.response, expected, atol=1e-10, rtol=1e-10)
    assert max(result.relative_residuals) <= 1e-10


@pytest.mark.parametrize("frequency", [0., .7])
@pytest.mark.parametrize("restart", [2, 6])
def test_noncommuting_factorized_frequency_response_matches_dense_solve(frequency, restart):
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, factorized_projected_response)
    rng = np.random.default_rng(81)
    mo = rng.normal(scale=.12, size=(3, 5, 5))
    mo = (mo+mo.transpose(0, 2, 1))/2
    ops = FactorizedDFOperators(
        np.arange(1., 7.), mo[:, :2, :2], mo[:, :2, 2:],
        mo[:, :2, 2:], mo[:, 2:, 2:], exact_exchange=.25)
    h1, h2 = ops.apply_h1(np.eye(6)), ops.apply_h2(np.eye(6))
    assert np.linalg.norm(h2 @ h1-h1 @ h2) > .01
    legs = rng.normal(size=(6, 3))
    expected = legs.T @ np.linalg.solve(h2 @ h1+frequency**2*np.eye(6), -4*h2 @ legs)
    result = factorized_projected_response(ops, legs, frequency, restart=restart)
    np.testing.assert_allclose(result.response, expected, atol=1e-10, rtol=1e-10)
    assert max(result.relative_residuals) <= 1e-10
    assert 0 < result.operator_actions <= 1000
    with pytest.raises(ValueError):
        result.response.setflags(write=True)


def test_frequency_solver_refuses_action_exhaustion_and_inconsistent_equations():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        factorized_projected_response(ops, np.ones((1, 1)), 0., max_actions=1)
    # H1=0, H2=1: 0*x=-4 has no solution. Never regularize or fall back.
    singular = FactorizedDFOperators(
        np.ones(1), zero, zero, zero, zero, exact_exchange=0., local_scale=1.,
        kernel_legs=np.ones((1, 1)), auxiliary_kernel=np.array([[-.25]]))
    with pytest.raises(RuntimeError, match="residual/convergence"):
        factorized_projected_response(singular, np.ones((1, 1)), 0.)


@pytest.mark.parametrize("kwargs, message", [
    ({"max_bytes": 1}, "byte resource"),
    ({"restart": 0}, "restart"),
    ({"restart": 65}, "restart"),
    ({"max_actions": 0}, "max_actions"),
])
def test_frequency_solver_admission(kwargs, message):
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    with pytest.raises(ValueError, match=message):
        factorized_projected_response(ops, np.ones((1, 1)), 0., **kwargs)


def test_zero_projected_legs_are_exactly_zero():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    result = factorized_projected_response(ops, np.zeros((1, 2)), .4)
    np.testing.assert_array_equal(result.response, np.zeros((2, 2)))
    assert result.relative_residuals == (0., 0.)
    assert result.operator_actions == 2


def test_frequency_action_budget_is_shared_across_projected_columns():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    one = factorized_projected_response(ops, np.ones((1, 1)), .4)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        factorized_projected_response(ops, np.ones((1, 2)), .4,
                                       max_actions=one.operator_actions)


def test_frequency_action_budget_is_shared_across_frequency_calls():
    from psi4.driver.procrouting.isapol_factorized_response import (
        FactorizedDFOperators, FactorizedResponseBudget, factorized_projected_response)
    zero = np.zeros((1, 1, 1))
    ops = FactorizedDFOperators(np.ones(1), zero, zero, zero, zero, exact_exchange=0.)
    legs = np.ones((1, 1))
    baseline = factorized_projected_response(ops, legs, .4)
    budget = FactorizedResponseBudget(max_actions=baseline.operator_actions)
    first = factorized_projected_response(ops, legs, .4, budget=budget)
    np.testing.assert_allclose(first.response, [[-4/1.16]], rtol=1e-12)
    assert budget.operator_actions == first.operator_actions
    with pytest.raises(RuntimeError, match="budget exhausted"):
        factorized_projected_response(ops, legs, .8, budget=budget)
    assert budget.operator_actions == first.operator_actions


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


def test_factorized_actions_match_small_native_df_integral_operators():
    import psi4
    from psi4.driver.procrouting.isapol_factorized_response import FactorizedDFOperators
    from psi4.driver.procrouting.isapol_oeprop import generated_recipe
    from psi4.driver.procrouting.isapol_native_partition import native_partition
    from psi4.driver.procrouting.isapol_native_propagator import _df_two_electron

    psi4.core.be_quiet()
    molecule = psi4.geometry("0 1\nO 0 0 0\nH .757 0 .586\nH -.757 0 .586\n"
                             "symmetry c1\nno_com\nno_reorient")
    psi4.set_options(dict(basis="sto-3g", reference="rhf", scf_type="pk",
                          e_convergence=1e-11, d_convergence=1e-11))
    _, wfn = psi4.energy("hf", molecule=molecule, return_wfn=True)
    partition = native_partition(wfn, generated_recipe(wfn), caller_converged=True)
    full = partition.main.transform @ np.asarray(wfn.Ca())
    nocc, nvir = wfn.nalpha(), wfn.nmo()-wfn.nalpha()
    naux, nao = partition.auxiliary.nfunction, partition.main.basis.nfunction
    ao = np.asarray(partition.coulomb.three_center(partition.main.basis)).reshape(naux, nao, nao)
    mo = np.array([full.T @ block @ full for block in ao])
    metric = np.asarray(partition.coulomb.metric())
    dual = np.linalg.solve(metric, mo.reshape(naux, -1)).reshape(mo.shape)
    energies = np.asarray(wfn.epsilon_a())
    gaps = (energies[nocc:, None]-energies[None, :nocc]).ravel()
    operators = FactorizedDFOperators(
        gaps, mo[:, :nocc, :nocc], mo[:, :nocc, nocc:],
        dual[:, :nocc, nocc:], dual[:, nocc:, nocc:], exact_exchange=.25)
    # Existing dense plain-DF implementation is an independent assembly route,
    # not a CamCASP numerical reference.
    v, x, y = _df_two_electron(partition, full, nocc, nvir)
    rhs = np.eye(nocc*nvir)
    np.testing.assert_allclose(operators.apply_h1(rhs),
                               np.diag(gaps)+4*v-.25*(x+y), atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(operators.apply_h2(rhs),
                               np.diag(gaps)-.25*(x-y), atol=2e-11, rtol=2e-11)
