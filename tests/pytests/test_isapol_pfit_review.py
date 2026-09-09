"""Independent review regressions for the supplied-input PFIT stage."""
import numpy as np
import pytest
from test_isapol_pfit import array, batch, c, lc, options, problem, solve


def test_condition_estimate_and_failure_contract(options):
    p = problem(tensors=[np.diag([1., 0.]), np.diag([0., 2.])],
                batches=[batch('a', [[1., 0.]], [1.]), batch('b', [[0., 1.]], [2.])])
    options.retain_pair_predictions = True
    r = solve(p, options)
    is_qr = options.solver == c.IsaPfitSolver.StreamingQR
    estimate = r.diagnostics.qr_r_rcond if is_qr else r.diagnostics.normal_h_rcond
    assert estimate == pytest.approx(.5 if is_qr else .25)
    options.minimum_solver_rcond = .75
    rejected = c.isa_pfit_solve(p, options)
    assert rejected.status == c.IsaPfitStatus.IllConditioned
    assert rejected.diagnostics.numerical_rank == 2
    assert rejected.diagnostics.condition_estimate_available
    assert not rejected.diagnostics.objective_available
    assert rejected.predictions == []
    np.testing.assert_array_equal(array(rejected.normal_matrix), np.diag([1., 4.]))
    np.testing.assert_array_equal(rejected.normal_rhs, [1., 4.])
    with pytest.raises(RuntimeError):
        _ = rejected.parameters
    p.batches = [p.batches[0]]
    deficient = c.isa_pfit_solve(p, options)
    assert deficient.status == c.IsaPfitStatus.RankDeficient
    assert deficient.diagnostics.numerical_rank == 1
    assert deficient.diagnostics.condition_estimate_available == is_qr
    assert not deficient.diagnostics.objective_available
    assert deficient.predictions == []
    np.testing.assert_array_equal(array(deficient.normal_matrix), np.diag([1., 0.]))
    np.testing.assert_array_equal(deficient.normal_rhs, [1., 0.])
    with pytest.raises(RuntimeError):
        _ = deficient.parameters


def test_exact_workspace_budget_boundary(options):
    p = problem()
    required = solve(p, options).diagnostics.work_budget_bytes
    options.maximum_work_bytes = required
    assert solve(p, options).diagnostics.work_budget_bytes == required
    options.maximum_work_bytes = required-1
    with pytest.raises(ValueError, match='maximum_work_bytes'):
        c.isa_pfit_solve(p, options)


def test_lc_scaled_rhs_underflow(options):
    p = problem(batches=[batch('zero', [[0.]], [0.])])
    p.linear_penalties = [lc([1e150], 1e-100, 1e-300)]
    r = solve(p, options)
    np.testing.assert_allclose(r.parameters, [1e-250], rtol=1e-14, atol=0)
    np.testing.assert_allclose(r.normal_rhs, [1e-250], rtol=1e-14, atol=0)
    np.testing.assert_allclose(r.lc_penalty_rhs, r.normal_rhs, rtol=1e-14, atol=0)
    np.testing.assert_allclose(r.effective_penalty_rhs, r.normal_rhs, rtol=1e-14, atol=0)


def test_lc_scaled_matrix_symmetry(options):
    p = problem(tensors=[[[0.]], [[0.]]], fixed=[True, True])
    p.linear_penalties = [lc([1e-100, 1e200], 0., 1e-300)]
    r = solve(p, options)
    a = np.sqrt(1e-300) * np.array([1e-100, 1e200])
    expected = np.outer(a, a)
    actual = array(r.lc_penalty_matrix)
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(array(r.effective_penalty_matrix), expected, rtol=1e-14, atol=0)


def test_lc_objective_uses_scaled_rows(options):
    p = problem(tensors=[[[0.]]], fixed=[True])
    m = p.model
    # Unscaled t*p overflows, but sqrt(s)*t*p and its square are finite.
    m.fixed_values = [1e109]
    p.model = m
    p.linear_penalties = [lc([1e200], 0., 1e-320)]
    r = solve(p, options)
    residual = (np.sqrt(1e-320) * 1e200) * 1e109
    np.testing.assert_allclose(r.diagnostics.lc_objective, residual**2, rtol=1e-14, atol=0)
