"""Pure native constrained least-squares tests with independent NumPy/literal oracles."""

from pathlib import Path
import re

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


def _matrix(values, columns=None):
    rows = len(values)
    columns = len(values[0]) if rows else columns
    matrix = psi4.core.Matrix(rows, columns)
    for row, entries in enumerate(values):
        assert len(entries) == columns
        for column, value in enumerate(entries):
            matrix.set(row, column, float(value))
    return matrix


def _solve(a, b, *, weights=None, anchor=None, reference=None, penalty=0.0,
           constraints=None, targets=None, **options):
    nrow = len(a)
    ncol = len(a[0])
    constraints = [] if constraints is None else constraints
    return psi4.core._atomic_polarizability_test_constrained_least_squares(
        _matrix(a),
        b,
        [1.0] * nrow if weights is None else weights,
        penalty,
        [0.0] * ncol if anchor is None else anchor,
        [0.0] * ncol if reference is None else reference,
        _matrix(constraints, ncol),
        [] if targets is None else targets,
        options,
    )


def test_exact_reconstruction_and_diagnostics_match_literal_residuals():
    a = [[2.0, -1.0], [1.0, 3.0], [-2.0, 4.0]]
    expected = np.array([1.25, -0.75])
    b = np.asarray(a) @ expected
    result = _solve(a, b.tolist())

    assert result["solution"] == pytest.approx(expected, abs=2.0e-14)
    assert result["rank"] == 2
    assert result["free_dimension"] == 2
    assert result["kept_columns"] == [0, 1]
    assert result["pruned_columns"] == []
    assert result["weighted_residual_norm"] == pytest.approx(0.0, abs=2.0e-14)
    assert result["anchor_residual_norm"] == 0.0
    assert result["constraint_residual_norm"] == 0.0
    assert result["objective_residual_norm"] == pytest.approx(
        np.linalg.norm(np.asarray(a) @ np.asarray(result["solution"]) - b), abs=1.0e-15
    )


def test_noisy_weighted_fit_matches_independent_numpy_lstsq():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 4.0]])
    b = np.array([0.8, 2.2, 2.9, 5.3])
    weights = np.array([4.0, 0.5, 2.0, 1.5])
    expected = np.linalg.lstsq(weights[:, None] * a, weights * b, rcond=1.0e-12)[0]

    result = _solve(a.tolist(), b.tolist(), weights=weights.tolist())

    assert result["solution"] == pytest.approx(expected, rel=2.0e-13, abs=2.0e-13)
    residual = weights * (a @ np.asarray(result["solution"]) - b)
    assert result["weighted_residual_norm"] == pytest.approx(np.linalg.norm(residual), rel=2.0e-13)


def test_equality_constraint_copies_h2_parameters_exactly():
    result = _solve(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 3.0],
        constraints=[[1.0, -1.0]],
        targets=[0.0],
    )

    assert result["solution"] == pytest.approx([2.0, 2.0], abs=3.0e-14)
    assert result["constraint_rank"] == 1
    assert result["free_dimension"] == 1
    assert result["constraint_residual_norm"] < 2.0e-14


def test_dipole_only_anchor_penalty_changes_only_masked_variable_and_reports_policy_metadata():
    result = _solve(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [2.0, 2.0, 2.0],
        anchor=[0.0, 1.0, 0.0],
        reference=[9.0, 0.0, -4.0],
        penalty=0.001,
    )

    assert result["solution"] == pytest.approx([2.0, 2.0 / 1.001, 2.0], abs=3.0e-14)
    assert result["anchor_residual_norm"] == pytest.approx(abs(result["solution"][1]), rel=2.0e-14)
    assert result["options_metadata"] == {
        "reference_anchor_coefficient": 0.001,
        "reference_point_weight": 4.0,
    }


def test_cutoff_is_strictly_below_and_preserves_full_to_reduced_mapping():
    cutoff = 1.0e-4
    result = _solve(
        [[0.5 * cutoff, 0.0, 0.0], [0.0, cutoff, 0.0], [0.0, 0.0, 2.0 * cutoff]],
        [5.0e-5, 2.0e-4, 6.0e-4],
        maximum_condition_number=3.0e4,
    )

    assert result["solution"] == pytest.approx([0.0, 2.0, 3.0], abs=2.0e-13)
    assert result["kept_columns"] == [1, 2]
    assert result["pruned_columns"] == [0]
    assert result["full_to_reduced"] == [-1, 0, 1]
    assert result["column_weighted_norms"] == pytest.approx([0.5 * cutoff, cutoff, 2.0 * cutoff])


def test_cutoff_rejection_mode_fails_closed():
    with pytest.raises(Exception, match="column 0.*cutoff"):
        _solve([[0.999e-4, 0.0], [0.0, 1.0]], [0.0, 1.0], prune_below_cutoff=False)


def test_underdetermined_and_rank_deficient_objectives_are_rejected():
    with pytest.raises(Exception, match="rank deficient"):
        _solve([[1.0, 1.0]], [2.0])
    with pytest.raises(Exception, match="rank deficient"):
        _solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_condition_threshold_accepts_and_rejects_on_explicit_sides():
    accepted = _solve(
        [[1.0, 0.0], [0.0, 0.1]], [1.0, 0.1], maximum_condition_number=10.0 * (1.0 + 1.0e-12)
    )
    assert accepted["condition_number"] == pytest.approx(10.0, rel=2.0e-14)
    with pytest.raises(Exception, match="condition number"):
        _solve([[1.0, 0.0], [0.0, 0.1]], [1.0, 0.1], maximum_condition_number=9.999)


def test_zero_weight_rows_are_ignored_without_changing_solution():
    result = _solve(
        [[1.0, 0.0], [0.0, 1.0], [1000.0, -2000.0]],
        [2.0, -3.0, 9.0e9],
        weights=[1.0, 1.0, 0.0],
    )
    assert result["solution"] == pytest.approx([2.0, -3.0], abs=2.0e-14)
    assert result["weighted_residual_norm"] == pytest.approx(0.0, abs=2.0e-14)


def test_column_permutation_covariance_including_anchor_and_constraint():
    a = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0], [1.0, -1.0, 1.0], [4.0, 0.0, 2.0]])
    b = [1.0, 2.0, -1.0, 3.0]
    anchor = np.array([1.0, 0.0, 2.0])
    reference = np.array([0.5, 8.0, -0.5])
    c = np.array([[1.0, -1.0, 0.0]])
    original = _solve(a.tolist(), b, anchor=anchor.tolist(), reference=reference.tolist(), penalty=0.2,
                      constraints=c.tolist(), targets=[0.25])
    permutation = np.array([2, 0, 1])
    permuted = _solve(a[:, permutation].tolist(), b, anchor=anchor[permutation].tolist(),
                      reference=reference[permutation].tolist(), penalty=0.2,
                      constraints=c[:, permutation].tolist(), targets=[0.25])
    restored = np.empty(3)
    restored[permutation] = permuted["solution"]
    assert restored == pytest.approx(original["solution"], rel=3.0e-13, abs=3.0e-13)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"weights": [1.0, -1.0]}, "nonnegative"),
        ({"weights": [1.0, float("nan")]}, "finite"),
        ({"penalty": -1.0}, "lambda"),
        ({"anchor": [1.0, float("inf")]}, "finite"),
        ({"constraints": [[float("nan"), 0.0]], "targets": [0.0]}, "finite"),
    ],
)
def test_invalid_numeric_inputs_fail_closed(change, match):
    arguments = dict(weights=[1.0, 1.0], penalty=0.0, anchor=[0.0, 0.0],
                     constraints=None, targets=None)
    arguments.update(change)
    with pytest.raises(Exception, match=match):
        _solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0], **arguments)


def test_dependent_or_inconsistent_constraints_are_rejected_as_ambiguous():
    with pytest.raises(Exception, match="ambiguous"):
        _solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0],
               constraints=[[1.0, -1.0], [2.0, -2.0]], targets=[0.0, 0.0])
    with pytest.raises(Exception, match="inconsistent"):
        _solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0],
               constraints=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], targets=[0.0, 0.0, 1.0])


def test_source_uses_svd_without_normal_equation_solve():
    source = (Path(__file__).parents[2] / "psi4/src/psi4/libmints/atomic_polarizability.cc").read_text()
    start = source.index("ConstrainedLeastSquaresResult solve_constrained_least_squares")
    end = source.index("\n}  // namespace detail", start)
    body = source[start:end]
    assert "C_DGESDD" in body
    assert "C_DGESV(" not in body
    assert "C_DPOSV(" not in body
    assert re.search(r"(?:A|design)T[A-Za-z_]*\s*[*x]", body) is None
    assert "normal_equation" not in body
