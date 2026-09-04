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
    assert result["allocation_plan"] == {
        "constraint_rows": 1,
        "constraint_columns": 2,
        "constraint_u_elements": 1,
        "constraint_vt_elements": 4,
        "fit_rows": 4,
        "fit_columns": 1,
        "fit_u_elements": 4,
        "fit_vt_elements": 1,
    }


def test_dipole_only_anchor_penalty_changes_only_masked_variable_and_reports_actual_inputs():
    result = _solve(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [2.0, 2.0, 2.0],
        weights=[4.0, 2.0, 0.5],
        anchor=[0.0, 1.0, 0.0],
        reference=[9.0, 0.0, -4.0],
        penalty=0.001,
    )

    assert result["solution"] == pytest.approx([2.0, 8.0 / 4.001, 2.0], abs=3.0e-14)
    assert result["anchor_residual_norm"] == pytest.approx(abs(result["solution"][1]), rel=2.0e-14)
    assert result["input_metadata"] == {
        "lambda": 0.001,
        "row_weight_min": 0.5,
        "row_weight_max": 4.0,
        "row_weight_source": "caller_explicit",
    }
    assert "reference_anchor_coefficient" not in result
    assert "reference_point_weight" not in result
    assert "options_metadata" not in result


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


# The design below puts a below-cutoff column under a constraint on purpose. Norm-only
# pruning drops it and the constraints are only then restricted to the kept columns, so
# the constraint either loses its whole row or silently comes to mean something else.
# Both failures are exercised, because the second one does not raise.
_PRUNED_CONSTRAINT_DESIGN = [[0.5e-4, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_pruning_both_members_of_a_copy_class_destroys_its_constraint_row():
    """A homogeneous row whose every column is below cutoff becomes identically zero.

    This is the production failure: a PDef COPY class is ``x_i - x_j = 0``, so once both
    of its columns are pruned the row is satisfiable but carries no rank, and the solve
    dies with "constraints are ambiguous (linearly dependent)" rather than fitting worse.
    """
    design = [[0.5e-4, 0.0, 0.0], [0.0, 0.6e-4, 0.0], [0.0, 0.0, 1.0]]
    observations = [0.5e-4 * 2.0, 0.6e-4 * 2.0, 3.0]
    copy_class = [[1.0, -1.0, 0.0]]

    with pytest.raises(Exception, match="constraints are ambiguous"):
        _solve(design, observations, constraints=copy_class, targets=[0.0])

    protected = _solve(design, observations, constraints=copy_class, targets=[0.0],
                       protect_constrained_columns=True)
    assert protected["kept_columns"] == [0, 1, 2]
    assert protected["pruned_columns"] == []
    assert protected["constraint_protected_columns"] == [0, 1]
    assert protected["constraint_rank"] == 1
    assert protected["solution"] == pytest.approx([2.0, 2.0, 3.0], abs=2.0e-11)
    assert protected["constraint_residual_norm"] == pytest.approx(0.0, abs=2.0e-14)


def test_pruning_an_inhomogeneous_constrained_column_fails_closed_instead():
    """With a nonzero target the same zeroed row is caught by the consistency gate.

    Worth pinning separately: it proves the ambiguity throw above is the rank loss and
    not the target, so the two gates are not interchangeable.
    """
    with pytest.raises(Exception, match="constraints are inconsistent"):
        _solve(_PRUNED_CONSTRAINT_DESIGN, [0.5e-4 * 0.7, 2.0, 3.0],
               constraints=[[1.0, 0.0, 0.0]], targets=[0.7])

    protected = _solve(_PRUNED_CONSTRAINT_DESIGN, [0.5e-4 * 0.7, 2.0, 3.0],
                       constraints=[[1.0, 0.0, 0.0]], targets=[0.7],
                       protect_constrained_columns=True)
    assert protected["constraint_protected_columns"] == [0]
    assert protected["solution"] == pytest.approx([0.7, 2.0, 3.0], abs=2.0e-13)


def test_pruning_a_constrained_column_silently_rewrites_a_copy_constraint():
    """The failure with a surviving partner column is wrong answers, not an exception.

    ``x0 = x1`` over the full basis becomes ``-x1 = 0`` once column 0 is pruned, which is
    a different and satisfiable constraint -- so nothing raises and the two arms differ by
    the whole value of the copy class.
    """
    observations = [0.5e-4 * 2.0, 2.0, 3.0]
    copy_class = [[1.0, -1.0, 0.0]]

    pruned = _solve(_PRUNED_CONSTRAINT_DESIGN, observations,
                    constraints=copy_class, targets=[0.0])
    assert pruned["pruned_columns"] == [0]
    assert pruned["constraint_protected_columns"] == []
    assert pruned["solution"] == pytest.approx([0.0, 0.0, 3.0], abs=2.0e-13)

    protected = _solve(_PRUNED_CONSTRAINT_DESIGN, observations,
                       constraints=copy_class, targets=[0.0],
                       protect_constrained_columns=True)
    assert protected["pruned_columns"] == []
    assert protected["constraint_protected_columns"] == [0]
    assert protected["solution"] == pytest.approx([2.0, 2.0, 3.0], abs=2.0e-13)


def test_protecting_constrained_columns_leaves_unconstrained_pruning_alone():
    """The arm is a strict exemption: a below-cutoff column no constraint touches is
    still pruned, so enabling it cannot be mistaken for turning the cutoff off."""
    arguments = ([[0.5e-4, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                 [0.5e-4 * 0.7, 2.0, 3.0])
    baseline = _solve(*arguments, constraints=[[0.0, 1.0, 0.0]], targets=[2.0])
    protected = _solve(*arguments, constraints=[[0.0, 1.0, 0.0]], targets=[2.0],
                       protect_constrained_columns=True)

    for result in (baseline, protected):
        assert result["kept_columns"] == [1, 2]
        assert result["pruned_columns"] == [0]
        assert result["constraint_protected_columns"] == []
    assert protected["solution"] == pytest.approx(baseline["solution"], abs=0.0)


def test_underdetermined_and_rank_deficient_objectives_are_rejected():
    with pytest.raises(Exception, match="rank deficient"):
        _solve([[1.0, 1.0]], [2.0])
    with pytest.raises(Exception, match="rank deficient"):
        _solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_condition_threshold_equality_and_adjacent_floats_have_exact_semantics():
    arguments = ([[1.0, 0.0], [0.0, 0.1]], [1.0, 0.1])
    measured = _solve(*arguments, maximum_condition_number=11.0)["condition_number"]
    assert measured == pytest.approx(10.0, rel=2.0e-14)
    assert _solve(*arguments, maximum_condition_number=measured)["condition_number"] == measured
    assert _solve(
        *arguments, maximum_condition_number=np.nextafter(measured, np.inf)
    )["condition_number"] == measured
    with pytest.raises(Exception, match="condition number"):
        _solve(*arguments, maximum_condition_number=np.nextafter(measured, 0.0))


def test_rank_cutoff_equality_and_adjacent_floats_are_deterministic():
    arguments = ([[1.0, 0.0], [0.0, 1.0e-12]], [1.0, 1.0e-12])
    with pytest.raises(Exception, match="rank deficient"):
        _solve(*arguments, column_cutoff=0.0, rank_tolerance=1.0e-12,
               maximum_condition_number=2.0e12)
    accepted = _solve(*arguments, column_cutoff=0.0,
                      rank_tolerance=np.nextafter(1.0e-12, 0.0),
                      maximum_condition_number=2.0e12)
    assert accepted["rank"] == 2
    with pytest.raises(Exception, match="rank deficient"):
        _solve(*arguments, column_cutoff=0.0,
               rank_tolerance=np.nextafter(1.0e-12, np.inf),
               maximum_condition_number=2.0e12)


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


def test_combined_anchor_and_constraints_match_independent_numpy_kkt_oracle():
    a = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0], [1.0, 1.0, -2.0], [4.0, 0.5, 1.0]])
    b = np.array([1.2, -0.7, 2.1, 0.3])
    weights = np.array([2.0, 0.5, 3.0, 1.25])
    diagonal = np.array([1.0, 0.0, 2.0])
    reference = np.array([0.4, 8.0, -0.2])
    penalty = 0.35
    constraints = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
    targets = np.array([0.25, -0.1])

    augmented = np.vstack((weights[:, None] * a, np.sqrt(penalty) * np.diag(diagonal)))
    augmented_target = np.concatenate((weights * b, np.sqrt(penalty) * diagonal * reference))
    kkt = np.block([
        [augmented.T @ augmented, constraints.T],
        [constraints, np.zeros((len(targets), len(targets)))],
    ])
    rhs = np.concatenate((augmented.T @ augmented_target, targets))
    expected = np.linalg.solve(kkt, rhs)[: a.shape[1]]

    result = _solve(a.tolist(), b.tolist(), weights=weights.tolist(), anchor=diagonal.tolist(),
                    reference=reference.tolist(), penalty=penalty,
                    constraints=constraints.tolist(), targets=targets.tolist())
    assert result["solution"] == pytest.approx(expected, rel=4.0e-13, abs=4.0e-13)


def test_constraint_row_permutation_covariance():
    constraints = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
    targets = np.array([0.5, -0.25])
    arguments = ([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [1.0, 1.0, 2.0]], [1.0, 2.0, -1.0])
    first = _solve(*arguments, constraints=constraints.tolist(), targets=targets.tolist())
    order = [1, 0]
    second = _solve(*arguments, constraints=constraints[order].tolist(), targets=targets[order].tolist())
    assert second["solution"] == pytest.approx(first["solution"], abs=4.0e-13)


def test_pruned_column_is_fixed_to_zero_before_constraint_elimination():
    result = _solve(
        [[0.5e-4, 0.0], [0.0, 1.0]], [7.0, 3.0],
        constraints=[[1.0, 1.0]], targets=[2.0],
    )
    assert result["pruned_columns"] == [0]
    assert result["solution"] == pytest.approx([0.0, 2.0], abs=2.0e-14)
    assert result["constraint_residual_norm"] < 2.0e-14


def test_large_tall_problem_uses_economy_allocation_plan():
    rows = 20000
    grid = np.linspace(-1.0, 1.0, rows)
    a = np.column_stack((np.ones(rows), grid, grid * grid))
    expected = np.array([0.75, -1.25, 2.5])
    result = _solve(a.tolist(), (a @ expected).tolist(), column_cutoff=0.0)

    assert result["solution"] == pytest.approx(expected, rel=2.0e-12, abs=2.0e-12)
    plan = result["allocation_plan"]
    assert plan["fit_rows"] == rows + 3
    assert plan["fit_columns"] == 3
    assert plan["fit_u_elements"] == (rows + 3) * 3
    assert plan["fit_vt_elements"] == 9
    assert plan["fit_u_elements"] < (rows + 3) ** 2


def test_explicit_workspace_cap_rejects_queried_lwork_before_allocation():
    with pytest.raises(Exception, match="workspace exceeds the explicit allocation cap"):
        _solve([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0],
               maximum_workspace_elements=1)


def test_source_uses_only_allowlisted_economy_svd_without_normal_equations():
    source = (Path(__file__).parents[2] / "psi4/src/psi4/libmints/atomic_polarizability.cc").read_text()
    start = source.index("struct LeastSquaresSVD")
    end = source.index("\n}  // namespace detail", start)
    body = source[start:end]
    lapack_calls = set(re.findall(r"C_(D[A-Z0-9]+)\s*\(", body))
    assert lapack_calls == {"DGESVD", "DGESDD"}
    for forbidden in (
        "C_DGESV(", "C_DPOSV(", "C_DGETRF(", "C_DGETRI(",
        "invert(", "inverse(", "normal_equation", "normal equation", "AtA", "gram_matrix",
    ):
        assert forbidden not in body
    assert re.search(r"\brows\s*\*\s*rows\b", body) is None
    assert re.search(r"(?:design|matrix|a)\s*\.?(?:transpose|T)\s*\([^)]*\)\s*[*@]", body,
                     re.IGNORECASE) is None
    assert re.search(r"(?:transpose|T)\s*\([^)]*\)\s*[*@]\s*(?:design|matrix|a)", body,
                     re.IGNORECASE) is None
    cpp_sources = source + (
        Path(__file__).parents[2] / "psi4/src/export_oeprop.cc"
    ).read_text() + (
        Path(__file__).parents[2] / "psi4/src/psi4/libmints/atomic_polarizability.h"
    ).read_text()
    assert "reference_anchor_coefficient" not in cpp_sources
    assert "reference_point_weight" not in cpp_sources
