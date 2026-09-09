# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
# Graph tests ported from our atomic_polarizability math tests (LGPL v3).
import pytest
import psi4


def _matrix_values(matrix):
    rows, columns = matrix.shape
    return [[matrix.get(row, column) for column in range(columns)] for row in range(rows)]


def _matmul(first, second):
    return [
        [sum(first[row][k] * second[k][column] for k in range(len(second)))
         for column in range(len(second[0]))]
        for row in range(len(first))
    ]


def _transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def _assert_matrix_close(actual, expected, tolerance):
    assert len(actual) == len(expected)
    assert all(actual[row] == pytest.approx(expected[row], abs=tolerance)
               for row in range(len(expected)))


def test_lw_graph_operator_is_symmetric_with_one_null_mode_for_connected_chain():
    operator, pseudoinverse, eigenvalues = psi4.core.isa_lw_graph_math(
        3, [(0, 1), (1, 2)]
    )
    assert [operator.get(i, j) for i in range(3) for j in range(3)] == [
        -1.0, 1.0, 0.0, 1.0, -2.0, 1.0, 0.0, 1.0, -1.0
    ]
    assert all(operator.get(i, j) == operator.get(j, i) for i in range(3) for j in range(3))
    assert all(sum(operator.get(i, j) for j in range(3)) == 0.0 for i in range(3))
    assert sum(abs(value) < 1.0e-12 for value in eigenvalues) == 1
    assert [
        sum(operator.get(i, k) * pseudoinverse.get(k, j) for k in range(3))
        for i in range(3) for j in range(3)
    ] == pytest.approx([
        2 / 3, -1 / 3, -1 / 3, -1 / 3, 2 / 3, -1 / 3, -1 / 3, -1 / 3, 2 / 3
    ], abs=1.0e-12)


def test_lw_lapack_pseudoinverse_identities_on_realistic_chain():
    count = 24
    bonds = [(site, site + 1) for site in range(count - 1)]
    operator, pseudoinverse, eigenvalues = psi4.core.isa_lw_graph_math(count, bonds)
    b = _matrix_values(operator)
    inverse = _matrix_values(pseudoinverse)
    projector = _matmul(b, inverse)
    assert sum(abs(value) < 1.0e-10 for value in eigenvalues) == 1
    assert max(abs(inverse[i][j] - inverse[j][i]) for i in range(count) for j in range(count)) < 1.0e-11
    _assert_matrix_close(_matmul(_matmul(b, inverse), b), b, 2.0e-10)
    _assert_matrix_close(_matmul(_matmul(inverse, b), inverse), inverse, 2.0e-10)
    _assert_matrix_close(projector, _transpose(projector), 2.0e-10)
    _assert_matrix_close(_matmul(projector, projector), projector, 2.0e-10)


def test_lw_disconnected_components_and_isolated_site_literal_oracle():
    operator, inverse, eigenvalues = psi4.core.isa_lw_graph_math(5, [(0, 2), (3, 1)])
    expected = [
        [-1, 0, 1, 0, 0],
        [0, -1, 0, 1, 0],
        [1, 0, -1, 0, 0],
        [0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0],
    ]
    expected_inverse = [
        [-0.25, 0, 0.25, 0, 0],
        [0, -0.25, 0, 0.25, 0],
        [0.25, 0, -0.25, 0, 0],
        [0, 0.25, 0, -0.25, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _matrix_values(operator) == expected
    _assert_matrix_close(_matrix_values(inverse), expected_inverse, 1e-12)
    # Component order, NOT a global eigenvalue sort.
    assert eigenvalues == pytest.approx([-2, 0, -2, 0, 0], abs=1e-12)
    a, p = _matrix_values(operator), _matrix_values(inverse)
    ap, pa = _matmul(a, p), _matmul(p, a)
    _assert_matrix_close(_matmul(ap, a), a, 1e-12)
    _assert_matrix_close(_matmul(pa, p), p, 1e-12)
    _assert_matrix_close(ap, _transpose(ap), 1e-12)
    _assert_matrix_close(pa, _transpose(pa), 1e-12)


@pytest.mark.parametrize("count", [1, 4])
def test_lw_all_isolated_sites(count):
    operator, inverse, eigenvalues = psi4.core.isa_lw_graph_math(count, [])
    assert operator.shape == (count, count)
    assert inverse.shape == (count, count)
    assert _matrix_values(operator) == [[0.0] * count for _ in range(count)]
    assert _matrix_values(inverse) == [[0.0] * count for _ in range(count)]
    assert eigenvalues == [0.0] * count


@pytest.mark.parametrize("count,bonds,message", [
    (0, [], "at least one site"),
    (257, [], "dense graph limit"),
    (2**31, [], "dense graph limit"),
    (3, [(0, 0)], "invalid bond"),
    (3, [(0, 3)], "invalid bond"),
    (3, [(3, 0)], "invalid bond"),
    (3, [(0, 1), (0, 1)], "duplicate bond"),
    (3, [(0, 1), (1, 0)], "duplicate bond"),
    (2, [(0, 1), (1, 0)], "simple graph capacity"),
])
def test_lw_invalid_graph_rejected(count, bonds, message):
    with pytest.raises(RuntimeError, match=message):
        psi4.core.isa_lw_graph_math(count, bonds)


@pytest.mark.parametrize("count,bonds", [
    (-1, []), (2**100, []), (2.5, []),
    (3, [(-1, 1)]), (3, [(0, 2**100)]), (3, [(0, 1.5)]),
    (3, [(0,)]), (3, [(0, 1, 2)]), (3, [0, 1]),
])
def test_lw_invalid_binding_dimensions_rejected(count, bonds):
    with pytest.raises((TypeError, OverflowError)):
        psi4.core.isa_lw_graph_math(count, bonds)


def test_lw_reversed_edges_and_owned_returns():
    bonds = [(0, 1)]
    a, p, values = psi4.core.isa_lw_graph_math(2, bonds)
    b, q, other_values = psi4.core.isa_lw_graph_math(2, [(1, 0)])
    assert _matrix_values(a) == _matrix_values(b)
    _assert_matrix_close(_matrix_values(p), _matrix_values(q), 1e-12)
    assert values == pytest.approx(other_values, abs=1e-12)
    bonds.clear()
    a.set(0, 0, 123.0)
    p.set(0, 0, 456.0)
    values[0] = 789.0
    assert b.get(0, 0) == -1.0
    assert q.get(0, 0) == pytest.approx(-0.25, abs=1e-12)
    assert other_values == pytest.approx([-2, 0], abs=1e-12)
    fresh_a, fresh_p, fresh_values = psi4.core.isa_lw_graph_math(2, [(0, 1)])
    assert fresh_a.get(0, 0) == -1.0
    assert fresh_p.get(0, 0) == pytest.approx(-0.25, abs=1e-12)
    assert fresh_values == pytest.approx([-2, 0], abs=1e-12)
