"""Anisotropic distributed dispersion coefficients (Part B).

Every expected number in this file is a checked-in literal: closed-form binomials,
the reviewed protocol quadrature, or a measured residual bound recorded once and
then held.  Nothing here reads generated JSON and nothing here touches an external
package; the coefficients are validated entirely against the directly computable
second-order dispersion energy of the distributed multipole expansion.

References for the mathematics, all published:

  [S13]  A. J. Stone, "The Theory of Intermolecular Forces", 2nd ed., Oxford
         University Press (2013): Sec. 3.3, Sec. 4.3, App. B, App. F.
  [S78]  A. J. Stone, Mol. Phys. 36, 241 (1978) -- the S function expansion.
  [ST84] A. J. Stone and R. J. A. Tough, Chem. Phys. Lett. 110, 123 (1984).
  [WS03] G. J. Williams and A. J. Stone, J. Chem. Phys. 119, 4620 (2003).
  [BS68] D. M. Brink and G. R. Satchler, "Angular Momentum", 2nd ed. (1968).
"""

import math

import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


# The 15-component real-spherical L3 ordering.  Rank offsets are l*l - 1, so
# rank 1 occupies [0, 3), rank 2 occupies [3, 8) and rank 3 occupies [8, 15).
_L3_DIMENSION = 15
_L3_RANKS = (1, 2, 3)
_RANK_SLICE = {1: (0, 3), 2: (3, 8), 3: (8, 15)}
_COMPONENT_ORDER = (
    "10", "11c", "11s",
    "20", "21c", "21s", "22c", "22s",
    "30", "31c", "31s", "32c", "32s", "33c", "33s",
)

# The real spherical dipole order is (10, 11c, 11s) = (z, x, y), so reading the
# dipole block out in Cartesian x, y, z order permutes the spherical indices as
# below.  This is the same permutation local_spherical_dipole_to_cartesian uses.
_CARTESIAN_FROM_SPHERICAL_DIPOLE = (1, 2, 0)

# binom(2 la + 2 lb, 2 la) for every ordered rank pair an L3 model can supply,
# including (3, 3), which the published isotropic table stops short of.
_NORM_IDENTITY = {
    (1, 1): 6.0,
    (1, 2): 15.0,
    (2, 1): 15.0,
    (1, 3): 28.0,
    (3, 1): 28.0,
    (2, 2): 70.0,
    (2, 3): 210.0,
    (3, 2): 210.0,
    (3, 3): 924.0,
}

_REVIEWED_FREQUENCIES = (
    0.0,
    0.0066096015960872435,
    0.03617481199863096,
    0.09544736369034827,
    0.1976442118453127,
    0.3704172128053672,
    0.6749146404580301,
    1.264899172436498,
    2.619244684547324,
    6.910885950408292,
    37.82376235021415,
)

_REVIEWED_WEIGHTS = (
    0.0,
    0.01711141976082999,
    0.04296478632573829,
    0.07767872676394183,
    0.13105411733467387,
    0.22389687302154546,
    0.4079488542407823,
    0.8387305806393448,
    2.131641821338567,
    8.208052005690483,
    97.92092081488428,
)


def _matrix(values):
    matrix = psi4.core.Matrix(len(values), len(values[0]))
    for row, entries in enumerate(values):
        for column, value in enumerate(entries):
            matrix.set(row, column, value)
    return matrix


def _as_rows(matrix):
    rows, columns = matrix.rows(0), matrix.cols(0)
    return [[matrix.get(row, column) for column in range(columns)] for row in range(rows)]


def _interaction_tensor(separation):
    return _as_rows(psi4.core._atomic_polarizability_test_multipole_interaction_tensor(
        list(separation)))


def _rank_rotation(rotation):
    return _as_rows(psi4.core._atomic_polarizability_test_l3_rank_rotation(_matrix(rotation)))


def _block(tensor, first_rank, second_rank):
    row_lo, row_hi = _RANK_SLICE[first_rank]
    col_lo, col_hi = _RANK_SLICE[second_rank]
    return [row[col_lo:col_hi] for row in tensor[row_lo:row_hi]]


def _matmul(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def _transpose(values):
    return [list(row) for row in zip(*values)]


def _max_deviation(left, right):
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def _rotation_z(angle):
    cos, sin = math.cos(angle), math.sin(angle)
    return [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]


def _rotation_y(angle):
    cos, sin = math.cos(angle), math.sin(angle)
    return [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]]


def _rotation(alpha, beta, gamma):
    return _matmul(_matmul(_rotation_z(alpha), _rotation_y(beta)), _rotation_z(gamma))


def _rotate_vector(rotation, vector):
    return [sum(rotation[row][column] * vector[column] for column in range(3))
            for row in range(3)]


# ---------------------------------------------------------------------------
# B2/B3 -- the real-spherical multipole interaction tensor T_{tu}(R).
# ---------------------------------------------------------------------------

_PROBE_SEPARATION = (0.527, -1.989, 1.411)


def test_interaction_tensor_dipole_block_is_the_analytic_cartesian_form():
    """T^{(1,1)} must be (delta_ab - 3 n_a n_b)/R^3 once reordered to x, y, z."""
    separation = _PROBE_SEPARATION
    length = math.sqrt(sum(component * component for component in separation))
    unit = [component / length for component in separation]
    analytic = [[((1.0 if row == column else 0.0) - 3.0 * unit[row] * unit[column])
                 / length ** 3 for column in range(3)] for row in range(3)]

    dipole = _block(_interaction_tensor(separation), 1, 1)
    permuted = [[dipole[row][column] for column in _CARTESIAN_FROM_SPHERICAL_DIPOLE]
                for row in _CARTESIAN_FROM_SPHERICAL_DIPOLE]
    assert _max_deviation(permuted, analytic) < 1.0e-14


@pytest.mark.parametrize("first_rank,second_rank", sorted(_NORM_IDENTITY))
def test_interaction_tensor_satisfies_the_norm_identity(first_rank, second_rank):
    """sum_{t,u} |T^{(la,lb)}_{tu}|^2 = binom(2 la + 2 lb, 2 la) / R^{2 (la + lb + 1)}.

    This identity is a corollary of the Clebsch-Gordan structure of T and it is the
    identity that reproduces the implemented isotropic prefactors, so it holds for
    all nine ordered rank pairs an L3 model can supply, not only the eight the
    published isotropic table uses.
    """
    separation = _PROBE_SEPARATION
    length = math.sqrt(sum(component * component for component in separation))
    block = _block(_interaction_tensor(separation), first_rank, second_rank)
    total = sum(value * value for row in block for value in row)
    expected = _NORM_IDENTITY[(first_rank, second_rank)] / length ** (
        2 * (first_rank + second_rank + 1))
    assert abs(total / expected - 1.0) < 1.0e-13


@pytest.mark.parametrize("first_rank,second_rank", sorted(_NORM_IDENTITY))
def test_interaction_tensor_is_covariant_under_rotation(first_rank, second_rank):
    """T(O R) = W^{(la)}(O) T(R) W^{(lb)}(O)^T on every rank block."""
    rotation = _rotation(0.83, 1.27, -0.41)
    separation = _PROBE_SEPARATION
    rotated = _rotate_vector(rotation, separation)

    left = _block(_rank_rotation(rotation), first_rank, first_rank)
    right = _block(_rank_rotation(rotation), second_rank, second_rank)
    lhs = _block(_interaction_tensor(rotated), first_rank, second_rank)
    rhs = _matmul(_matmul(left, _block(_interaction_tensor(separation),
                                       first_rank, second_rank)), _transpose(right))
    scale = max(abs(value) for row in rhs for value in row)
    assert _max_deviation(lhs, rhs) / scale < 1.0e-12


def test_rank_rotation_is_block_diagonal_and_orthogonal():
    """The L3 index rotation is block diagonal over ranks 1, 2, 3 and orthogonal."""
    rotation = _rank_rotation(_rotation(-1.11, 0.55, 2.02))
    identity = [[1.0 if row == column else 0.0 for column in range(_L3_DIMENSION)]
                for row in range(_L3_DIMENSION)]
    assert _max_deviation(_matmul(rotation, _transpose(rotation)), identity) < 1.0e-13
    for first in _L3_RANKS:
        for second in _L3_RANKS:
            if first == second:
                continue
            block = _block(rotation, first, second)
            assert max(abs(value) for row in block for value in row) == 0.0


def test_interaction_tensor_is_antisymmetric_in_the_separation_by_rank_parity():
    """T_{tu}(-R) = (-1)^{la + lb} T_{tu}(R): one factor per irregular harmonic."""
    forward = _interaction_tensor(_PROBE_SEPARATION)
    reversed_tensor = _interaction_tensor([-value for value in _PROBE_SEPARATION])
    for first_rank in _L3_RANKS:
        for second_rank in _L3_RANKS:
            sign = -1.0 if (first_rank + second_rank) % 2 else 1.0
            expected = [[sign * value for value in row]
                        for row in _block(forward, first_rank, second_rank)]
            block = _block(reversed_tensor, first_rank, second_rank)
            scale = max(abs(value) for row in expected for value in row)
            assert _max_deviation(block, expected) / scale < 1.0e-13


def test_interaction_tensor_component_order_is_the_l3_ordering():
    order = psi4.core._atomic_polarizability_anisotropic_component_order()
    assert tuple(order) == _COMPONENT_ORDER


def test_interaction_tensor_rejects_a_vanishing_separation():
    with pytest.raises(RuntimeError):
        _interaction_tensor([0.0, 0.0, 0.0])


def test_rank_rotation_rejects_an_improper_frame():
    with pytest.raises(RuntimeError):
        _rank_rotation([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
