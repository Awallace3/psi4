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


# ---------------------------------------------------------------------------
# B4/B5 -- the full block-product frequency integral M.
# ---------------------------------------------------------------------------
#
# M_{(t t')(u u')} = (1/2 pi) sum_k w_k alpha^A_{t t'}(i w_k) alpha^B_{u u'}(i w_k)
#
# The 1/(2 pi) lives inside M.  That bookkeeping is load bearing: with it, the
# recoupling table traced over the diagonal rank blocks must equal
# binom(2 la + 2 lb, 2 la) and *not* binom/(2 pi), because the 1/(2 pi) is
# already spent.  The isotropic engine spends it in the rank-pair prefactor
# instead, which is why tracing M below divides by (2 la + 1)(2 lb + 1).

_ISOTROPIC_ORDERS = (6, 8, 10, 12)
_DISPERSION_SITES = ([0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11])


def _synthetic_l3(seed, scale=1.0):
    """A symmetric, rank-complete 15 by 15 L3 tensor from a reproducible sequence.

    A plain linear congruential recurrence keeps the tensor reproducible without
    depending on a random-number stream that is not part of any contract.
    """
    state = (seed * 6364136223846793005 + 1442695040888963407) % (1 << 63)
    values = [[0.0] * _L3_DIMENSION for _ in range(_L3_DIMENSION)]
    for row in range(_L3_DIMENSION):
        for column in range(row, _L3_DIMENSION):
            state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 63)
            entry = (float(state >> 11) / float(1 << 51) - 1.0) * scale
            values[row][column] = entry
            values[column][row] = entry
    for index in range(_L3_DIMENSION):
        values[index][index] += (2.0 + 0.25 * index) * scale
    return values


def _synthetic_models(site_seeds, scale=1.0):
    """models[frequency][site], one tensor per protocol grid point per site."""
    return [[_synthetic_l3(seed * 1000 + point, scale) for seed in site_seeds]
            for point in range(len(_REVIEWED_FREQUENCIES))]


def _block_product(first_tensors, second_tensors, weights=None):
    return psi4.core._atomic_polarizability_test_anisotropic_block_product(
        [_matrix(tensor) for tensor in first_tensors],
        [_matrix(tensor) for tensor in second_tensors],
        list(weights if weights is not None else _REVIEWED_WEIGHTS))


def _product_at(product, first, second, third, fourth):
    return product[((first * _L3_DIMENSION + second) * _L3_DIMENSION + third) * _L3_DIMENSION
                   + fourth]


def _isotropic_dispersion(models, sites=None):
    site_count = len(models[0])
    return psi4.core._atomic_polarizability_compute_dispersion(
        _matrix(list(sites if sites is not None else _DISPERSION_SITES[:site_count])),
        list(_REVIEWED_FREQUENCIES),
        [_matrix(tensor) for frequency in models for tensor in frequency],
        list(_REVIEWED_FREQUENCIES),
        list(_REVIEWED_WEIGHTS))


def _within_ulp(actual, expected, allowed):
    """True when actual is within `allowed` units in the last place of expected."""
    if actual == expected:
        return True
    return abs(actual - expected) <= allowed * math.ulp(abs(expected))


def test_block_product_traced_reproduces_the_isotropic_path_to_the_last_place():
    """Tracing the diagonal rank blocks of M must return the isotropic engine's value.

    This is the seam that says the generalisation really is a generalisation.

    Exact bit equality is *not* available here and asserting it would be wrong.  The
    isotropic engine traces each rank block first and then sums over the frequency
    grid; M sums over the grid first and the rank traces are taken afterwards.  The
    two are the same real number but a different floating-point summation order, so
    the honest claim -- and the one asserted -- is agreement to a couple of units in
    the last place.  The spec's "bit-for-bit" wording is unachievable through M by
    construction, not by defect.
    """
    models = _synthetic_models((1, 2))
    reference = _isotropic_dispersion(models)
    traced = _block_product([frequency[0] for frequency in models],
                            [frequency[1] for frequency in models])["isotropic"]
    for index, order in enumerate(_ISOTROPIC_ORDERS):
        expected = reference["c%d" % order].get(0, 1)
        assert _within_ulp(traced[index], expected, 4)


def test_block_product_traced_reproduces_the_isotropic_path_for_every_ordered_pair():
    """Every ordered site pair, on and off the diagonal, agrees to the last place."""
    models = _synthetic_models((3, 4, 5))
    reference = _isotropic_dispersion(models)
    for first in range(3):
        for second in range(3):
            traced = _block_product([frequency[first] for frequency in models],
                                    [frequency[second] for frequency in models])["isotropic"]
            for index, order in enumerate(_ISOTROPIC_ORDERS):
                expected = reference["c%d" % order].get(first, second)
                assert _within_ulp(traced[index], expected, 8)


def test_block_product_is_bit_exactly_symmetric_under_site_exchange():
    """M^{AB}_{(t t')(u u')} = M^{BA}_{(u u')(t t')}, element for element, exactly.

    Parenthesizing the two site factors as one product makes this exact rather than
    approximate, which is what lets the permutation relation on the coefficients be
    asserted at machine precision later.
    """
    models = _synthetic_models((6, 7))
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    forward = _block_product(first, second)["values"]
    backward = _block_product(second, first)["values"]
    for t in range(_L3_DIMENSION):
        for tp in range(_L3_DIMENSION):
            for u in range(_L3_DIMENSION):
                for up in range(_L3_DIMENSION):
                    assert (_product_at(forward, t, tp, u, up)
                            == _product_at(backward, u, up, t, tp))


def test_block_product_traced_is_symmetric_under_site_exchange():
    """The ordered rank-pair table is exchange closed, so tracing M is site symmetric."""
    models = _synthetic_models((6, 7))
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    forward = _block_product(first, second)["isotropic"]
    backward = _block_product(second, first)["isotropic"]
    for index in range(len(_ISOTROPIC_ORDERS)):
        assert _within_ulp(forward[index], backward[index], 8)


def test_block_product_carries_the_one_over_two_pi_inside_m():
    """M already carries the 1/(2 pi); one unit-weight point of unit tensors shows it."""
    identity = [[1.0 if row == column else 0.0 for column in range(_L3_DIMENSION)]
                for row in range(_L3_DIMENSION)]
    unit = _block_product([identity], [identity], [1.0])["values"]
    assert abs(_product_at(unit, 0, 0, 0, 0) - 1.0 / (2.0 * math.pi)) < 1.0e-17
    assert _product_at(unit, 0, 1, 0, 0) == 0.0


def test_block_product_is_linear_in_the_quadrature_weights():
    models = _synthetic_models((7, 8))
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    single = _block_product(first, second)["isotropic"]
    doubled = _block_product(first, second,
                             [2.0 * weight for weight in _REVIEWED_WEIGHTS])["isotropic"]
    for index in range(len(_ISOTROPIC_ORDERS)):
        assert doubled[index] == 2.0 * single[index]


@pytest.mark.parametrize("indices", [(0, 0, 0, 0), (2, 7, 1, 14), (14, 3, 8, 5),
                                     (5, 5, 12, 12), (9, 0, 13, 6)])
def test_block_product_is_the_weighted_outer_product_of_the_two_tensors(indices):
    models = _synthetic_models((11, 12))
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    values = _block_product(first, second)["values"]
    t, tp, u, up = indices
    expected = sum(weight * (first[point][t][tp] * second[point][u][up])
                   for point, weight in enumerate(_REVIEWED_WEIGHTS)) / (2.0 * math.pi)
    # Two units in the last place: the C++ accumulation is fused-multiply-add
    # contracted under -march=native while Python's is not, so the two are the same
    # expression evaluated with different rounding, not different expressions.
    assert _within_ulp(_product_at(values, t, tp, u, up), expected, 2)


def test_block_product_static_point_carries_no_quadrature_weight():
    models = _synthetic_models((21, 22))
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    baseline = _block_product(first, second)
    first[0] = _synthetic_l3(999, scale=17.0)
    second[0] = _synthetic_l3(998, scale=17.0)
    perturbed = _block_product(first, second)
    assert perturbed["values"] == baseline["values"]
    assert perturbed["isotropic"] == baseline["isotropic"]


def test_block_product_rejects_a_tensor_count_that_misses_the_grid():
    models = _synthetic_models((31, 32))
    with pytest.raises(RuntimeError):
        _block_product([frequency[0] for frequency in models][:-1],
                       [frequency[1] for frequency in models])


def test_block_product_rejects_a_negative_quadrature_weight():
    models = _synthetic_models((33, 34))
    weights = list(_REVIEWED_WEIGHTS)
    weights[4] = -weights[4]
    with pytest.raises(RuntimeError):
        _block_product([frequency[0] for frequency in models],
                       [frequency[1] for frequency in models], weights)


def test_direct_energy_is_exactly_the_double_sum_over_the_interaction_tensor():
    """E_disp = -sum_{t t' u u'} T_{tu} T_{t'u'} M_{(t t')(u u')}, no table involved.

    Contracted independently in Python over all 15^4 terms.  This is the oracle the
    recoupling table is later gated against, so it is pinned here on its own.
    """
    models = _synthetic_models((41, 42))
    separation = [1.7, -2.9, 5.3]
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    energy = psi4.core._atomic_polarizability_test_direct_anisotropic_energy(
        [_matrix(tensor) for tensor in first], [_matrix(tensor) for tensor in second],
        list(_REVIEWED_WEIGHTS), separation)
    tensor = _interaction_tensor(separation)
    values = _block_product(first, second)["values"]
    total = 0.0
    for t in range(_L3_DIMENSION):
        for tp in range(_L3_DIMENSION):
            for u in range(_L3_DIMENSION):
                row = ((t * _L3_DIMENSION + tp) * _L3_DIMENSION + u) * _L3_DIMENSION
                total += sum(tensor[t][u] * tensor[tp][up] * values[row + up]
                             for up in range(_L3_DIMENSION))
    assert abs(energy / (-total) - 1.0) < 1.0e-14
    assert energy < 0.0


# ---------------------------------------------------------------------------
# B6/B7 -- the real recoupling table W.
# ---------------------------------------------------------------------------
#
# W is stored factorised and never materialised densely: one label costs 15^4
# doubles and there are 29762 labels, so a dense table would be twelve gigabytes.
# The factorised form is exact,
#
#   W^{n, l1 k1, l2 k2, j}_{(t t')(u u')}
#     = sum_{blocks with that n} g^r Pcheck^{(la, la', l1)}_{k1,(t t')}
#                                    Pcheck^{(lb, lb', l2)}_{k2,(u u')},
#
# so the 2723 nonzero g^r scalars plus 37 universal coupling matrices are the
# whole table.  dense_anisotropic_recoupling exists only as a single-label test
# seam and is used below to pin the isotropic reduction and the exchange rule.
#
# The radial order is n = la + la' + lb + lb' + 2, NOT 2 (la + lb + 1).  T_{tu}
# and T_{t'u'} may carry different ranks, so an L3 model produces every order
# from 6 to 14, including the odd C7, C9, C11, C13 that vanish on orientational
# averaging but are genuine anisotropic coefficients.

_TABLE_VERSION = "partB-recoupling-1"
_RECOUPLING_ENTRY_COUNT = 2723
_INTERNAL_LABEL_COUNT = 29762
_PUBLISHED_LABEL_COUNT = 16985
_COUPLING_MATRIX_COUNT = 37
_DISTINCT_RANK_QUADRUPLES = 530
_LABELS_PER_ORDER = {6: 104, 7: 391, 8: 896, 9: 1748, 10: 3063,
                     11: 4486, 12: 6297, 13: 7457, 14: 5320}

# The single n = 6 block is (la, la', lb, lb') = (1, 1, 1, 1). Closed forms are
# annotations; the numbers are what the generator must reproduce.
_N6_SCALARS = {
    (0, 0, 0): 2.0,                    # 2
    (0, 2, 2): -1.4142135623730951,    # -sqrt(2)
    (1, 1, 0): 1.7320508075688772,     # sqrt(3), symmetry null
    (1, 1, 2): 2.449489742783178,      # sqrt(6), symmetry null
    (2, 0, 2): -1.4142135623730951,    # -sqrt(2)
    (2, 2, 0): 0.4472135954999579,     # 1/sqrt(5)
    (2, 2, 2): 0.5345224838248488,     # sqrt(2/7)
    (2, 2, 4): 4.30282299360405,       # 18 sqrt(2/35)
}

# The C6 (l1, l2, j) triples that survive a symmetric L3 model. This is the
# textbook anisotropic-C6 label set and it is an external sanity check on the
# whole construction: dipole-dipole dispersion anisotropy carries only the
# rank-0 and rank-2 parts of each site's polarizability.
_LIVE_C6_TRIPLES = ((0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0), (2, 2, 2), (2, 2, 4))

_TABLE_REJECTIONS = (
    "version",
    "generator",
    "component_order",
    "conventions",
    "nonfinite_scalar",
    "site_rank",
    "order_mismatch",
    "order_range",
    "first_rank_triangle",
    "second_rank_triangle",
    "capital_triangle",
    "coupled_triangle",
    "parity",
    "zero_scalar",
    "duplicate_entry",
    "missing_site_rank",
    "component_index",
    "label_triangle",
    "label_order",
    "duplicate_label",
    "exchange_closure",
    "exchange_scalar",
    "empty_label",
    "offset_count",
    "offset_monotonic",
    "entry_index_range",
    "missing_coupling_matrix",
    "coupling_matrix_shape",
    "isotropic_reduction",
    "rotation_orthogonality",
    "collapse_residual",
)


def _table():
    return psi4.core._atomic_polarizability_anisotropic_recoupling_table()


def _dense_recoupling(label):
    return psi4.core._atomic_polarizability_test_dense_anisotropic_recoupling(list(label))


def _coefficients(first_tensors, second_tensors, weights=None):
    return psi4.core._atomic_polarizability_test_anisotropic_coefficients(
        [_matrix(tensor) for tensor in first_tensors],
        [_matrix(tensor) for tensor in second_tensors],
        list(weights if weights is not None else _REVIEWED_WEIGHTS))


def _label_index(table):
    return {tuple(label): index for index, label in enumerate(table["labels"])}


def test_recoupling_table_declares_its_version_and_conventions():
    table = _table()
    assert table["version"] == _TABLE_VERSION
    assert tuple(table["component_order"]) == _COMPONENT_ORDER
    conventions = dict(table["conventions"])
    for key in ("racah", "real_transform", "interaction_tensor", "reality_phase",
                "energy", "block_product"):
        assert conventions[key]
    # The block product carries the 1/(2 pi); the table therefore carries the bare
    # binomial. Stating it in the table is what makes the two halves auditable.
    assert "2 pi" in conventions["block_product"]


def test_recoupling_table_has_the_expected_size():
    table = _table()
    assert table["entry_count"] == _RECOUPLING_ENTRY_COUNT
    assert table["label_count"] == _INTERNAL_LABEL_COUNT
    assert table["coupling_matrix_count"] == _COUPLING_MATRIX_COUNT
    assert len(table["labels"]) == _INTERNAL_LABEL_COUNT
    assert len(table["entries"]) == _RECOUPLING_ENTRY_COUNT


def test_recoupling_table_spans_every_order_from_six_to_fourteen():
    """An L3 model produces C6..C14, odd orders included; n = la + la' + lb + lb' + 2."""
    table = _table()
    per_order = {}
    for label in table["labels"]:
        per_order[label[0]] = per_order.get(label[0], 0) + 1
    assert per_order == _LABELS_PER_ORDER
    published = sum(count for order, count in _LABELS_PER_ORDER.items() if order <= 12)
    assert published == _PUBLISHED_LABEL_COUNT


def test_recoupling_table_has_the_expected_distinct_rank_quadruples():
    table = _table()
    quadruples = {(label[0], label[1], label[3], label[5]) for label in table["labels"]}
    assert len(quadruples) == _DISTINCT_RANK_QUADRUPLES


def test_recoupling_table_labels_are_sorted_and_unique():
    labels = [tuple(label) for label in _table()["labels"]]
    assert labels == sorted(labels)
    assert len(set(labels)) == len(labels)


def test_recoupling_table_order_six_scalars_match_their_closed_forms():
    entries = {(entry["first_rank"], entry["second_rank"], entry["coupled_rank"]):
               entry["scalar"] for entry in _table()["entries"] if entry["order"] == 6}
    assert set(entries) == set(_N6_SCALARS)
    for key, expected in _N6_SCALARS.items():
        assert abs(entries[key] - expected) <= 1.0e-13 * abs(expected)


@pytest.mark.parametrize("first_rank,second_rank,binomial", [
    (1, 1, 6.0), (1, 2, 15.0), (2, 1, 15.0), (1, 3, 28.0), (3, 1, 28.0),
    (2, 2, 70.0), (2, 3, 210.0), (3, 2, 210.0),
])
def test_recoupling_table_isotropic_reduction_returns_the_bare_binomial(
        first_rank, second_rank, binomial):
    """W[n, 00, 00, 0] traced over the diagonal rank blocks equals binom, not binom/(2 pi).

    The 1/(2 pi) is spent inside M, so the table carries the bare binomial.  This is
    B.5 check 1 with the spec's double-counted 2 pi removed; asserting binom/(2 pi)
    here would make every coefficient 2 pi too small.
    """
    order = 2 * (first_rank + second_rank + 1)
    dense = _dense_recoupling((order, 0, 0, 0, 0, 0))
    first_lo, first_hi = _RANK_SLICE[first_rank]
    second_lo, second_hi = _RANK_SLICE[second_rank]
    traced = 0.0
    for first in range(first_lo, first_hi):
        for second in range(second_lo, second_hi):
            traced += dense[((first * _L3_DIMENSION + first) * _L3_DIMENSION + second)
                            * _L3_DIMENSION + second]
    assert abs(traced / binomial - 1.0) < 1.0e-13


def test_recoupling_table_isotropic_label_matches_the_isotropic_engine():
    """C_n[00 00 0] must equal compute_dispersion's C6..C12 to machine precision."""
    models = _synthetic_models((51, 52))
    reference = _isotropic_dispersion(models)
    coefficients = _coefficients([frequency[0] for frequency in models],
                                 [frequency[1] for frequency in models])
    index = _label_index(_table())
    for order in _ISOTROPIC_ORDERS:
        expected = reference["c%d" % order].get(0, 1)
        actual = coefficients[index[(order, 0, 0, 0, 0, 0)]]
        assert abs(actual / expected - 1.0) < 1.0e-14


@pytest.mark.parametrize("label", [
    (6, 0, 0, 0, 0, 0),
    (6, 2, 4, 2, 4, 0),
    (7, 2, 4, 3, 6, 1),
    (8, 2, 4, 4, 8, 2),
    (9, 2, 4, 5, 10, 3),
    (10, 2, 4, 6, 12, 4),
    (11, 3, 6, 6, 12, 3),
    (12, 4, 8, 6, 12, 2),
    (13, 5, 10, 6, 12, 1),
    (14, 6, 12, 6, 12, 0),
])
def test_recoupling_factorised_contraction_equals_the_dense_contraction(label):
    """The factorised (g^r, Pcheck) form is exact, not an approximation of a dense W."""
    models = _synthetic_models((61, 62))
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    product = _block_product(first, second)["values"]
    dense = _dense_recoupling(label)
    contracted = sum(a * b for a, b in zip(dense, product))
    coefficients = _coefficients(first, second)
    factorised = coefficients[_label_index(_table())[label]]
    scale = max(abs(value) for value in coefficients)
    assert abs(factorised - contracted) <= 1.0e-13 * scale


@pytest.mark.parametrize("label", [
    (6, 0, 0, 2, 3, 2),
    (7, 1, 1, 2, 1, 1),
    (8, 1, 1, 3, 1, 2),
    (9, 2, 4, 5, 10, 3),
    (11, 1, 1, 2, 1, 1),
    (12, 4, 8, 6, 12, 2),
    (14, 6, 12, 6, 12, 0),
])
def test_recoupling_table_exchange_rule_carries_minus_one_to_the_l1_plus_l2(label):
    """W^{l2 k2, l1 k1} with the site blocks transposed = (-1)^{l1+l2} W^{l1 k1, l2 k2}.

    The sign is (-1)^{l1 + l2}, not (-1)^j.  Because L1 + L2 + j is even and
    L1 + L2 = n - 2, j always has the parity of n, so (-1)^j is a different factor
    and using it fails by a full sign on part of the table.
    """
    order, first_rank, first_component, second_rank, second_component, coupled = label
    forward = _dense_recoupling(label)
    mirror = _dense_recoupling((order, second_rank, second_component,
                                first_rank, first_component, coupled))
    sign = -1.0 if (first_rank + second_rank) % 2 else 1.0
    scale = max(abs(value) for value in forward)
    assert scale > 0.0
    worst = 0.0
    for t in range(_L3_DIMENSION):
        for tp in range(_L3_DIMENSION):
            for u in range(_L3_DIMENSION):
                for up in range(_L3_DIMENSION):
                    lhs = mirror[((u * _L3_DIMENSION + up) * _L3_DIMENSION + t)
                                 * _L3_DIMENSION + tp]
                    rhs = sign * forward[((t * _L3_DIMENSION + tp) * _L3_DIMENSION + u)
                                         * _L3_DIMENSION + up]
                    worst = max(worst, abs(lhs - rhs))
    assert worst / scale < 1.0e-12


def test_recoupling_coefficients_obey_the_exchange_rule_over_the_whole_label_set():
    """C^{BA}[l2 k2, l1 k1, j] = (-1)^{l1+l2} C^{AB}[l1 k1, l2 k2, j], all 29762 labels."""
    models = _synthetic_models((71, 72))
    first = [frequency[0] for frequency in models]
    second = [frequency[1] for frequency in models]
    forward = _coefficients(first, second)
    backward = _coefficients(second, first)
    table = _table()
    index = _label_index(table)
    scale = max(abs(value) for value in forward)
    worst = 0.0
    for slot, label in enumerate(table["labels"]):
        order, first_rank, first_component, second_rank, second_component, coupled = label
        mirror = (order, second_rank, second_component, first_rank, first_component, coupled)
        assert mirror in index
        sign = -1.0 if (first_rank + second_rank) % 2 else 1.0
        worst = max(worst, abs(backward[index[mirror]] - sign * forward[slot]))
    assert worst / scale < 1.0e-13


def test_recoupling_table_label_set_is_closed_under_the_exchange():
    table = _table()
    labels = {tuple(label) for label in table["labels"]}
    for order, first_rank, first_component, second_rank, second_component, coupled in labels:
        assert (order, second_rank, second_component, first_rank, first_component,
                coupled) in labels


def test_recoupling_table_live_order_six_triples_are_the_textbook_set():
    """A symmetric L3 model leaves exactly the standard anisotropic C6 label set."""
    models = _synthetic_models((81, 82))
    coefficients = _coefficients([frequency[0] for frequency in models],
                                 [frequency[1] for frequency in models])
    table = _table()
    scale = max(abs(value) for value in coefficients)
    live = set()
    for slot, label in enumerate(table["labels"]):
        if label[0] == 6 and abs(coefficients[slot]) > 1.0e-12 * scale:
            live.add((label[1], label[3], label[5]))
    assert tuple(sorted(live)) == _LIVE_C6_TRIPLES


def test_recoupling_table_orientational_average_returns_the_isotropic_coefficient():
    """<S_label> = delta_{label, (n, 0, 0, 0, 0, 0)} under SO(3) x SO(3) x S^2 averaging.

    Checked by explicit quadrature over the whole label set, exact for l <= 6 in the
    Euler angles and for j <= 12 on the sphere, then propagated to the coefficients.
    """
    models = _synthetic_models((91, 92))
    result = psi4.core._atomic_polarizability_test_anisotropic_orientational_average(
        [_matrix(frequency[0]) for frequency in models],
        [_matrix(frequency[1]) for frequency in models],
        list(_REVIEWED_WEIGHTS))
    assert result["max_s_function_deviation"] < 1.0e-12
    assert result["max_isotropic_deviation"] < 1.0e-12
    assert result["label_count"] == _INTERNAL_LABEL_COUNT


def test_recoupling_table_rank_completeness_covers_every_l3_rank():
    """A missing rank is an error, not a zero contribution, as in the isotropic engine."""
    entries = _table()["entries"]
    for key in ("first_site_rank", "first_site_rank_prime",
                "second_site_rank", "second_site_rank_prime"):
        assert {entry[key] for entry in entries} == {1, 2, 3}


def test_recoupling_table_reports_its_generation_residuals():
    table = _table()
    assert table["max_collapse_residual"] < 1.0e-13
    assert table["max_rotation_orthogonality_deviation"] < 1.0e-12


@pytest.mark.parametrize("mutation", _TABLE_REJECTIONS)
def test_recoupling_table_loader_refuses_a_structurally_invalid_table(mutation):
    """Following validate_dispersion_rank_pairs: fail closed, never trust the table."""
    with pytest.raises(RuntimeError):
        psi4.core._atomic_polarizability_test_anisotropic_table_rejection(mutation)


def test_recoupling_table_rejection_seam_rejects_an_unknown_mutation():
    with pytest.raises(RuntimeError):
        psi4.core._atomic_polarizability_test_anisotropic_table_rejection("not-a-mutation")


def test_recoupling_table_accepts_the_generated_table():
    """The generated table itself must pass every invariant the loader enforces."""
    assert psi4.core._atomic_polarizability_test_anisotropic_table_rejection("none") is True
