import math
from pathlib import Path

import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

_PUBLIC_ARRAYS = (
    "ATOMIC POLARIZABILITIES",
    "ATOMIC DYNAMIC POLARIZABILITIES",
    "ATOMIC POLARIZABILITY FREQUENCIES",
    "ATOMIC C6",
    "ATOMIC C8",
    "ATOMIC C10",
    "ATOMIC C12",
)

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


def _as_packed_rows(matrix):
    return [matrix.get(row, column) for row in range(3) for column in range(3)]


def test_casimir_frequency_grid_matches_reviewed_eleven_point_protocol_exactly():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.5)
    assert tuple(frequencies) == _REVIEWED_FREQUENCIES
    assert tuple(weights) == _REVIEWED_WEIGHTS


def test_casimir_frequency_grid_scales_frequencies_and_weights():
    frequencies, weights = psi4.core._atomic_polarizability_make_casimir_grid(10, 0.25)
    assert tuple(frequencies) == tuple(value * 0.5 for value in _REVIEWED_FREQUENCIES)
    assert tuple(weights) == tuple(value * 0.5 for value in _REVIEWED_WEIGHTS)


@pytest.mark.parametrize("nonzero_count", [0, 9, 11])
def test_casimir_frequency_grid_requires_ten_nonzero_points(nonzero_count):
    with pytest.raises(RuntimeError, match=r"exactly ten nonzero"):
        psi4.core._atomic_polarizability_make_casimir_grid(nonzero_count, 0.5)


@pytest.mark.parametrize(
    "scale",
    [
        0.0,
        -0.5,
        float("inf"),
        float("nan"),
        float.fromhex("0x1.fffffffffffffp+1023"),
        float.fromhex("0x0.0000000000001p-1022"),
    ],
)
def test_casimir_frequency_grid_rejects_invalid_scale(scale):
    with pytest.raises(RuntimeError, match=r"finite and positive"):
        psi4.core._atomic_polarizability_make_casimir_grid(10, scale)


def test_casimir_frequency_grid_rejects_weight_overflow():
    weight_overflow_scale = float.fromhex("0x1.fffffffffffffp+1023") / 100.0
    assert math.isfinite(weight_overflow_scale * (_REVIEWED_FREQUENCIES[-1] / 0.5))
    with pytest.raises(RuntimeError, match=r"finite and positive at every grid point"):
        psi4.core._atomic_polarizability_make_casimir_grid(10, weight_overflow_scale)


def test_local_spherical_dipole_maps_10_11c_11s_to_z_x_y():
    spherical = [[0.0] * 15 for _ in range(15)]
    dipole_block = (
        (1.0, 2.0, 3.0),
        (2.0, 4.0, 5.0),
        (3.0, 5.0, 6.0),
    )
    for row in range(3):
        for column in range(3):
            spherical[row][column] = dipole_block[row][column]

    cartesian = psi4.core._atomic_polarizability_local_spherical_dipole_to_cartesian(_matrix(spherical))
    assert _as_packed_rows(cartesian) == pytest.approx(
        [4.0, 5.0, 2.0, 5.0, 6.0, 3.0, 2.0, 3.0, 1.0]
    )


def test_rotate_tensor_applies_global_r_local_rt_with_right_handed_frame():
    local = _matrix([[2.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 4.0]])
    rotation = _matrix([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    rotated = psi4.core._atomic_polarizability_rotate_tensor(local, rotation)
    assert _as_packed_rows(rotated) == pytest.approx(
        [3.0, -0.5, 0.0, -0.5, 2.0, 0.0, 0.0, 0.0, 4.0]
    )


@pytest.mark.parametrize(
    "rotation",
    [
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_rotate_tensor_rejects_invalid_frames(rotation):
    with pytest.raises(RuntimeError, match=r"orthonormal right-handed"):
        psi4.core._atomic_polarizability_rotate_tensor(
            _matrix([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            _matrix(rotation),
        )


def test_pack_symmetric_tensor_uses_xx_xy_xz_yy_yz_zz_order():
    tensor = _matrix([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert psi4.core._atomic_polarizability_pack_symmetric_tensor(tensor) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )


def test_tensor_algebra_rejects_asymmetry_and_nonfinite_values():
    asymmetric = _matrix([[1.0, 2.0, 0.0], [2.1, 3.0, 0.0], [0.0, 0.0, 4.0]])
    with pytest.raises(RuntimeError, match=r"finite symmetric"):
        psi4.core._atomic_polarizability_pack_symmetric_tensor(asymmetric)

    nonfinite = _matrix([[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(RuntimeError, match=r"finite symmetric"):
        psi4.core._atomic_polarizability_rotate_tensor(
            nonfinite, _matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )


def _working_l3_matrix():
    return [[0.0] * 16 for _ in range(16)]


def _lw_localize(positions, values, bonds, tolerance=1.0e-9):
    return psi4.core._atomic_polarizability_localize_lw(
        _matrix(positions), [_matrix(block) for block in values], bonds, tolerance
    )


def test_lw_graph_operator_is_symmetric_with_one_null_mode_for_connected_chain():
    operator, pseudoinverse, eigenvalues = psi4.core._atomic_polarizability_lw_graph_math(
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


def test_lw_rank3_translation_matches_arbitrary_displacement_fixtures():
    displacement = [0.2, -0.3, 0.4]
    expected_scalar = [
        2.0, 0.8, 0.4, -0.6, 0.19, 0.277128129211, -0.415692193817,
        -0.0866025403784, -0.207846096908, -0.028, 0.124923976882,
        -0.187385965323, -0.0774596669241, -0.185903200618,
        -0.0727323861839, -0.0142302494708,
    ]
    assert psi4.core._atomic_polarizability_translate_l3(
        [2.0] + [0.0] * 15, displacement
    ) == pytest.approx(expected_scalar, abs=5.0e-12)
    expected_dense = [
        0.1, 0.24, 0.32, 0.37, 0.7295, 0.890984535672, 0.852420471066,
        1.10743901834, 0.872287187079, 1.88348457268, 2.29457302188,
        1.30348504869, 3.00906108622, 2.04427885923, 2.75223325142,
        1.30215605855,
    ]
    assert psi4.core._atomic_polarizability_translate_l3(
        [0.1 * value for value in range(1, 17)], displacement
    ) == pytest.approx(expected_dense, abs=5.0e-11)


def test_lw_two_site_charge_flow_localizes_to_full_rank3_fixture():
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0], values[1][0][0] = -2.0, 2.0
    values[2][0][0], values[3][0][0] = 2.0, -2.0
    result = _lw_localize([[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]], values, [(0, 1)])
    local = result["local"]
    assert len(local) == 2
    assert local[0].shape == (15, 15)
    assert local[0].get(0, 0) == pytest.approx(-0.16, abs=1.0e-11)
    assert local[0].get(0, 1) == pytest.approx(-0.08, abs=1.0e-11)
    assert local[0].get(0, 2) == pytest.approx(0.12, abs=1.0e-11)
    assert local[0].get(0, 3) == pytest.approx(-0.038, abs=1.0e-11)
    assert local[1].get(0, 3) == pytest.approx(0.038, abs=1.0e-11)
    assert local[0].get(14, 14) == pytest.approx(-0.000050625, abs=1.0e-12)
    assert local[1].get(14, 14) == pytest.approx(-0.000050625, abs=1.0e-12)
    assert max(result["residuals"]) < 1.0e-10


def test_lw_three_site_preserves_sum_reciprocity_and_transfers_only_on_bonds():
    positions = [[0.0, 0.0, 0.0], [0.7, -0.2, 0.1], [1.1, 0.4, -0.3]]
    graph_operator = [[-1.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -1.0]]
    values = [_working_l3_matrix() for _ in range(9)]
    for a in range(3):
        for b in range(3):
            values[3 * a + b][0][0] = 1.7 * graph_operator[a][b]
    result = _lw_localize(positions, values, [(0, 1), (1, 2)], 2.0e-9)
    assert max(result["residuals"]) < 2.0e-9
    assert all(
        (first, second) in {(0, 1), (1, 2)}
        for first, second, _mu, _nu, _site, _amount in result["transfers"]
    )
    for matrix in result["local"]:
        assert all(
            matrix.get(row, column) == pytest.approx(matrix.get(column, row), abs=2.0e-9)
            for row in range(15) for column in range(15)
        )


def test_lw_rejects_disconnected_graph_and_postcondition_residual():
    with pytest.raises(RuntimeError, match=r"connected bond graph"):
        _lw_localize(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [_working_l3_matrix() for _ in range(9)], [(0, 1)]
        )
    values = [_working_l3_matrix() for _ in range(4)]
    values[0][0][0], values[1][0][0] = -2.0 + 1.0e-5, 2.0
    values[2][0][0], values[3][0][0] = 2.0, -2.0
    with pytest.raises(RuntimeError, match=r"residual tolerance"):
        _lw_localize([[0.0, 0.0, 0.0], [0.2, -0.3, 0.4]], values, [(0, 1)], 1.0e-9)


def test_atomic_polarizabilities_api_is_registered():
    assert "ATOMIC_POLARIZABILITIES" in psi4.core.OEProp.valid_methods
    assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_N_FREQUENCIES") == 10
    assert psi4.core.get_global_option("ATOMIC_POLARIZABILITY_FREQUENCY_SCALE") == pytest.approx(0.5)


def test_atomic_polarizabilities_fail_closed_without_response_data():
    molecule = psi4.geometry(
        """
        H 0.0 0.0 0.0
        H 0.0 0.0 0.7
        symmetry c1
        units angstrom
        """
    )
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk"})
    _, wfn = psi4.energy("scf", return_wfn=True)

    oeprop = psi4.core.OEProp(wfn)
    oeprop.add("MULTIPOLE(2)")
    oeprop.add("ATOMIC_POLARIZABILITIES")

    with pytest.raises(RuntimeError, match=r"AtomicPolarizabilityCalculator.*response data"):
        oeprop.compute()

    unpublished = ("DIPOLE", "QUADRUPOLE", *_PUBLIC_ARRAYS)
    assert all(not wfn.has_array_variable(name) for name in unpublished)
    assert all(not psi4.core.has_array_variable(name) for name in unpublished)


def test_atomic_polarizabilities_reject_incomplete_wavefunction_prerequisites():
    molecule = psi4.geometry(
        """
        He 0.0 0.0 0.0
        symmetry c1
        """
    )
    wfn = psi4.core.Wavefunction.build(molecule, "sto-3g")
    calculator = psi4.core.AtomicPolarizabilityCalculator(wfn)

    with pytest.raises(RuntimeError, match=r"unsupported wavefunction.*orbital response data"):
        calculator.compute()

    assert all(not wfn.has_array_variable(name) for name in _PUBLIC_ARRAYS)


def test_native_atomic_polarizability_source_guard():
    from test_native_atomic_polarizability_source_guard import source_violations

    repo_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "psi4/src/psi4/libmints/atomic_polarizability.cc").is_file()
    )
    native_sources = (
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.cc",
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.h",
        repo_root / "psi4/src/psi4/libmints/oeprop.cc",
        repo_root / "psi4/src/psi4/libmints/oeprop.h",
        repo_root / "psi4/src/export_oeprop.cc",
    )

    violations = []
    for source in native_sources:
        violations.extend(f"{source.name}: {item}" for item in source_violations(source.read_text()))

    cmake_text = (repo_root / "psi4/src/psi4/libmints/CMakeLists.txt").read_text()
    assert "atomic_polarizability.cc" in cmake_text
    assert violations == []

    canary = 'void launch() { std::system("camcasp"); }'
    assert source_violations(canary) == ["forbidden process API: std::system(", "forbidden external term: camcasp"]
