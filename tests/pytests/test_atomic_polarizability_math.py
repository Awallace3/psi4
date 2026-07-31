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
