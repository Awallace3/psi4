"""Deterministic, symmetry-faithful WSM fit-point generation tests.

The generated set is a family of nested equidistant (offset) surfaces around the
nuclear framework, sampled with Psi4's own Lebedev angular grids.  Every test
here is independent of SCF, of any external program, and of any reviewed oracle
literal; the only fixed numbers are published Lebedev sizes and Bondi radii.
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]


# Reviewed H2O frame: C2 along z, molecule in the xz plane.
_WATER_CENTERS = (
    (0.0, 0.0, 0.0),
    (1.45365196, 0.0, -1.12168732),
    (-1.45365196, 0.0, -1.12168732),
)
_WATER_NUMBERS = (8, 1, 1)

# Psi4's C2v symmetry operations in that frame: E, C2(z), sigma(xz), sigma(yz).
_C2V_OPERATIONS = (
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
)

def _matrix(values):
    return psi4.core.Matrix.from_array(np.asarray(values, dtype=float))


def _octahedral_operations():
    """The 48 signed coordinate permutations, i.e. the full O_h matrix group."""
    operations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            operations.append(matrix)
    return operations


def _lebedev(npoints):
    return np.asarray(psi4.core._atomic_polarizability_lebedev_unit_sphere(npoints))


def _plan(atom_count, **options):
    return psi4.core._atomic_polarizability_plan_fit_points(atom_count, options)


def _generate(numbers=_WATER_NUMBERS, centers=_WATER_CENTERS,
              operations=_C2V_OPERATIONS, frame=None, **options):
    result = psi4.core._atomic_polarizability_generate_fit_points(
        list(numbers), _matrix(centers), [_matrix(x) for x in operations],
        None if frame is None else _matrix(frame), options)
    result["points"] = np.asarray(result["points"])
    return result


def _water_molecule():
    molecule = psi4.geometry(
        """
        units bohr
        symmetry c2v
        no_reorient
        no_com
        O 0.0        0.0  0.0
        H 1.45365196 0.0 -1.12168732
        H -1.45365196 0.0 -1.12168732
        """
    )
    molecule.update_geometry()
    return molecule


def _sets_match(first, second, tolerance=1.0e-11):
    """True when two point arrays are equal as sets to the given tolerance."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        return False
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return bool(np.all(distances.min(axis=1) < tolerance)
                and np.all(distances.min(axis=0) < tolerance))


def _max_set_deviation(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return max(distances.min(axis=1).max(), distances.min(axis=0).max())


# ==> reused angular-grid facility <== #


def test_lebedev_unit_sphere_reuses_psi4_grid_tables_and_is_octahedrally_invariant():
    for npoints in (26, 50, 74, 302):
        grid = _lebedev(npoints)
        assert grid.shape == (npoints, 4)
        directions, weights = grid[:, :3], grid[:, 3]
        assert np.linalg.norm(directions, axis=1) == pytest.approx(1.0, abs=1e-14)
        assert weights.sum() == pytest.approx(4.0 * math.pi, rel=1e-13)
        # Only the directions are used here; some orders (74) carry negative weights.
        assert np.all(np.isfinite(weights))
        for operation in _octahedral_operations():
            assert _sets_match(directions @ operation.T, directions, 1e-13)


def test_lebedev_unit_sphere_rejects_unsupported_sizes():
    for npoints in (0, 2, 18, 51, 100, -6):
        with pytest.raises(RuntimeError, match="supported Lebedev"):
            _lebedev(npoints)


# ==> planning and storage bounds <== #


def test_fit_point_plan_bounds_the_candidate_set_before_generation():
    plan = _plan(3, spherical_points=50, radial_shells=5)
    assert plan["atom_count"] == 3
    assert plan["spherical_points"] == 50
    assert plan["radial_shells"] == 5
    assert plan["lebedev_order"] == 11
    assert plan["candidate_count"] == 3 * 50 * 5
    assert plan["shell_offsets"] == pytest.approx([4.5, 6.25, 8.0, 9.75, 11.5], abs=1e-14)
    assert plan["maximum_points"] == 500
    assert plan["candidate_bytes"] == plan["candidate_count"] * 24
    assert plan["estimated_bytes"] >= plan["candidate_bytes"]
    assert plan["algorithm"] == "nested_equidistant_lebedev_surfaces"
    assert plan["radial_units"] == "BOHR"


def test_fit_point_plan_single_shell_uses_the_inner_limit_only():
    plan = _plan(1, spherical_points=26, radial_shells=1, inner_limit=2.5, outer_limit=4.0)
    assert plan["shell_offsets"] == pytest.approx([2.5], abs=1e-14)


def test_fit_point_plan_matches_the_generated_set():
    result = _generate(spherical_points=50, radial_shells=5)
    plan = result["plan"]
    assert plan["candidate_count"] == 3 * 50 * 5
    assert len(result["points"]) == plan["point_count"]
    assert plan["point_count"] <= plan["candidate_count"]
    assert plan["point_count"] <= plan["maximum_points"]
    assert len(result["shell_index"]) == plan["point_count"]
    assert len(result["generator_atom"]) == plan["point_count"]
    assert len(result["nearest_offsets"]) == plan["point_count"]


def test_generated_water_default_grid_stays_inside_the_wsm_refinement_envelope():
    result = _generate()
    count = len(result["points"])
    assert 250 <= count <= 500
    envelope = psi4.core._atomic_polarizability_plan_wsm_refinement(
        count, 3, 360, 0, 8 * 1024 ** 3)
    assert envelope["point_count"] == count
    assert envelope["pair_rows"] == count * (count + 1) // 2


# ==> symmetry faithfulness <== #


def test_generated_points_are_invariant_under_the_full_c2v_point_group():
    result = _generate(spherical_points=74, radial_shells=4)
    points = result["points"]
    for operation in _C2V_OPERATIONS:
        rotated = points @ np.asarray(operation, dtype=float).T
        assert _max_set_deviation(rotated, points) < 1e-12


def test_generated_points_report_verified_invariance_and_octahedral_alignment():
    result = _generate(spherical_points=74, radial_shells=4)
    assert result["plan"]["symmetry_operation_count"] == 4
    assert result["max_symmetry_deviation"] < 1e-12
    assert result["max_octahedral_deviation"] == 0.0


def test_generation_fails_closed_when_operations_are_not_octahedral_in_the_angular_frame():
    """A C2 about a general axis cannot leave a Lebedev node set invariant."""
    axis = np.asarray([0.3, -0.5, 0.81])
    axis = axis / np.linalg.norm(axis)
    c2 = 2.0 * np.outer(axis, axis) - np.eye(3)
    centers = [(0.0, 0.0, 0.0), tuple(1.6 * axis), tuple(-1.6 * axis)]
    with pytest.raises(RuntimeError, match="signed coordinate permutation"):
        _generate(numbers=(8, 1, 1), centers=centers, operations=(np.eye(3), c2),
                  spherical_points=26, radial_shells=2)


def test_supplying_the_symmetry_frame_restores_invariance_for_a_rotated_molecule():
    """The same off-axis C2 becomes exact once the Lebedev axes follow the molecule."""
    axis = np.asarray([0.3, -0.5, 0.81])
    axis = axis / np.linalg.norm(axis)
    # A proper rotation carrying z onto the molecular axis.
    helper = np.asarray([1.0, 0.0, 0.0])
    first = np.cross(axis, helper)
    first = first / np.linalg.norm(first)
    second = np.cross(axis, first)
    frame = np.column_stack((first, second, axis))
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-14)

    c2 = 2.0 * np.outer(axis, axis) - np.eye(3)
    centers = [(0.0, 0.0, 0.0), tuple(1.6 * axis), tuple(-1.6 * axis)]
    result = _generate(numbers=(8, 1, 1), centers=centers, operations=(np.eye(3), c2),
                       frame=frame, spherical_points=26, radial_shells=2)
    assert result["max_symmetry_deviation"] < 1e-12
    for operation in (np.eye(3), c2):
        assert _max_set_deviation(result["points"] @ operation.T, result["points"]) < 1e-11


def test_generated_points_are_deterministic_and_bit_identical_across_calls():
    first = _generate(spherical_points=50, radial_shells=5)
    second = _generate(spherical_points=50, radial_shells=5)
    assert np.array_equal(first["points"], second["points"])
    assert first["shell_index"] == second["shell_index"]
    assert first["generator_atom"] == second["generator_atom"]


def test_generated_points_are_orientation_equivariant_under_rotation():
    angle = 0.37
    axis = np.asarray([0.3, -0.5, 0.81])
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    rotation = (np.eye(3) + math.sin(angle) * cross
                + (1.0 - math.cos(angle)) * (cross @ cross))
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)

    reference = _generate(spherical_points=50, radial_shells=3)
    rotated = _generate(
        centers=[tuple(rotation @ np.asarray(center)) for center in _WATER_CENTERS],
        operations=[rotation @ np.asarray(operation) @ rotation.T
                    for operation in _C2V_OPERATIONS],
        frame=rotation, spherical_points=50, radial_shells=3)

    # Equivariance holds point by point and in order, not merely as a set.
    assert len(rotated["points"]) == len(reference["points"])
    assert np.max(np.abs(rotated["points"] - reference["points"] @ rotation.T)) < 1e-12
    assert rotated["nearest_offsets"] == pytest.approx(reference["nearest_offsets"], abs=1e-12)
    assert rotated["shell_index"] == reference["shell_index"]
    assert rotated["generator_atom"] == reference["generator_atom"]


# ==> shell geometry <== #


def test_generated_points_exclude_the_region_inside_the_inner_shell_limit():
    inner = 2.0
    result = _generate(spherical_points=74, radial_shells=4, inner_limit=inner)
    centers = np.asarray(_WATER_CENTERS)
    distances = np.linalg.norm(result["points"][:, None, :] - centers[None, :, :], axis=2)
    assert distances.min() >= inner - 1e-12
    assert np.all(distances > 0.0)


def test_generated_points_lie_exactly_on_the_requested_nested_surfaces():
    inner, outer, shells = 2.0, 4.0, 5
    result = _generate(spherical_points=50, radial_shells=shells,
                       inner_limit=inner, outer_limit=outer)
    centers = np.asarray(_WATER_CENTERS)
    distances = np.linalg.norm(result["points"][:, None, :] - centers[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    expected = np.linspace(inner, outer, shells)

    assert nearest == pytest.approx(result["nearest_offsets"], abs=1e-12)
    assert nearest.min() == pytest.approx(inner, abs=1e-12)
    assert nearest.max() == pytest.approx(outer, abs=1e-12)
    for point, shell in zip(nearest, result["shell_index"]):
        assert point == pytest.approx(expected[shell], abs=1e-12)
    assert set(result["shell_index"]) == set(range(shells))


def test_vdw_units_scale_every_shell_by_the_bondi_radius_of_the_nearest_atom():
    bohr = psi4.constants.bohr2angstroms
    bondi = {1: 1.20 / bohr, 8: 1.52 / bohr}
    result = _generate(spherical_points=50, radial_shells=2,
                       inner_limit=2.0, outer_limit=3.0, radial_units="VDW")
    assert result["plan"]["radial_units"] == "VDW"
    assert result["scaling_radii"] == pytest.approx(
        [bondi[number] for number in _WATER_NUMBERS], rel=1e-12)

    centers = np.asarray(_WATER_CENTERS)
    radii = np.asarray([bondi[number] for number in _WATER_NUMBERS])
    reduced = (np.linalg.norm(result["points"][:, None, :] - centers[None, :, :], axis=2)
               / radii[None, :]).min(axis=1)
    assert reduced.min() == pytest.approx(2.0, abs=1e-12)
    assert reduced.max() == pytest.approx(3.0, abs=1e-12)


def test_radial_spacing_defaults_to_linear_and_equal_volume_uses_the_volume_quantiles():
    """EQUAL_VOLUME puts the shells at the equal-volume quantiles of the shell region.

    The reviewed spacing is LINEAR and stays the default. EQUAL_VOLUME exists because a
    volume-uniform cloud -- which is what a random sample of the shell region is -- has a
    shell-offset density that grows as t**2, not a flat one. Placing K shells at
    cbrt(t_in**3 + k/(K-1) (t_out**3 - t_in**3)) and keeping the equal Lebedev count per
    shell reproduces that density deterministically, so the construction stays
    symmetry-faithful and RNG-free.
    """
    inner, outer, shells = 2.0, 4.0, 5
    linear = _plan(3, spherical_points=50, radial_shells=shells,
                   inner_limit=inner, outer_limit=outer)
    assert linear["radial_spacing"] == "LINEAR"
    assert linear["shell_offsets"] == pytest.approx([2.0, 2.5, 3.0, 3.5, 4.0], abs=1e-14)

    equal = _plan(3, spherical_points=50, radial_shells=shells,
                  inner_limit=inner, outer_limit=outer, radial_spacing="EQUAL_VOLUME")
    assert equal["radial_spacing"] == "EQUAL_VOLUME"
    quantiles = np.cbrt(np.linspace(inner ** 3, outer ** 3, shells))
    assert equal["shell_offsets"] == pytest.approx(quantiles, abs=1e-14)
    # Both endpoints stay exact, so the limit interval is closed under either spacing.
    assert equal["shell_offsets"][0] == pytest.approx(inner, abs=0.0)
    assert equal["shell_offsets"][-1] == pytest.approx(outer, abs=0.0)

    # A single shell has no interval to distribute, so the spacing cannot matter.
    for spacing in ("LINEAR", "EQUAL_VOLUME"):
        single = _plan(3, spherical_points=50, radial_shells=1,
                       inner_limit=inner, outer_limit=outer, radial_spacing=spacing)
        assert single["shell_offsets"] == pytest.approx([inner], abs=1e-14)


def test_equal_volume_spacing_converges_on_the_volume_uniform_mean_offset():
    """Only EQUAL_VOLUME approaches the mean offset of a volume-uniform cloud.

    A cloud sampled uniformly by volume between offsets t_in and t_out has mean offset
    (t_out**4 - t_in**4) / 4 / ((t_out**3 - t_in**3) / 3); on [2, 4] that is 3.2143. An
    equal count on equally spaced shells averages 3.0 instead and does not improve with
    shell count, because the error is in the weighting rather than the resolution.
    """
    inner, outer = 2.0, 4.0
    target = ((outer ** 4 - inner ** 4) / 4.0) / ((outer ** 3 - inner ** 3) / 3.0)
    assert target == pytest.approx(3.2143, abs=5e-5)

    previous = None
    for shells in (5, 6, 8):
        offsets = np.asarray(_plan(3, spherical_points=50, radial_shells=shells,
                                   inner_limit=inner, outer_limit=outer,
                                   radial_spacing="EQUAL_VOLUME")["shell_offsets"])
        linear = np.asarray(_plan(3, spherical_points=50, radial_shells=shells,
                                  inner_limit=inner, outer_limit=outer)["shell_offsets"])
        # Equally spaced shells always average the midpoint of the interval.
        assert linear.mean() == pytest.approx(0.5 * (inner + outer), abs=1e-14)
        gap = abs(offsets.mean() - target)
        assert gap < abs(linear.mean() - target)
        if previous is not None:
            assert gap < previous
        previous = gap


def test_equal_volume_spacing_preserves_symmetry_and_the_shell_membership_invariants():
    """The spacing changes only where the shells sit, not the construction's guarantees."""
    result = _generate(spherical_points=50, radial_shells=5, inner_limit=2.0,
                       outer_limit=4.0, radial_units="VDW",
                       radial_spacing="EQUAL_VOLUME")
    assert result["plan"]["radial_spacing"] == "EQUAL_VOLUME"
    assert result["max_symmetry_deviation"] < 1e-12
    assert result["max_octahedral_deviation"] == 0.0

    points = result["points"]
    for operation in _C2V_OPERATIONS:
        assert _sets_match(points @ np.asarray(operation).T, points)

    # Every point still lands exactly on one of the requested surfaces.
    offsets = np.asarray(result["nearest_offsets"])
    expected = np.asarray(result["plan"]["shell_offsets"])
    for offset, shell in zip(offsets, result["shell_index"]):
        assert offset == pytest.approx(expected[shell], abs=1e-12)
    assert set(result["shell_index"]) == set(range(5))
    assert offsets.min() == pytest.approx(2.0, abs=1e-12)
    assert offsets.max() == pytest.approx(4.0, abs=1e-12)


def test_bondi_radius_table_is_aligned_with_atomic_number():
    bohr = psi4.constants.bohr2angstroms
    published = {1: 1.20, 2: 1.40, 6: 1.70, 8: 1.52, 10: 1.54, 17: 1.75, 18: 1.88,
                 30: 1.39, 35: 1.85, 36: 2.02, 47: 1.72, 52: 2.06, 53: 1.98, 54: 2.16}
    for number, angstrom in published.items():
        assert psi4.core._atomic_polarizability_bondi_vdw_radius(number) == pytest.approx(
            angstrom / bohr, rel=1e-12)


def test_vdw_units_fail_closed_for_elements_outside_the_tabulated_range():
    for number in (0, 21, 55, 95):
        with pytest.raises(RuntimeError, match="van der Waals radius"):
            psi4.core._atomic_polarizability_bondi_vdw_radius(number)
    with pytest.raises(RuntimeError, match="van der Waals radius"):
        _generate(numbers=(8, 1, 95), radial_units="VDW", spherical_points=26,
                  radial_shells=2, operations=(np.eye(3),))


# ==> fail-closed behaviour <== #


@pytest.mark.parametrize("options,message", [
    ({"spherical_points": 0}, "supported Lebedev"),
    ({"spherical_points": 51}, "supported Lebedev"),
    ({"radial_shells": 0}, "at least one radial shell"),
    ({"inner_limit": 4.0, "outer_limit": 2.0}, "outer shell limit"),
    ({"inner_limit": 2.0, "outer_limit": 2.0, "radial_shells": 2}, "outer shell limit"),
    ({"inner_limit": 0.0}, "inner shell limit must be positive"),
    ({"inner_limit": -1.0}, "inner shell limit must be positive"),
    ({"inner_limit": float("nan")}, "must be finite"),
    ({"outer_limit": float("inf")}, "must be finite"),
    ({"maximum_points": 0}, "maximum point count"),
    ({"radial_units": "SURFACE"}, "radial units"),
    ({"radial_spacing": "GAUSSIAN"}, "radial spacing"),
])
def test_fit_point_generation_fails_closed_on_degenerate_options(options, message):
    with pytest.raises(RuntimeError, match=message):
        _generate(**options)


def test_fit_point_plan_fails_closed_before_any_generation_work():
    with pytest.raises(RuntimeError, match="storage ceiling"):
        _plan(3, spherical_points=5810, radial_shells=20)
    with pytest.raises(RuntimeError, match="at least one atom"):
        _plan(0)


def test_fit_point_generation_requires_consistent_geometry_input():
    with pytest.raises(RuntimeError, match="one atomic number per center"):
        psi4.core._atomic_polarizability_generate_fit_points(
            [8, 1], _matrix(_WATER_CENTERS), [_matrix(np.eye(3))], None, {})
    with pytest.raises(RuntimeError, match="at least the identity"):
        _generate(operations=())
    with pytest.raises(RuntimeError, match="angular frame must be a proper rotation"):
        _generate(frame=np.diag([1.0, 1.0, -1.0]))


@pytest.mark.parametrize("operation,message", [
    (((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "orthogonal"),
    (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), "orthogonal"),
    (((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)), "nuclear framework"),
])
def test_fit_point_generation_rejects_operations_that_are_not_molecular_symmetries(
        operation, message):
    with pytest.raises(RuntimeError, match=message):
        _generate(operations=(np.eye(3), operation))


def test_fit_point_generation_rejects_coincident_or_nonfinite_nuclei():
    with pytest.raises(RuntimeError, match="finite"):
        _generate(centers=((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0),
                           (-1.45365196, 0.0, -1.12168732)))
    with pytest.raises(RuntimeError, match="coincident"):
        _generate(numbers=(8, 8, 1), centers=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                              (0.0, 0.0, 2.0)),
                  operations=(np.eye(3),))


# ==> options plumbing and production entry point <== #


def _from_molecule(molecule):
    result = psi4.core._atomic_polarizability_wsm_fit_points(molecule)
    result["points"] = np.asarray(result["points"])
    return result


def test_wsm_fit_points_reads_the_atomic_polarizability_fit_options():
    molecule = _water_molecule()
    psi4.core.clean_options()
    default = _from_molecule(molecule)
    assert default["plan"]["spherical_points"] == 50
    assert default["plan"]["radial_shells"] == 5
    assert default["plan"]["shell_offsets"] == pytest.approx([4.5, 6.25, 8.0, 9.75, 11.5])
    assert default["plan"]["radial_units"] == "BOHR"
    assert default["plan"]["radial_spacing"] == "LINEAR"
    assert default["plan"]["maximum_points"] == 500
    assert default["plan"]["symmetry_operation_count"] == 4

    psi4.set_options({
        "atomic_polarizability_fit_spherical_points": 26,
        "atomic_polarizability_fit_radial_shells": 2,
        "atomic_polarizability_fit_inner_limit": 2.5,
        "atomic_polarizability_fit_outer_limit": 3.5,
        "atomic_polarizability_fit_max_points": 400,
        "atomic_polarizability_fit_radial_spacing": "equal_volume",
    })
    tuned = _from_molecule(molecule)
    psi4.core.clean_options()

    assert tuned["plan"]["spherical_points"] == 26
    assert tuned["plan"]["radial_shells"] == 2
    assert tuned["plan"]["radial_spacing"] == "EQUAL_VOLUME"
    assert tuned["plan"]["shell_offsets"] == pytest.approx([2.5, 3.5])
    assert tuned["plan"]["maximum_points"] == 400
    assert len(tuned["points"]) < len(default["points"])
    assert np.asarray(tuned["nearest_offsets"]).min() == pytest.approx(2.5, abs=1e-12)


def test_wsm_fit_points_from_a_molecule_matches_the_explicit_geometry_seam():
    molecule = _water_molecule()
    psi4.core.clean_options()
    from_molecule = _from_molecule(molecule)
    psi4.core.clean_options()
    explicit = _generate(spherical_points=50, radial_shells=5)
    # Psi4's own symmetrization leaves the H y coordinates at ~4e-17 rather than 0.
    assert from_molecule["points"].shape == explicit["points"].shape
    assert np.max(np.abs(from_molecule["points"] - explicit["points"])) < 1e-13
    assert from_molecule["shell_index"] == explicit["shell_index"]
    assert from_molecule["generator_atom"] == explicit["generator_atom"]


def test_wsm_fit_points_fails_closed_when_the_generated_set_exceeds_the_envelope():
    molecule = _water_molecule()
    psi4.set_options({"atomic_polarizability_fit_spherical_points": 302,
                      "atomic_polarizability_fit_radial_shells": 5})
    try:
        with pytest.raises(RuntimeError, match="maximum point count"):
            psi4.core._atomic_polarizability_wsm_fit_points(molecule)
    finally:
        psi4.core.clean_options()


# ==> convergence of a refined model under grid refinement <== #


def _regular(displacement):
    x, y, z = np.asarray(displacement, dtype=float)
    r2 = x * x + y * y + z * z
    return np.array([
        1., z, x, y,
        (3 * z * z - r2) / 2, np.sqrt(3) * x * z, np.sqrt(3) * y * z,
        np.sqrt(3) * (x * x - y * y) / 2, np.sqrt(3) * x * y,
        (5 * z ** 3 - 3 * z * r2) / 2,
        np.sqrt(3 / 8) * x * (5 * z * z - r2), np.sqrt(3 / 8) * y * (5 * z * z - r2),
        np.sqrt(15) * z * (x * x - y * y) / 2, np.sqrt(15) * x * y * z,
        np.sqrt(10) * x * (x * x - 3 * y * y) / 4, np.sqrt(10) * y * (3 * x * x - y * y) / 4,
    ])


def _irregular(point, site):
    displacement = np.asarray(point, dtype=float) - np.asarray(site, dtype=float)
    r2 = displacement @ displacement
    regular = _regular(displacement)
    values = []
    for rank, begin in ((1, 1), (2, 4), (3, 9)):
        values.extend(regular[begin:(rank + 1) ** 2] / r2 ** (rank + 0.5))
    return np.asarray(values)


def _reference_l3_model():
    """A deterministic, symmetric, positive-definite synthetic L3 model per site."""
    index = np.arange(15, dtype=float)
    ranks = np.array([1] * 3 + [2] * 5 + [3] * 7, dtype=float)
    tensors = []
    for site, scale in enumerate((6.0, 1.7, 1.7)):
        diagonal = scale / (1.0 + 0.6 * (ranks - 1.0) + 0.05 * index)
        coupling = np.cos(0.4 * (index[:, None] - index[None, :]))
        tensor = 0.08 * np.sqrt(np.outer(diagonal, diagonal)) * coupling
        np.fill_diagonal(tensor, diagonal)
        tensors.append(0.5 * (tensor + tensor.T))
    return tensors


def _synthetic_response(points, sites, tensors):
    irregular = np.array([[_irregular(point, site) for site in sites] for point in points])
    response = np.einsum("gat,atu,hau->gh", irregular, np.asarray(tensors), irregular)
    return 0.5 * (response + response.T)


@pytest.fixture
def wsm_refinement_memory():
    """The dense pair-row design needs well above Psi4's 500 MB default."""
    previous = psi4.core.get_memory()
    psi4.set_memory("8 GB")
    yield
    psi4.core.set_memory_bytes(previous)


def _fit(points, sites, tensors):
    response = _synthetic_response(points, sites, tensors)
    models = psi4.core._atomic_polarizability_test_refine_wsm(
        _matrix(points), [0.0], [_matrix(response)], _matrix(sites),
        [_matrix(tensor) for tensor in tensors], [0.0],
        [True] * (120 * len(sites)), _matrix(np.empty((0, 120 * len(sites)))), [], {})
    assert len(models) == 1
    return np.asarray([np.asarray(tensor) for tensor in models[0]["tensors"]]), models[0]


def test_refined_l3_model_is_stable_under_fit_point_refinement(wsm_refinement_memory):
    sites = np.asarray(_WATER_CENTERS)
    tensors = _reference_l3_model()

    fitted = []
    for spherical_points in (26, 38, 50):
        points = _generate(spherical_points=spherical_points, radial_shells=3)["points"]
        recovered, diagnostics = _fit(points, sites, tensors)
        assert diagnostics["condition_number"] < 1.0e12
        assert not diagnostics["pruned_variables"]
        fitted.append(recovered)

    scale = max(np.max(np.abs(tensor)) for tensor in tensors)
    for recovered in fitted:
        assert np.max(np.abs(recovered - np.asarray(tensors))) < 1.0e-6 * scale

    coarse_change = np.max(np.abs(fitted[1] - fitted[0]))
    fine_change = np.max(np.abs(fitted[2] - fitted[1]))
    assert fine_change <= max(coarse_change, 1.0e-9 * scale)
    assert fine_change < 1.0e-6 * scale


def _weighted_column_norms(points, sites):
    """Weighted design-column norms over all 120 variables of every site."""
    irregular = np.array([[_irregular(point, site) for site in sites] for point in points])
    norms = []
    for site in range(len(sites)):
        block = irregular[:, site, :]
        for first in range(15):
            for second in range(first, 15):
                outer = np.outer(block[:, first], block[:, second])
                design = outer if first == second else outer + outer.T
                norms.append(np.linalg.norm(design))
    return np.asarray(norms)


def test_the_default_grid_needs_the_relative_wsm_column_cutoff():
    """The default shell limits keep rank 3 only because the 1e-4 cutoff is relative.

    This replaces an earlier test which read the cutoff as an absolute weighted column
    norm in atomic units and, on that basis, argued the shell limits had to be small
    enough to keep the rank-3 columns above 1e-4. That inference is what put every fit
    point inside the charge density, where a rank-3 multipole model cannot represent the
    point-to-point response, and it cost 36 percent of the molecular polarizability.

    The irregular harmonics fall off as r^-(2l+1), so an absolute threshold silently makes
    the retained rank a function of how far out the points sit. `refine_wsm` therefore
    scales the policy cutoff by the largest weighted column norm. The numbers below are
    the whole argument: on the default grid the smallest rank-3 column is 2.4e-5 in
    absolute terms -- it would be pruned outright under the absolute reading -- but 2.8e-4
    relative to the largest column, so the relative reading retains it and `wsm_rank=3`
    remains satisfiable at physically valid radii.
    """
    cutoff = 1.0e-4
    sites = np.asarray(_WATER_CENTERS)
    norms = _weighted_column_norms(
        _generate(spherical_points=50, radial_shells=5)["points"], sites)

    assert norms.min() < cutoff                 # rejected by the absolute reading
    assert norms.min() > cutoff * norms.max()   # retained by the relative reading


def test_default_shell_limits_bracket_the_reviewed_point_grid_span():
    """The default band is 4.5 to 11.5 bohr from the nearest nucleus.

    The reviewed point-to-point grid spans 4.63 to 11.46 bohr from the nearest nucleus, so
    the default band brackets it, and BOHR stays the default convention so that the
    reviewed numbers remain reproducible. That absolute span is not, however, evidence
    that the reviewed grid is defined absolutely: for water it is exactly the envelope of
    a per-atom van der Waals band, 2.0 * R_Bondi(H) = 4.535 to 4.0 * R_Bondi(O) = 11.490,
    with a finite random sample undersampling both extremes. BOHR consequently flattens
    the per-atom scaling, and 4.5 bohr sits 1.245 bohr inside oxygen's own 2.0 * R_Bondi
    surface; ATOMIC_POLARIZABILITY_FIT_RADIAL_UNITS VDW with limits 2.0 and 4.0 selects
    the unflattened band. The physical requirement behind the band -- that a rank-3
    multipole model can actually reproduce the point-to-point response there -- is
    asserted against a real density by the molecular-conservation tests in
    test_atomic_polarizabilities.py.
    """
    psi4.core.clean_options()
    default = _from_molecule(_water_molecule())
    offsets = np.asarray(default["nearest_offsets"])
    assert offsets.min() == pytest.approx(4.5, abs=1e-12)
    assert offsets.max() == pytest.approx(11.5, abs=1e-12)
    assert default["plan"]["radial_units"] == "BOHR"


def test_refined_l3_model_respects_c2_site_symmetry_on_the_generated_grid(wsm_refinement_memory):
    """A C2v-invariant grid must not inject symmetry-violating anisotropy."""
    sites = np.asarray(_WATER_CENTERS)
    tensors = _reference_l3_model()
    points = _generate(spherical_points=38, radial_shells=3)["points"]
    recovered, _ = _fit(points, sites, tensors)
    scale = max(np.max(np.abs(tensor)) for tensor in tensors)
    assert np.max(np.abs(recovered[1] - recovered[2])) < 1.0e-6 * scale


# ==> source guard <== #


def test_wsm_fit_point_source_is_free_of_external_dependencies():
    from test_native_atomic_polarizability_source_guard import source_violations

    repo_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "psi4/src/psi4/libmints/atomic_polarizability.cc").is_file()
    )
    source = repo_root / "psi4/src/psi4/libmints/wsm_fit_points.cc"
    assert source.is_file()
    assert source_violations(source.read_text()) == []
    assert "wsm_fit_points.cc" in (
        repo_root / "psi4/src/psi4/libmints/CMakeLists.txt").read_text()
