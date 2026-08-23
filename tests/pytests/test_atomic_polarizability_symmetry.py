"""Derived PDef site-symmetry constraints and covalent bond graphs, without SCF or oracles.

Characters are re-derived here numerically from the exported irregular-harmonic
evaluator, so the production integer parity table is never mirrored in the tests.
"""

from pathlib import Path

import numpy as np
import pytest

import psi4


pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.mints]

_COMPONENTS = 15
_VARIABLES_PER_SITE = 120


def _matrix(values):
    return psi4.core.Matrix.from_array(np.asarray(values, dtype=float))


def _upper_index(t, u):
    assert 0 <= t <= u < _COMPONENTS
    return t * _COMPONENTS - t * (t - 1) // 2 + (u - t)


def _variable(site, t, u):
    return site * _VARIABLES_PER_SITE + _upper_index(t, u)


def _derive(molecule, site_axes=None):
    axes = [] if site_axes is None else [_matrix(frame) for frame in site_axes]
    return psi4.core._atomic_polarizability_derive_pdef_constraints(molecule, axes)


def _bond_graph(molecule, scale=None):
    if scale is None:
        return psi4.core._atomic_polarizability_derive_bond_graph(molecule)
    return psi4.core._atomic_polarizability_derive_bond_graph(molecule, scale)


def _numeric_characters(signs):
    """Character of every L3 component under diag(signs), from the exported harmonics."""
    probe = np.array([0.31, -0.53, 0.79])
    reference = np.asarray(
        psi4.core._atomic_polarizability_test_irregular_harmonics(list(probe), [0.0, 0.0, 0.0])
    )[1:]
    imaged = np.asarray(
        psi4.core._atomic_polarizability_test_irregular_harmonics(
            list(probe * np.asarray(signs, dtype=float)), [0.0, 0.0, 0.0]
        )
    )[1:]
    assert np.all(np.abs(reference) > 1.0e-6)
    ratios = imaged / reference
    characters = np.rint(ratios).astype(int)
    assert np.allclose(ratios, characters, atol=1.0e-12)
    assert set(np.unique(characters)) <= {-1, 1}
    return characters


def _expected_active_pairs(operation_signs):
    """Pairs sharing a character tuple over the site-group generators."""
    tuples = list(zip(*(_numeric_characters(signs) for signs in operation_signs)))
    return {
        (t, u)
        for t in range(_COMPONENTS)
        for u in range(t, _COMPONENTS)
        if tuples[t] == tuples[u]
    }


def _water(angstrom=True, symmetry=None, reorient=True):
    lines = [
        "O 0.000000 0.000000 0.117300",
        "H 0.000000 0.757200 -0.469200",
        "H 0.000000 -0.757200 -0.469200",
    ]
    if symmetry is not None:
        lines.append(f"symmetry {symmetry}")
    if not reorient:
        lines.extend(["no_reorient", "no_com"])
    lines.append("units angstrom" if angstrom else "units bohr")
    return psi4.geometry("\n".join(lines))


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _explicit_molecule(rows, *, fixed=True):
    lines = [f"{label} {x:.12f} {y:.12f} {z:.12f}" for label, (x, y, z) in rows]
    if fixed:
        lines.extend(["no_reorient", "no_com"])
    lines.append("units bohr")
    return psi4.geometry("\n".join(lines))


# --------------------------------------------------------------------------------------
# PDef active-variable constraint derivation
# --------------------------------------------------------------------------------------


def test_derived_c2v_site_has_thirty_eight_active_pairs_with_exact_membership():
    derived = _derive(_water())
    oxygen = derived["sites"][0]

    assert oxygen["point_group"].lower().startswith("c2v")
    assert len(oxygen["operation_signs"]) == 4
    pairs = {tuple(pair) for pair in oxygen["active_pairs"]}
    assert len(pairs) == 38
    assert pairs == _expected_active_pairs(oxygen["operation_signs"])

    classes = np.asarray(oxygen["component_class"])
    assert oxygen["class_count"] == 4
    sizes = sorted(int(np.count_nonzero(classes == label)) for label in range(4))
    assert sizes == [2, 4, 4, 5]
    assert sum(size * (size + 1) // 2 for size in sizes) == 38


def test_derived_cs_site_has_sixty_six_active_pairs_with_exact_membership():
    derived = _derive(_water())
    hydrogen = derived["sites"][1]

    assert hydrogen["point_group"].lower().startswith("cs")
    assert len(hydrogen["operation_signs"]) == 2
    pairs = {tuple(pair) for pair in hydrogen["active_pairs"]}
    assert len(pairs) == 66
    assert pairs == _expected_active_pairs(hydrogen["operation_signs"])

    classes = np.asarray(hydrogen["component_class"])
    assert hydrogen["class_count"] == 2
    sizes = sorted(int(np.count_nonzero(classes == label)) for label in range(2))
    assert sizes == [6, 9]
    assert sum(size * (size + 1) // 2 for size in sizes) == 66


def test_derived_mask_marks_exactly_the_active_pairs_of_every_site():
    derived = _derive(_water())
    mask = np.asarray(derived["active_variables"], dtype=bool)

    assert mask.size == 3 * _VARIABLES_PER_SITE == derived["variable_count"]
    expected = np.zeros_like(mask)
    for site, record in enumerate(derived["sites"]):
        for t, u in record["active_pairs"]:
            expected[_variable(site, t, u)] = True
    assert np.array_equal(mask, expected)
    assert int(mask.sum()) == derived["active_variable_count"] == 38 + 66 + 66


def test_symmetry_copy_is_an_equality_constraint_that_removes_the_dependent_variables():
    derived = _derive(_water())
    equality = np.asarray(derived["equality"].to_array())
    targets = np.asarray(derived["equality_targets"], dtype=float)

    assert [record["symmetry_source"] for record in derived["sites"]] == [0, 1, 1]
    assert derived["equality_row_count"] == 66 == equality.shape[0]
    assert equality.shape[1] == derived["variable_count"]
    assert np.all(targets == 0.0)
    assert derived["independent_variable_count"] == 38 + 66

    copy_signs = derived["sites"][2]["copy_signs"]
    characters = _numeric_characters(copy_signs)
    source_pairs = {tuple(pair) for pair in derived["sites"][1]["active_pairs"]}
    seen = set()
    for row in equality:
        columns = np.flatnonzero(row)
        assert len(columns) == 2
        dependent, source = sorted(columns, reverse=True)
        assert dependent // _VARIABLES_PER_SITE == 2
        assert source // _VARIABLES_PER_SITE == 1
        assert row[dependent] == 1.0
        pair = next(
            (t, u)
            for t in range(_COMPONENTS)
            for u in range(t, _COMPONENTS)
            if _upper_index(t, u) == source % _VARIABLES_PER_SITE
        )
        assert pair in source_pairs
        assert row[source] == -float(characters[pair[0]] * characters[pair[1]])
        seen.add(pair)
    assert seen == source_pairs


def test_frozen_components_are_absent_from_the_design_matrix_not_zeroed_afterwards():
    derived = _derive(_water())
    mask = list(np.asarray(derived["active_variables"], dtype=bool))
    frozen = [index for index, active in enumerate(mask) if not active]
    assert frozen

    positions = np.asarray(_water().geometry().to_array())
    generator = np.random.default_rng(20260817)
    directions = generator.normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = positions.mean(axis=0) + directions * np.linspace(2.6, 4.2, 30)[:, None]
    tensors = [0.05 * np.eye(_COMPONENTS) for _ in positions]
    harmonics = np.array(
        [
            [
                np.asarray(
                    psi4.core._atomic_polarizability_test_irregular_harmonics(
                        list(point), list(site)
                    )
                )[1:]
                for site in positions
            ]
            for point in points
        ]
    )
    response = np.einsum("gat,atu,hau->gh", harmonics, np.asarray(tensors), harmonics)

    def _fit(active):
        return psi4.core._atomic_polarizability_test_refine_wsm(
            _matrix(points), [0.0], [_matrix(response)], _matrix(positions),
            [_matrix(np.zeros((_COMPONENTS, _COMPONENTS))) for _ in positions], [0.0],
            list(active), derived["equality"], list(derived["equality_targets"]), {},
        )[0]

    constrained = _fit(mask)
    assert constrained["active_variable_count"] == 38 + 66 + 66
    touched = set(constrained["kept_variables"]) | set(constrained["pruned_variables"])
    assert touched.isdisjoint(frozen)
    solution = np.asarray(constrained["solution"], dtype=float)
    assert np.all(solution[frozen] == 0.0)

    # Widening the mask by exactly one frozen dipole variable on the independent hydrogen
    # gives that variable a design-matrix column, and the same data then fits it nonzero:
    # the zeros above are structural, not a post-hoc overwrite of a fitted value.
    hydrogen = derived["sites"][1]
    off_diagonal = next(
        (t, u)
        for t in range(3)
        for u in range(t + 1, 3)
        if [t, u] not in [list(pair) for pair in hydrogen["active_pairs"]]
    )
    widened = list(mask)
    released = _variable(1, *off_diagonal)
    widened[released] = True
    relaxed = _fit(widened)
    assert relaxed["active_variable_count"] == 38 + 66 + 66 + 1
    assert released in set(relaxed["kept_variables"])
    assert abs(np.asarray(relaxed["solution"], dtype=float)[released]) > 1.0e-8


def test_derived_constraints_are_invariant_to_molecule_reorientation():
    first = _derive(_water())
    rotated = np.asarray(_water().geometry().to_array()) @ _rotation([0.31, -0.72, 0.62], 0.83).T
    labels = ("O", "H", "H")
    second = _derive(_explicit_molecule(list(zip(labels, rotated)), fixed=False))

    assert first["active_variables"] == second["active_variables"]
    assert first["equality_targets"] == second["equality_targets"]
    assert np.array_equal(
        np.asarray(first["equality"].to_array()), np.asarray(second["equality"].to_array())
    )
    assert [record["point_group"] for record in first["sites"]] == [
        record["point_group"] for record in second["sites"]
    ]


def test_derivation_rejects_site_axes_inconsistent_with_the_detected_site_symmetry():
    molecule = _water()
    tilted = _rotation([0.0, 0.0, 1.0], np.pi / 4.0)
    axes = [np.eye(3), np.eye(3), np.eye(3)]
    axes[0] = tilted

    with pytest.raises(RuntimeError, match=r"local axes.*site-group operation"):
        _derive(molecule, axes)


def test_derivation_rejects_non_orthonormal_and_improper_site_axes():
    molecule = _water()
    skewed = [np.eye(3), np.eye(3), np.eye(3)]
    skewed[1] = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(RuntimeError, match=r"orthonormal"):
        _derive(molecule, skewed)

    improper = [np.eye(3), np.eye(3), np.eye(3)]
    improper[2] = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(RuntimeError, match=r"right-handed"):
        _derive(molecule, improper)

    with pytest.raises(RuntimeError, match=r"one local axis frame per site"):
        _derive(molecule, [np.eye(3)])


def test_derivation_fails_closed_when_the_molecular_frame_hides_the_detected_symmetry():
    rotated = np.asarray(_water().geometry().to_array()) @ _rotation([0.0, 1.0, 0.0], 0.4).T
    molecule = _explicit_molecule(list(zip(("O", "H", "H"), rotated)))

    with pytest.raises(RuntimeError, match=r"molecular frame does not realize"):
        _derive(molecule)


def test_derived_d2h_site_keeps_twenty_five_active_pairs():
    molecule = psi4.geometry(
        """
        O 0.0 0.0  1.16
        C 0.0 0.0  0.00
        O 0.0 0.0 -1.16
        units angstrom
        """
    )
    derived = _derive(molecule)
    labels = [record["point_group"].lower() for record in derived["sites"]]
    carbon = labels.index("d2h")

    record = derived["sites"][carbon]
    pairs = {tuple(pair) for pair in record["active_pairs"]}
    assert len(pairs) == 25
    assert pairs == _expected_active_pairs(record["operation_signs"])
    assert record["class_count"] == 8

    axial = [index for index in range(3) if index != carbon]
    for index in axial:
        assert len(derived["sites"][index]["active_pairs"]) == 38
    assert derived["independent_variable_count"] == 25 + 38


def test_derived_homonuclear_diatomic_has_a_single_independent_site():
    molecule = psi4.geometry(
        """
        H 0.0 0.0  0.3705
        H 0.0 0.0 -0.3705
        units angstrom
        """
    )
    derived = _derive(molecule)

    assert [len(record["active_pairs"]) for record in derived["sites"]] == [38, 38]
    assert [record["symmetry_source"] for record in derived["sites"]] == [0, 0]
    assert derived["equality_row_count"] == 38
    assert derived["independent_variable_count"] == 38


def test_derived_c1_sites_keep_every_upper_triangle_variable():
    molecule = psi4.geometry(
        """
        N  0.10  0.20  0.30
        O  1.30 -0.40  0.90
        F -0.70  1.10 -0.50
        Cl 0.40 -1.20 -1.30
        units angstrom
        """
    )
    derived = _derive(molecule)

    # With only the identity every component carries the trivial irrep, so nothing freezes.
    assert all(record["point_group"].lower() == "c1" for record in derived["sites"])
    assert all(record["class_count"] == 1 for record in derived["sites"])
    assert all(
        [tuple(signs) for signs in record["operation_signs"]] == [(1, 1, 1)]
        for record in derived["sites"]
    )
    assert all(len(record["active_pairs"]) == _VARIABLES_PER_SITE for record in derived["sites"])
    assert all(derived["active_variables"])
    assert derived["equality_row_count"] == 0
    assert derived["independent_variable_count"] == 4 * _VARIABLES_PER_SITE


def test_active_pair_counts_do_not_depend_on_which_cartesian_axis_carries_the_c2_axis():
    along_z = _derive(_water())
    positions = np.asarray(_water().geometry().to_array())
    swapped = positions[:, [2, 0, 1]]
    along_x = _derive(_explicit_molecule(list(zip(("O", "H", "H"), swapped))))

    assert [record["point_group"] for record in along_z["sites"]] != [
        record["point_group"] for record in along_x["sites"]
    ]
    assert [len(record["active_pairs"]) for record in along_x["sites"]] == [38, 66, 66]
    assert along_x["independent_variable_count"] == 38 + 66
    for record in along_x["sites"]:
        assert {tuple(pair) for pair in record["active_pairs"]} == _expected_active_pairs(
            record["operation_signs"]
        )
    assert along_z["active_variables"] != along_x["active_variables"]


def test_derived_constraints_are_accepted_by_the_wsm_refinement_gates():
    derived = _derive(_water())
    assert derived["geometry_tolerance"] > 0.0
    plan = psi4.core._atomic_polarizability_plan_wsm_refinement(
        60, 3, derived["active_variable_count"], derived["equality_row_count"],
        psi4.core.get_memory(),
    )
    assert plan["active_variable_count"] == derived["active_variable_count"]
    assert plan["constraint_rows"] == derived["equality_row_count"]


# --------------------------------------------------------------------------------------
# Covalent bond-graph derivation
# --------------------------------------------------------------------------------------


def test_derived_water_bond_graph_connects_oxygen_to_both_hydrogens_only():
    graph = _bond_graph(_water())

    assert graph["site_count"] == 3
    assert [tuple(bond) for bond in graph["bonds"]] == [(0, 1), (0, 2)]
    assert graph["component_count"] == 1
    assert graph["radius_table"] == "Slater-1964-bohr-v1"
    assert graph["covalent_scale"] == pytest.approx(1.3)
    assert len(graph["bond_distances"]) == len(graph["bonds"]) == len(graph["bond_thresholds"])
    assert all(
        distance <= threshold
        for distance, threshold in zip(graph["bond_distances"], graph["bond_thresholds"])
    )


def test_derived_bond_graph_of_carbon_dioxide_and_methane_matches_chemical_connectivity():
    carbon_dioxide = _bond_graph(
        psi4.geometry("O 0 0 1.16\nC 0 0 0\nO 0 0 -1.16\nunits angstrom")
    )
    assert [tuple(bond) for bond in carbon_dioxide["bonds"]] == [(0, 1), (1, 2)]

    methane = _bond_graph(
        psi4.geometry(
            """
            C  0.000000  0.000000  0.000000
            H  0.629118  0.629118  0.629118
            H -0.629118 -0.629118  0.629118
            H -0.629118  0.629118 -0.629118
            H  0.629118 -0.629118 -0.629118
            units angstrom
            """
        )
    )
    assert [tuple(bond) for bond in methane["bonds"]] == [(0, 1), (0, 2), (0, 3), (0, 4)]
    assert methane["component_count"] == 1


def test_derived_bond_graph_is_deterministic_sorted_and_frame_independent():
    molecule = _water()
    repeated = [_bond_graph(molecule)["bonds"] for _ in range(3)]
    assert repeated[0] == repeated[1] == repeated[2]
    bonds = [tuple(bond) for bond in repeated[0]]
    assert all(first < second for first, second in bonds)
    assert bonds == sorted(bonds)

    rotated = np.asarray(molecule.geometry().to_array()) @ _rotation([0.4, 0.5, -0.77], 1.21).T
    displaced = rotated + np.array([3.1, -2.2, 0.9])
    moved = _bond_graph(_explicit_molecule(list(zip(("O", "H", "H"), displaced))))
    assert [tuple(bond) for bond in moved["bonds"]] == bonds
    assert moved["bond_distances"] == pytest.approx(
        _bond_graph(molecule)["bond_distances"], abs=1.0e-12
    )


def test_derived_bond_graph_fails_closed_on_a_disconnected_geometry():
    dimer = psi4.geometry(
        """
        O 0.000000 0.000000 0.117300
        H 0.000000 0.757200 -0.469200
        H 0.000000 -0.757200 -0.469200
        --
        O 12.000000 0.000000 0.117300
        H 12.000000 0.757200 -0.469200
        H 12.000000 -0.757200 -0.469200
        units angstrom
        """
    )
    with pytest.raises(RuntimeError, match=r"disconnected.*2 components"):
        _bond_graph(dimer)


def test_derived_bond_graph_accepts_a_single_site_and_rejects_invalid_scales():
    lone = _bond_graph(psi4.geometry("He 0 0 0\nunits angstrom"))
    assert lone["site_count"] == 1
    assert lone["bonds"] == []
    assert lone["component_count"] == 1

    molecule = _water()
    with pytest.raises(RuntimeError, match=r"scale must be finite and positive"):
        _bond_graph(molecule, 0.0)
    with pytest.raises(RuntimeError, match=r"scale must be finite and positive"):
        _bond_graph(molecule, float("nan"))
    with pytest.raises(RuntimeError, match=r"disconnected"):
        _bond_graph(molecule, 0.5)


def test_bond_scale_separates_covalent_bonds_from_the_tightest_nonbonded_contact():
    molecule = _water()
    generous = _bond_graph(molecule, 2.5)
    assert [tuple(bond) for bond in generous["bonds"]] == [(0, 1), (0, 2), (1, 2)]

    default = _bond_graph(molecule)
    distances = np.asarray(molecule.distance_matrix().to_array())
    radii = np.asarray(default["radii"], dtype=float)
    bonded = max(distances[i, j] / (radii[i] + radii[j]) for i, j in [(0, 1), (0, 2)])
    nonbonded = distances[1, 2] / (radii[1] + radii[2])
    assert bonded < default["covalent_scale"] < nonbonded


def test_derived_bond_graph_yields_a_connected_lw_graph_laplacian():
    graph = _bond_graph(_water())
    operator, pseudoinverse, eigenvalues = psi4.core._atomic_polarizability_lw_graph_math(
        graph["site_count"], [list(bond) for bond in graph["bonds"]]
    )
    values = np.asarray(eigenvalues, dtype=float)

    assert np.count_nonzero(np.abs(values) < 1.0e-10) == 1
    laplacian = np.asarray(operator.to_array())
    assert laplacian == pytest.approx(laplacian.T, abs=1.0e-14)
    assert laplacian.sum(axis=1) == pytest.approx(np.zeros(graph["site_count"]), abs=1.0e-13)
    del pseudoinverse


def test_pure_site_seam_reproduces_the_molecular_bond_graph():
    molecule = _water()
    positions = np.asarray(molecule.geometry().to_array())
    numbers = [int(molecule.true_atomic_number(index)) for index in range(molecule.natom())]
    seam = psi4.core._atomic_polarizability_derive_bond_graph_from_sites(
        _matrix(positions), numbers, 1.3
    )

    assert seam["bonds"] == _bond_graph(molecule)["bonds"]
    assert seam["radii"] == pytest.approx(_bond_graph(molecule)["radii"], abs=0.0)
    with pytest.raises(RuntimeError, match=r"radius table"):
        psi4.core._atomic_polarizability_derive_bond_graph_from_sites(
            _matrix(positions), [1, 1, 0], 1.3
        )


def test_symmetry_and_bond_derivations_stay_inside_the_native_source_guard():
    from test_native_atomic_polarizability_source_guard import source_violations

    repo_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "psi4/src/psi4/libmints/atomic_polarizability.cc").is_file()
    )
    sources = (
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.cc",
        repo_root / "psi4/src/psi4/libmints/atomic_polarizability.h",
        repo_root / "psi4/src/psi4/libmints/isa_weights.cc",
        repo_root / "psi4/src/export_oeprop.cc",
    )
    violations = []
    for source in sources:
        violations.extend(f"{source.name}: {item}" for item in source_violations(source.read_text()))
    assert violations == []

    text = (repo_root / "psi4/src/psi4/libmints/atomic_polarizability.cc").read_text()
    assert "slater_radius" in text, "the bond graph must reuse the existing libmints radius table"
    assert "derive_pdef_constraints" in text and "derive_bond_graph" in text
    # Nothing dispatches on an element name or a fixed atom index, so no molecule is special.
    for literal in ("label(", "flabel(", "symbol()", "Element_to_Z"):
        assert literal not in text
