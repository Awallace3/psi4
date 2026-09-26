# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Analytic geometry checks; no archived numerical fixtures."""
import numpy as np
import pytest


def test_explicit_ring_bond_frames_are_radial_and_tangential():
    from psi4.driver.procrouting.isapol_geometry import frames_from_axis_pairs

    theta = np.arange(6)*np.pi/3
    radial = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(6)))
    origins = np.concatenate((2*radial, 3*radial))
    declarations = [(site, (i, i+6), (i, (i+1) % 6))
                    for i in range(6) for site in (i, i+6)]
    frames = frames_from_axis_pairs(origins, declarations)
    tangential = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(6)))
    for offset in (0, 6):
        np.testing.assert_allclose(frames[offset:offset+6, :, 2], radial, atol=1e-14)
        np.testing.assert_allclose(frames[offset:offset+6, :, 0], tangential, atol=1e-14)
        np.testing.assert_allclose(frames[offset:offset+6, :, 1],
                                   np.tile([0., 0., 1.], (6, 1)), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(frames), 1., atol=1e-14)


def test_bond_frames_transform_covariantly():
    from psi4.driver.procrouting.isapol_geometry import frames_from_axis_pairs
    origins = np.array([[0., 0., 0.], [0., 0., 2.], [3., 0., 1.]])
    declarations = [(i, (0, 1), (0, 2)) for i in range(3)]
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    frames = frames_from_axis_pairs(origins, declarations)
    transformed = frames_from_axis_pairs(origins @ rotation.T + [4., -7., 9.], declarations)
    np.testing.assert_allclose(transformed, rotation @ frames, atol=1e-14)


def test_oblique_nearly_parallel_axes_produce_proper_rotations():
    from psi4.driver.procrouting.isapol_geometry import frames_from_axis_pairs
    frames = frames_from_axis_pairs(
        [[0., 0., 0.], [1., 1., 0.], [1., 1., 1e-10]],
        [(i, (0, 1), (0, 2)) for i in range(3)])
    for frame in frames:
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12, rtol=0)
        assert np.linalg.det(frame) == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize("declarations, message", [
    ([(0, (0, 1), (1, 0))], "collinear"),
    ([(0, (0, 0), (0, 2))], "degenerate"),
    ([(0, (0, 3), (0, 2))], "outside"),
    ([(True, (0, 1), (0, 2))], "integers"),
    ([(0, (0, 1), (0, 2))]*2, "duplicate"),
    ([(0, (0, 1), (0, 2))], "every site"),
])
def test_invalid_axis_declarations_fail_closed(declarations, message):
    from psi4.driver.procrouting.isapol_geometry import frames_from_axis_pairs
    with pytest.raises(ValueError, match=message):
        frames_from_axis_pairs([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]], declarations)


@pytest.mark.parametrize("directive, error", [
    ("A z from A to B x from A to C", None),
    ("A z from A to B x from A to C\nA z from A to B x from A to C", "duplicate"),
    ("A z from A to B x from A to D", "unsupported"),
    ("A z from A to A x from A to C", "degenerate"),
    ("A z from A to B x from B to A", "collinear"),
    ("A z from A to C x from A to B", "frame mismatch"),
])
def test_supplied_reader_accepts_explicit_bond_z_axes(tmp_path, directive, error):
    import hashlib
    import json
    from psi4 import core
    from psi4.driver.procrouting.isapol_supplied import MODES, read_orient_local_response

    def write(name, text):
        (tmp_path/name).write_text(text)
        return dict(name=name, sha256=hashlib.sha256(text.encode()).hexdigest())

    coordinates = "A 0 0 0 Type C\nB 1 0 0 Type H\nC 0 1 0 Type H\n"
    sources = []
    for role, text in [
        ("sites", "Units BOHR\n"+coordinates),
        ("recipe", "Units BOHR\nMolecule ring at 0 0 0\n"+coordinates+
         "End\nLocalise\nWrite all local ranks\n"
         "Edit ring\n#include {AXES}\nEnd\n"),
        ("axes", "Axes\n"+directive+"\nEnd\n"),
    ]:
        sources.append(dict(write(role+".txt", text), role=role))
    # Explicit x=y(global), y=z(global), z=x(global), derived by inspection.
    frame = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    files = []
    grid = core.CasimirGrid(10, .5)
    for k in range(11):
        text = "".join(
            f"ALPHA ring SITE-NAMES {label} {label} RANK 1 TO 1 INDEX {k} "
            f"FREQSQ {-grid.omega(k)**2:.17e}\n2 0 0\n0 3 0\n0 0 5\n"
            for label in "ABC") + "ENDFILE\n"
        files.append(dict(write(f"node{k}.pol", text), sections={"file": k}))
    manifest = dict(
        schema=1, units="atomic", geometry_units="bohr",
        components="real_Racah_10_11c_11s", frame_convention="local_to_global_columns",
        molecule="ring", track="synthetic parser contract", tensor_origin=MODES[0],
        authority="headers", header_index_base=0, grid={"n": 10, "beta": .5},
        global_frame_sites=["B", "C"],
        sites=[dict(label=label, origin=origin, frame=frame if label == "A" else np.eye(3).tolist(),
                    ranks=[1]) for label, origin in
               zip("ABC", [[0, 0, 0], [1, 0, 0], [0, 1, 0]])],
        provenance_sources=sources, files=files)
    spec = write("manifest.json", json.dumps(manifest))
    if error:
        with pytest.raises(ValueError, match=error):
            read_orient_local_response(tmp_path, tmp_path/"manifest.json",
                                       manifest_sha256=spec["sha256"])
        return
    result = read_orient_local_response(tmp_path, tmp_path/"manifest.json",
                                        manifest_sha256=spec["sha256"])
    np.testing.assert_array_equal(result.sites[0].frame, frame)
