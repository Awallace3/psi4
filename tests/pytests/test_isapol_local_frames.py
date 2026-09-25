# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Tinker-style atom-defined local frames; analytic geometry, no archived fixtures."""
import numpy as np
import pytest

# Site 0 at the origin with three references; every expected axis is hand-derived.
ORIGINS = np.array([[0., 0., 0.], [0., 0., 2.], [3., 0., 0.], [0., 5., 0.]])
S2, S3 = np.sqrt(2.), np.sqrt(3.)
HAND = {
    "z_then_x": (dict(z=1, x=2), [1., 0., 0.], [0., 0., 1.]),
    "bisector": (dict(z=1, x=2), [1/S2, 0., -1/S2], [1/S2, 0., 1/S2]),
    "z_bisect": (dict(z=1, x=2, y=3), [1/S2, 1/S2, 0.], [0., 0., 1.]),
    "three_fold": (dict(z=1, x=2, y=3),
                   np.array([-1., -1., 2.])/np.sqrt(6.), np.array([1., 1., 1.])/S3),
    "z_only": (dict(z=1), [1., 0., 0.], [0., 0., 1.]),
}


def _declarations(kind, refs, n=4):
    from psi4.driver.procrouting.isapol_geometry import LocalFrame
    return [LocalFrame(0, kind, **refs)] + [LocalFrame(i, "global") for i in range(1, n)]


def _proper(frame):
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14, rtol=0)
    assert np.linalg.det(frame) == pytest.approx(1., abs=1e-14)


@pytest.mark.parametrize("kind", sorted(HAND))
def test_recipes_match_hand_derived_axes(kind):
    from psi4.driver.procrouting.isapol_geometry import resolve_local_frames
    refs, x, z = HAND[kind]
    resolved = resolve_local_frames(ORIGINS, _declarations(kind, refs))
    frame = resolved.frames[0]
    _proper(frame)
    np.testing.assert_allclose(frame[:, 0], x, atol=1e-15)
    np.testing.assert_allclose(frame[:, 2], z, atol=1e-15)
    np.testing.assert_allclose(frame[:, 1], np.cross(z, x), atol=1e-15)
    np.testing.assert_array_equal(resolved.frames[1:], np.tile(np.eye(3), (3, 1, 1)))
    np.testing.assert_array_equal(resolved.handedness, 0)


@pytest.mark.parametrize("kind", sorted(set(HAND) - {"z_only"}))
def test_recipes_are_rigidly_covariant(kind):
    from psi4.driver.procrouting.isapol_geometry import resolve_local_frames
    rng = np.random.default_rng(7)
    origins = rng.normal(size=(4, 3))
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = q*np.sign(np.diag(r))
    rotation *= np.linalg.det(rotation)
    refs = HAND[kind][0]
    frames = resolve_local_frames(origins, _declarations(kind, refs)).frames
    moved = resolve_local_frames(origins @ rotation.T + [4., -7., 9.],
                                 _declarations(kind, refs)).frames
    _proper(frames[0])
    np.testing.assert_allclose(moved[0], rotation @ frames[0], atol=1e-13)


def test_z_only_is_lab_dependent_and_switches_seed():
    from psi4.driver.procrouting.isapol_geometry import Z_ONLY_SWITCH, resolve_local_frames
    below = np.cos(np.arccos(Z_ONLY_SWITCH)+1e-3)
    above = np.cos(np.arccos(Z_ONLY_SWITCH)-1e-3)
    frames = []
    for c in (below, above):
        origins = [[0., 0., 0.], [c, np.sqrt(1-c*c), 0.]]
        frames.append(resolve_local_frames(origins, _declarations("z_only", dict(z=1), 2)).frames[0])
        _proper(frames[-1])
    # z = (c, s, 0). Global X seeds below the switch, x = (s, -c, 0) and y = -Z;
    # global Y seeds above it, x = (-s, c, 0) and y = +Z. The frame flips at the switch.
    for frame, c, sign in zip(frames, (below, above), (1., -1.)):
        s = np.sqrt(1-c*c)
        np.testing.assert_allclose(frame[:, 0], sign*np.array([s, -c, 0.]), atol=1e-15)
        np.testing.assert_allclose(frame[:, 1], [0., 0., -sign], atol=1e-15)


def test_symmetric_reference_swaps():
    from psi4.driver.procrouting.isapol_geometry import LocalFrame, resolve_local_frames
    rng = np.random.default_rng(3)
    origins = rng.normal(size=(4, 3))

    def frame(**refs):
        kind = refs.pop("kind")
        return resolve_local_frames(origins[:4], [LocalFrame(0, kind, **refs)] +
                                    [LocalFrame(i, "none") for i in (1, 2, 3)]).frames[0]

    # z_bisect's x seed and three_fold's z are symmetric sums; three_fold seeds x from z.
    np.testing.assert_allclose(frame(kind="z_bisect", z=1, x=2, y=3),
                               frame(kind="z_bisect", z=1, x=3, y=2), atol=1e-14)
    np.testing.assert_allclose(frame(kind="three_fold", z=1, x=2, y=3),
                               frame(kind="three_fold", z=1, x=3, y=2), atol=1e-14)
    # bisector's z is symmetric, its x seed is not.
    a, b = frame(kind="bisector", z=1, x=2), frame(kind="bisector", z=2, x=1)
    np.testing.assert_allclose(a[:, 2], b[:, 2], atol=1e-14)
    assert not np.allclose(a[:, 0], b[:, 0])


def test_z_then_x_is_the_axis_pair_frame():
    from psi4.driver.procrouting.isapol_geometry import (LocalFrame, frames_from_axis_pairs,
                                                         resolve_local_frames)
    rng = np.random.default_rng(11)
    origins = rng.normal(size=(3, 3))
    pairs = [(0, (0, 1), (0, 2)), (1, (1, 2), (1, 0)), (2, (0, 1), (2, 0))]
    declared = [LocalFrame(0, "z_then_x", z=1, x=2), LocalFrame(1, "z_then_x", z=2, x=0),
                LocalFrame(2, "axis_pair", z=(0, 1), x=(2, 0))]
    np.testing.assert_allclose(resolve_local_frames(origins, declared).frames,
                               frames_from_axis_pairs(origins, pairs), atol=1e-15)


def test_chirality_reference_records_handedness_not_a_reflection():
    from psi4.driver.procrouting.isapol_geometry import LocalFrame, resolve_local_frames
    origins = np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [.3, .8, -.2]])
    declared = [LocalFrame(0, "z_then_x", z=1, x=2, y=3)] + \
               [LocalFrame(i, "global") for i in (1, 2, 3)]
    right = resolve_local_frames(origins, declared)
    left = resolve_local_frames(origins*[1., -1., 1.], declared)
    assert right.handedness[0] == 1 and left.handedness[0] == -1
    for resolved in (right, left):
        _proper(resolved.frames[0])
    plain = resolve_local_frames(origins, [LocalFrame(0, "z_then_x", z=1, x=2)] + declared[1:])
    np.testing.assert_array_equal(plain.frames, right.frames)
    assert plain.handedness[0] == 0


def _racah_rank2(theta):
    """Stone's real-Racah components 20,21c,21s,22c,22s of a traceless quadrupole."""
    return np.array([theta[2, 2], 2*theta[0, 2]/np.sqrt(3.), 2*theta[1, 2]/np.sqrt(3.),
                     (theta[0, 0]-theta[1, 1])/np.sqrt(3.), 2*theta[0, 1]/np.sqrt(3.)])


def test_recipe_frames_rotate_rank1_and_rank2_like_cartesian_tensors():
    from psi4 import core
    from psi4.driver.procrouting.isapol_geometry import resolve_local_frames
    rng = np.random.default_rng(5)
    origins = rng.normal(size=(4, 3))
    frame = resolve_local_frames(origins, _declarations("z_bisect", HAND["z_bisect"][0])).frames[0]
    d = np.asarray(core.isa_multipole_rotation(2, frame.tolist()))
    # Rank 1: a Cartesian polarizability F^T alpha F in dipole order z, x, y.
    a = rng.normal(size=(3, 3))
    alpha = a @ a.T
    order = [2, 0, 1]
    local = frame.T @ alpha @ frame
    racah = d[1:4, 1:4].T @ alpha[np.ix_(order, order)] @ d[1:4, 1:4]
    np.testing.assert_allclose(racah, local[np.ix_(order, order)], atol=1e-13)
    # Rank 2: a traceless quadrupole's Racah components transform as D^T q.
    b = rng.normal(size=(3, 3))
    theta = b + b.T
    theta -= np.trace(theta)/3*np.eye(3)
    np.testing.assert_allclose(d[4:9, 4:9].T @ _racah_rank2(theta),
                               _racah_rank2(frame.T @ theta @ frame), atol=1e-13)


def test_local_y_reflection_matches_cartesian_reflection():
    from psi4.driver.procrouting.isapol_geometry import local_y_reflection
    signs = local_y_reflection(2)
    np.testing.assert_array_equal(signs, [1, 1, 1, -1, 1, 1, -1, 1, -1])
    rng = np.random.default_rng(2)
    b = rng.normal(size=(3, 3))
    theta = b + b.T
    theta -= np.trace(theta)/3*np.eye(3)
    mirror = np.diag([1., -1., 1.])
    np.testing.assert_allclose(signs[4:]*_racah_rank2(theta), _racah_rank2(mirror @ theta @ mirror),
                               atol=1e-15)
    assert len(local_y_reflection(4)) == 25


@pytest.mark.parametrize("origins, frame, message", [
    ([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]], ("z_then_x", dict(z=1, x=2)), "degenerate"),
    ([[0., 0., 0.], [0., 0., 1.], [0., 0., 3.]], ("z_then_x", dict(z=1, x=2)), "collinear"),
    ([[0., 0., 0.], [0., 0., 1.], [0., 0., -2.]], ("bisector", dict(z=1, x=2)), "cancel"),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 1e-9]], ("bisector", dict(z=1, x=2)), None),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [-1., 0., 0.]],
     ("z_bisect", dict(z=1, x=2, y=3)), "cancel"),
    ([[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., -1., 1.]],
     ("z_bisect", dict(z=1, x=2, y=3)), "collinear"),
    ([[0., 0., 0.], [1., 0., 0.], [-.5, np.sqrt(.75), 0.], [-.5, -np.sqrt(.75), 0.]],
     ("three_fold", dict(z=1, x=2, y=3)), "cancel"),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [2., 0., 3.]],
     ("z_then_x", dict(z=1, x=2, y=3)), "coplanar"),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]], ("z_then_x", dict(z=1, x=1)), "distinct"),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]], ("z_then_x", dict(z=0, x=2)), "distinct"),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]], ("z_then_x", dict(z=1, x=5)), "outside"),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]], ("z_then_x", dict(z=True, x=2)), "integers"),
    ([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]], ("matrix", dict(frame=np.diag([1., 1., -1.]))),
     "proper"),
])
def test_degenerate_or_invalid_frames_fail_closed(origins, frame, message):
    from psi4.driver.procrouting.isapol_geometry import LocalFrame, resolve_local_frames
    declared = [LocalFrame(0, frame[0], **frame[1])] + \
               [LocalFrame(i, "global") for i in range(1, len(origins))]
    if message is None:
        _proper(resolve_local_frames(origins, declared).frames[0])
        return
    with pytest.raises(ValueError, match=message):
        resolve_local_frames(origins, declared)


@pytest.mark.parametrize("kind, refs, message", [
    ("z_then_x", dict(z=1), "requires x"),
    ("z_bisect", dict(z=1, x=2), "requires y"),
    ("bisector", dict(z=1, x=2, y=3), "does not take y"),
    ("global", dict(z=1), "does not take z"),
    ("matrix", dict(), "requires frame"),
    ("Z-then-X", dict(z=1, x=2), "unsupported"),
])
def test_declarations_are_typed_per_kind(kind, refs, message):
    from psi4.driver.procrouting.isapol_geometry import LocalFrame
    with pytest.raises(ValueError, match=message):
        LocalFrame(0, kind, **refs)


def test_every_site_declared_exactly_once():
    from psi4.driver.procrouting.isapol_geometry import LocalFrame, resolve_local_frames
    origins = [[0., 0., 0.], [0., 0., 1.]]
    with pytest.raises(ValueError, match="every site"):
        resolve_local_frames(origins, [LocalFrame(0, "global")])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_local_frames(origins, [LocalFrame(0, "global")]*2 + [LocalFrame(1, "global")])
    with pytest.raises(ValueError, match="LocalFrame"):
        resolve_local_frames(origins, [(0, (0, 1), (0, 1)), LocalFrame(1, "global")])


def test_supplied_local_frames_match_resolved_matrix_frames():
    from psi4.driver.procrouting.isapol_geometry import LocalFrame, resolve_local_frames
    from psi4.driver.procrouting.isapol_lw import Provenance, supplied_nonlocal_properties

    theta = np.arange(6)*np.pi/3
    radial = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(6)))
    origins = np.concatenate((2*radial, 3*radial))
    declared = ([LocalFrame(i, "z_then_x", z=i+6, x=(i+1) % 6) for i in range(6)] +
                [LocalFrame(i+6, "axis_pair", z=(i, i+6), x=(i, (i+1) % 6)) for i in range(6)])
    blocks = np.zeros((1, 12, 12, 16, 16))
    rng = np.random.default_rng(13)
    for site in range(12):
        a = rng.normal(size=(3, 3))
        blocks[0, site, site, 1:4, 1:4] = a @ a.T + np.eye(3)
    kwargs = dict(labels=tuple(f"S{i}" for i in range(12)), origins=origins,
                  bonds=tuple((i, (i+1) % 6) for i in range(6)) +
                        tuple((i, i+6) for i in range(6)),
                  frequencies=(0.,), tensors=blocks, input_rank=3,
                  localization_rank_limit=1,
                  provenance=Provenance("analytic ring", "0"*64, "synthetic",
                                        "exact charge-free local dipoles"))
    by_recipe = supplied_nonlocal_properties(local_frames=declared, **kwargs)
    by_matrix = supplied_nonlocal_properties(
        frames=resolve_local_frames(origins, declared).frames, **kwargs)
    assert by_recipe.frames.array.tobytes() == by_matrix.frames.array.tobytes()
    assert by_recipe.raw_local.array.tobytes() == by_matrix.raw_local.array.tobytes()
    with pytest.raises(ValueError, match="not both"):
        supplied_nonlocal_properties(local_frames=declared,
                                     frames=by_matrix.frames.array, **kwargs)
