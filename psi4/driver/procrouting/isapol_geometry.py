# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit local-frame construction, independent of partition/response policy."""
from dataclasses import dataclass
import operator
from typing import Optional

import numpy as np


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    scale = np.max(np.abs(vector))
    if vector.shape != (3,) or not np.isfinite(scale) or scale == 0:
        raise ValueError("degenerate or nonfinite axis direction")
    vector = vector/scale
    return vector/np.linalg.norm(vector)


def _frame_from_directions(z, x):
    """Shared numerical kernel for indexed axes and the narrow file grammar."""
    z, x = _unit(z), _unit(x)
    # Avoid cancellation in x - z*(z.x) for nearly parallel directions.
    y = np.cross(z, x)
    length = np.linalg.norm(y)
    if length <= 64*np.finfo(float).eps:
        raise ValueError("degenerate collinear axis directions")
    y /= length
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    frame = np.column_stack((x, y, z))
    if (not np.allclose(frame.T @ frame, np.eye(3), atol=1e-12, rtol=0)
            or abs(np.linalg.det(frame)-1.) > 1e-12):
        raise ValueError("axis directions do not define a proper rotation")
    return frame


def _axis_index(value, count):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError:
        raise ValueError("axis indices must be integers") from None
    if not 0 <= result < count:
        raise ValueError("axis index outside site range")
    return result


def _axis_pair_frame(origins, z_pair, x_pair):
    """One frame from ``(from, to)`` z and x site pairs of validated ``origins``."""
    def direction(pair):
        try:
            start, end = pair
        except (TypeError, ValueError):
            raise ValueError("axis direction requires two site indices") from None
        with np.errstate(over="ignore", invalid="ignore"):
            vector = (origins[_axis_index(end, len(origins))]
                      - origins[_axis_index(start, len(origins))])
        return vector

    return _frame_from_directions(direction(z_pair), direction(x_pair))


def frames_from_axis_pairs(origins, declarations):
    """Return local-to-global columns ``[x, y, z]`` in site order.

    ``origins`` is a finite, nonempty (nsite, 3) array. Each declaration is
    ``(site, (z_from, z_to), (x_from, x_to))`` with zero-based integer indices.
    Every site must be declared exactly once; no topology or symmetry is
    inferred. The x direction is projected perpendicular to z and y = z cross x.
    Coincident endpoints and directions with sine of their included angle
    at most 64 times float64 epsilon are refused,
    never replaced with fallback axes. Coordinates may use any common length
    unit. The returned array owns its storage.
    """
    origins = np.asarray(origins, dtype=float)
    if (origins.ndim != 2 or origins.shape[1] != 3 or not len(origins)
            or not np.isfinite(origins).all()):
        raise ValueError("origins must be a finite nonempty (nsite, 3) array")
    count = len(origins)
    frames = np.empty((count, 3, 3))
    seen = set()

    for declaration in declarations:
        try:
            site, z_pair, x_pair = declaration
        except (TypeError, ValueError):
            raise ValueError("expected (site, z_pair, x_pair) declaration") from None
        site = _axis_index(site, count)
        if site in seen:
            raise ValueError("duplicate axis declaration")
        frames[site] = _axis_pair_frame(origins, z_pair, x_pair)
        seen.add(site)
    if len(seen) != count:
        raise ValueError("explicit axis declaration required for every site")
    return frames


# ------------------------------------------------ atom-defined frame recipes ----
#
# Tinker-style local-axis recipes, implemented from the published frame equations
# (Ren & Ponder, J. Phys. Chem. B 107, 5933 (2003); the Tinker user guide's
# multipole-parameter section, and the axis types in TinkerTools/tinker
# source/kmpole.f and source/rotpole.f). Nothing is copied or translated from the
# Tinker sources. Every recipe resolves to a proper local-to-global frame, which
# then goes through the one real-Racah rotation, ``core.isa_multipole_rotation``;
# no second tensor rotation exists here. These recipes are NOT the CamCASP/Orient
# ``.axes`` convention, which :func:`frames_from_axis_pairs` and the supplied
# reader's grammar cover.

FRAME_KINDS = ("global", "none", "matrix", "axis_pair",
               "z_only", "z_then_x", "bisector", "z_bisect", "three_fold")

#: Refusal threshold for atom-defined recipes. Directions are unit vectors before
#: it is applied, so it is scale-free. A frame is refused when the sine of the
#: angle between its z direction and x seed is at most this, or when a bisector or
#: three-fold sum of unit directions has norm at most this. Such frames are not
#: repaired: a near-degenerate recipe's axes swing through large angles under
#: tiny displacements, which is a model error, not round-off.
FRAME_TOLERANCE = 1e-6

#: ``z_only`` seeds x with global X unless ``|z . X|`` exceeds this, then global
#: Y. The frame is therefore lab-orientation dependent and switches discontinuously
#: at the threshold; it is not rotationally covariant. 0.707 is Tinker's switch
#: (checked against Tinker 25.5); OpenMM's ZOnly switches at 0.866, so the two
#: engines disagree for 0.707 < |z . X| <= 0.866.
Z_ONLY_SWITCH = 0.707


@dataclass(frozen=True)
class LocalFrame:
    """One site's explicit frame declaration; every index is zero-based.

    ``kind`` is one of:

    ``global``/``none``
        identity frame.
    ``matrix``
        ``frame`` is a caller-supplied (3, 3) proper local-to-global matrix whose
        columns are local x, y, z in global coordinates.
    ``axis_pair``
        the existing Psi4 definition; ``z`` and ``x`` are ``(from, to)`` site
        pairs, as in :func:`frames_from_axis_pairs`.
    ``z_only``
        z toward ``z``; x from the lab-axis fallback (see :data:`Z_ONLY_SWITCH`).
    ``z_then_x``
        z toward ``z``; x toward ``x`` projected perpendicular to z. An optional
        ``y`` is Tinker's chirality reference: it never changes the frame, only
        the resolved ``handedness`` record. Tinker and OpenMM, given a positive
        y-axis type, negate the local y components exactly where handedness is +1.
    ``bisector``
        z along the sum of the unit directions toward ``z`` and ``x``; x toward
        ``x`` projected perpendicular to z.
    ``z_bisect``
        z toward ``z``; x along the sum of the unit directions toward ``x`` and
        ``y``, projected perpendicular to z.
    ``three_fold``
        z along the sum of the unit directions toward ``z``, ``x`` and ``y``; x
        toward ``z`` projected perpendicular to z. This is Tinker's 3-Fold
        (checked against Tinker 25.5). OpenMM's ThreeFold seeds x from its x
        reference instead, so OpenMM ``ThreeFold(z, x, y)`` is this recipe with
        ``z`` and ``x`` swapped.

    All directions point from ``site`` toward the reference site; y = z cross x.
    References must be distinct from each other and from ``site``. No
    topology, atom type or symmetry is inferred, and no planar fallback exists.
    """
    site: int
    kind: str
    z: Optional[object] = None
    x: Optional[object] = None
    y: Optional[int] = None
    frame: Optional[object] = None

    def __post_init__(self):
        if self.kind not in FRAME_KINDS:
            raise ValueError(f"unsupported frame kind {self.kind!r}")
        required = {"global": (), "none": (), "matrix": ("frame",),
                    "axis_pair": ("z", "x"), "z_only": ("z",),
                    "z_then_x": ("z", "x"), "bisector": ("z", "x"),
                    "z_bisect": ("z", "x", "y"), "three_fold": ("z", "x", "y")}[self.kind]
        allowed = required + (("y",) if self.kind == "z_then_x" else ())
        for name in ("z", "x", "y", "frame"):
            present = getattr(self, name) is not None
            if name in required and not present:
                raise ValueError(f"{self.kind} frame requires {name}")
            if present and name not in allowed:
                raise ValueError(f"{self.kind} frame does not take {name}")


@dataclass(frozen=True)
class ResolvedFrames:
    """Proper local-to-global frames in site order, plus handedness records.

    ``handedness[i]`` is +1 or -1 for a ``z_then_x`` declaration with a chirality
    reference ``y``: the sign of that reference's local y coordinate. It is 0
    for every other site. The frame itself is always proper; a caller matching
    Tinker's chiral parameter convention applies :func:`local_y_reflection` to
    the component space when the handedness disagrees with the parameter's.
    """
    frames: np.ndarray
    handedness: np.ndarray


def _unit_sum(directions, what):
    total = np.sum(directions, axis=0)
    if np.linalg.norm(total) <= FRAME_TOLERANCE:
        raise ValueError(f"degenerate {what}: unit directions cancel")
    return total/np.linalg.norm(total)


def _checked_frame(z, x, what):
    """Frame from unit z and x seed, refused within FRAME_TOLERANCE of collinear."""
    if np.linalg.norm(np.cross(z, x)) <= FRAME_TOLERANCE:
        raise ValueError(f"degenerate {what}: x seed collinear with z")
    return _frame_from_directions(z, x)


def resolve_local_frames(origins, declarations):
    """Resolve one :class:`LocalFrame` per site to :class:`ResolvedFrames`.

    ``origins`` is a finite, nonempty (nsite, 3) array in any common length unit.
    Every site must be declared exactly once. Undefined or numerically ambiguous
    frames (coincident sites, repeated references, collinear z/x, cancelling
    sums; see :data:`FRAME_TOLERANCE`) raise ValueError rather than switching
    recipe. Every kind except ``z_only`` is covariant under rigid rotation and
    invariant under translation.
    """
    origins = np.asarray(origins, dtype=float)
    if (origins.ndim != 2 or origins.shape[1] != 3 or not len(origins)
            or not np.isfinite(origins).all()):
        raise ValueError("origins must be a finite nonempty (nsite, 3) array")
    count = len(origins)
    frames = np.empty((count, 3, 3))
    handedness = np.zeros(count, dtype=int)
    seen = set()
    for declaration in declarations:
        if not isinstance(declaration, LocalFrame):
            raise ValueError("frame declarations must be LocalFrame instances")
        site = _axis_index(declaration.site, count)
        if site in seen:
            raise ValueError("duplicate frame declaration")
        kind = declaration.kind
        if kind in ("global", "none"):
            frame = np.eye(3)
        elif kind == "matrix":
            frame = np.array(declaration.frame, dtype=float)
            if (frame.shape != (3, 3) or not np.isfinite(frame).all()
                    or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-12, rtol=0)
                    or abs(np.linalg.det(frame)-1.) > 1e-12):
                raise ValueError("matrix frame must be a finite proper rotation")
        elif kind == "axis_pair":
            frame = _axis_pair_frame(origins, declaration.z, declaration.x)
        else:
            names = [n for n in ("z", "x", "y") if getattr(declaration, n) is not None]
            refs = [_axis_index(getattr(declaration, n), count) for n in names]
            if site in refs or len(set(refs)) != len(refs):
                raise ValueError("frame references must be distinct and differ from the site")
            d = {n: _unit(origins[r]-origins[site]) for n, r in zip(names, refs)}
            if kind == "z_only":
                seed = np.array([1., 0., 0.])
                if abs(d["z"][0]) > Z_ONLY_SWITCH:
                    seed = np.array([0., 1., 0.])
                frame = _checked_frame(d["z"], seed, kind)
            elif kind == "z_then_x":
                frame = _checked_frame(d["z"], d["x"], kind)
                if "y" in d:
                    local_y = frame[:, 1] @ d["y"]
                    if abs(local_y) <= FRAME_TOLERANCE:
                        raise ValueError("degenerate z_then_x chirality: y reference is coplanar")
                    handedness[site] = 1 if local_y > 0 else -1
            elif kind == "bisector":
                frame = _checked_frame(_unit_sum((d["z"], d["x"]), kind), d["x"], kind)
            elif kind == "z_bisect":
                frame = _checked_frame(d["z"], _unit_sum((d["x"], d["y"]), kind), kind)
            else:
                frame = _checked_frame(_unit_sum((d["z"], d["x"], d["y"]), kind),
                                       d["z"], kind)
        frames[site] = frame
        seen.add(site)
    if len(seen) != count:
        raise ValueError("explicit frame declaration required for every site")
    return ResolvedFrames(frames, handedness)


def local_y_reflection(rank):
    """Real-Racah component signs of the local reflection y -> -y, ranks 0..rank.

    In 00,10,11c,11s,... order, y -> -y sends phi -> -phi, so every ``s``
    component changes sign and every ``c``/m=0 component is unchanged (rank 1:
    y; rank 2: yz, xy). A local tensor T transforms as ``S[:, None]*T*S``.
    This is an improper operation and is never folded into a frame.
    """
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    signs = []
    for l in range(rank+1):
        signs.append(1.)
        for _ in range(1, l+1):
            signs.extend((1., -1.))
    return np.array(signs)
