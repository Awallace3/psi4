# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Explicit local-frame construction, independent of partition/response policy."""
import operator

import numpy as np


def _frame_from_directions(z, x):
    """Shared numerical kernel for indexed axes and the narrow file grammar."""
    def unit(vector):
        vector = np.asarray(vector, dtype=float)
        scale = np.max(np.abs(vector))
        if vector.shape != (3,) or not np.isfinite(scale) or scale == 0:
            raise ValueError("degenerate or nonfinite axis direction")
        vector = vector/scale
        return vector/np.linalg.norm(vector)

    z, x = unit(z), unit(x)
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

    def index(value):
        try:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError
            result = operator.index(value)
        except TypeError:
            raise ValueError("axis indices must be integers") from None
        if not 0 <= result < count:
            raise ValueError("axis index outside site range")
        return result

    def direction(pair):
        try:
            start, end = pair
        except (TypeError, ValueError):
            raise ValueError("axis direction requires two site indices") from None
        with np.errstate(over="ignore", invalid="ignore"):
            vector = origins[index(end)] - origins[index(start)]
        return vector

    for declaration in declarations:
        try:
            site, z_pair, x_pair = declaration
        except (TypeError, ValueError):
            raise ValueError("expected (site, z_pair, x_pair) declaration") from None
        site = index(site)
        if site in seen:
            raise ValueError("duplicate axis declaration")
        frames[site] = _frame_from_directions(direction(z_pair), direction(x_pair))
        seen.add(site)
    if len(seen) != count:
        raise ValueError("explicit axis declaration required for every site")
    return frames
