"""Declared geometry-only CHNO connectivity and frame preparation.

This is an explicit, inspectable preprocessing convention, not bond-order
perception or a molecular calculation. Linear azimuths require a disclosed
laboratory-axis fallback. Neither this policy nor the downstream LW algorithm
establishes rotational covariance of computed molecular properties.
"""
import numpy as np

BOHR_TO_ANGSTROM = .529177210903
RADII_ANGSTROM = dict(H=.31, C=.76, N=.71, O=.66)
BOND_SCALE = 1.20
THRESHOLD_GUARD_ANGSTROM = .02
LINEAR_SINE_THRESHOLD = 1e-5
PROFILE = 's66-chno-isa-a-v1'


def declare_geometry(symbols, geometry_bohr):
    """Return a portable native declaration and separate policy evidence.

    Frames have local basis vectors as columns (local to global). The z axis
    points to the first bonded atom in source order; x is the projection of
    the most noncollinear other atom, with source-order ties. All sites remain
    independent even when two frames or chemical environments coincide.
    """
    xyz = np.asarray(geometry_bohr, dtype=float) * BOHR_TO_ANGSTROM
    count = len(symbols)
    if (not 3 <= count <= 17 or any(s not in RADII_ANGSTROM for s in symbols)
            or xyz.shape != (count, 3) or not np.isfinite(xyz).all()):
        raise ValueError('finite3..17-atom CHNO geometry required')
    bonds, neighbors = [], [[] for _ in symbols]
    for a in range(count):
        for b in range(a + 1, count):
            distance = float(np.linalg.norm(xyz[b] - xyz[a]))
            radius_sum = RADII_ANGSTROM[symbols[a]] + RADII_ANGSTROM[symbols[b]]
            threshold = BOND_SCALE * radius_sum
            if distance < .6 * radius_sum:
                raise ValueError('overlapping atoms in connectivity preparation')
            if abs(distance - threshold) <= THRESHOLD_GUARD_ANGSTROM:
                raise ValueError('ambiguous connectivity near declared bond threshold')
            if distance < threshold:
                bonds.append([a, b])
                neighbors[a].append(b)
                neighbors[b].append(a)
    visited, pending = set(), [0]
    while pending:
        atom = pending.pop()
        if atom not in visited:
            visited.add(atom)
            pending.extend(neighbors[atom])
    if len(visited) != count:
        raise ValueError('declared connectivity must be connected')
    degrees = dict(H=(1, 1), C=(2, 4), N=(2, 3), O=(1, 2))
    for atom, symbol in enumerate(symbols):
        low, high = degrees[symbol]
        if not low <= len(neighbors[atom]) <= high:
            raise ValueError(f'unsupported connectivity degree for site{atom}:{symbol}')

    frames, frame_evidence, fallback_sites = [], [], []
    for atom in range(count):
        neighbor = neighbors[atom][0]
        z = xyz[neighbor] - xyz[atom]
        z /= np.linalg.norm(z)
        candidates = []
        for other in range(count):
            if other == atom:
                continue
            direction = xyz[other] - xyz[atom]
            direction /= np.linalg.norm(direction)
            projection = direction - np.dot(direction, z) * z
            candidates.append((other, float(np.linalg.norm(projection)), projection))
        best_sine = max(c[1] for c in candidates)
        if best_sine < LINEAR_SINE_THRESHOLD:
            # A linear molecule has no geometry-defined azimuth. Do not
            # amplify tiny input roundoff into a purported covariant frame.
            lab_axis = int(np.argmin(np.abs(z)))
            direction = np.eye(3)[lab_axis]
            x = direction - np.dot(direction, z) * z
            other = None
            fallback_sites.append(atom)
        else:
            other, _, x = next(c for c in candidates if best_sine - c[1] <= 1e-10)
            lab_axis = None
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        frame = np.column_stack((x, y, z))
        if (not np.allclose(frame.T @ frame, np.eye(3), atol=1e-12, rtol=0)
                or abs(np.linalg.det(frame) - 1.) > 1e-12):
            raise ValueError('improper prepared frame')
        frames.append(frame.tolist())
        frame_evidence.append(dict(site=atom, z_neighbor=neighbor, x_atom=other,
                                   lab_axis=lab_axis, best_sine=best_sine))
    declaration = dict(schema='native-monomer-declaration/1', profile=PROFILE,
                       frames=frames, bonds=bonds)
    evidence = dict(
        schema='s66-geometry-declaration-policy/1',
        bohr_to_angstrom=BOHR_TO_ANGSTROM, radii_angstrom=RADII_ANGSTROM.copy(),
        bond_scale=BOND_SCALE, threshold_guard_angstrom=THRESHOLD_GUARD_ANGSTROM,
        linear_sine_threshold=LINEAR_SINE_THRESHOLD, frames=frame_evidence,
        lab_axis_fallback_sites=fallback_sites, independent_parameters=36*count,
        qualification='Geometry-only declared connectivity, no bond orders; linear azimuth fallback explicit; LW rotational covariance not established.')
    return declaration, evidence
