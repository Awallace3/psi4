# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Grid-only bounded reductions and independent GAMINT point oracle.

Import is stdlib-only. NumPy is imported explicitly only for numerical work.
No density, SCF, ISA trajectory, Coulomb metric, fit or response API is used.
"""
import hashlib
import math
from pathlib import Path
import struct

GRID_RECIPE = dict(radial_points=200, spherical_points=590,
                   becke_smoothing=3, radius_scaling=1.0)
# Independently declared literal GAMINT order; never ask the producer for powers.
POWERS = (
    ((0, 0, 0),),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    ((3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1),
     (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)),
    ((4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1),
     (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0),
     (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)))


def double_factorial(n):
    return math.prod(range(n, 0, -2))


def angular_factor(powers):
    return math.sqrt(double_factorial(2*sum(powers)-1) /
                     math.prod(double_factorial(2*p-1) for p in powers))


def oracle(point, row):
    """Independent separable polynomial * contracted Gaussian, exact effective c.

    No native evaluation, producer function, raw-GBS re-normalization or charge
    oracle is called. Python scalar work is bounded to selected point/columns.
    """
    displacement = tuple(float(x)-c for x, c in zip(point, row["centre_bohr"]))
    radius2 = math.fsum(x*x for x in displacement)
    angular = angular_factor(row["powers"])*math.prod(
        x**p for x, p in zip(displacement, row["powers"]))
    radial = math.fsum(c*math.exp(-a*radius2)
                       for a, c in zip(row["exponents"], row["coefficients"]))
    value = angular*radial
    if not math.isfinite(value):
        raise ValueError("nonfinite independent point oracle")
    return value


def columns(snapshot):
    """Lossless shell snapshot -> exact GAMINT column identity; no normalization."""
    centres = tuple(tuple(float.fromhex(x) for x in c)
                    for c in snapshot["centres_bohr_hex"])
    if not centres or any(len(c) != 3 or not all(map(math.isfinite, c)) for c in centres):
        raise ValueError("invalid centres")
    rows = []
    for index, shell in enumerate(snapshot["ordered_shells"]):
        l, centre = shell["l"], shell["centre"]
        if (type(l) is not int or not 0 <= l <= 4 or shell["is_pure"]
                or type(centre) is not int or not 0 <= centre < len(centres)):
            raise ValueError("Cartesian S-G/all-centre AUX required")
        exponents = tuple(float.fromhex(x) for x in shell["exponents_hex"])
        coefficients = tuple(float.fromhex(x) for x in shell["native_normalized_coefficients_hex"])
        if (not exponents or len(exponents) != shell["nprimitive"]
                or len(exponents) != len(coefficients)
                or not all(math.isfinite(x) and x > 0 for x in exponents)
                or not all(map(math.isfinite, coefficients))):
            raise ValueError("invalid exact effective primitive data")
        for within, powers in enumerate(POWERS[l]):
            rows.append(dict(column=len(rows), shell=index, centre=centre,
                site=("C" if centre < 6 else "H")+str(centre % 6+1), l=l,
                component=within, powers=powers, centre_bohr=centres[centre],
                exponents=exponents, coefficients=coefficients))
    if len(rows) != snapshot["metadata"]["nbf"]:
        raise ValueError("complete AUX column count differs from snapshot")
    return rows


def configure_grid(options, recipe):
    if recipe != GRID_RECIPE:
        raise ValueError("not the exact captured Q-grid recipe")
    for key, value in recipe.items():
        setattr(options, key, value)


def validate_grid(grid, points, pin):
    import numpy as np
    if (not isinstance(grid, np.ndarray) or grid.dtype != np.dtype("<f8")
            or grid.shape != (points, 4) or not grid.flags.c_contiguous
            or not np.isfinite(grid).all()):
        raise ValueError("invalid exact canonical grid shape/dtype/values/layout")
    digest = hashlib.sha256(memoryview(grid).cast("B")).hexdigest()
    if digest != pin:
        raise ValueError("captured Q integration grid hash mismatch; comparison forbidden")
    return digest


def all_centres(auxiliary, ncentres, ncolumns):
    if auxiliary.nfunction != ncolumns or ncentres <= 0:
        raise ValueError("complete Cartesian AUX width/centres mismatch")
    return list(range(ncentres))


def selected_columns(rows):
    """All components of most diffuse shell at each l on C1,C2,H1.

    C2 diffuse f includes the captured worst-delta column254 (x*z^2).
    C has S-G, H S-F: all-even and odd-power components are both retained.
    """
    selected = []
    for centre in (0, 1, 6):
        for l in range(5):
            candidates = [r for r in rows if r["centre"] == centre and r["l"] == l]
            if candidates:
                shell = min(candidates, key=lambda r: (min(r["exponents"]), r["shell"]))["shell"]
                selected.extend(r["column"] for r in candidates if r["shell"] == shell)
    if 254 not in selected:
        raise ValueError("selected collocation checks omit worst-delta diffuse f")
    return tuple(selected)


def selected_points():
    # Actual atom-major grid indices, radial zero-based100/150, angular137/389.
    # Native grid has199 shells x590 directions per atom. No invented points.
    return tuple(a*199*590+r*590+ang for a in range(12)
                 for r, ang in ((100, 137), (150, 389)))


def stream(auxiliary, grid, rows, limits, *, selected_points, selected_columns, check_deadline):
    """One public collocation pass; C-level NumPy reductions, no scalar Ngrid*Naux.

    The primary U sums tile reductions in ascending tile order. A second U uses
    vector Kahan accumulation of the SAME tile reductions (no re-evaluation).
    A reports sum(abs(w*chi)); gamma/(1-gamma)*A describes an ordinary-U
    binary64 *summation* roundoff bound, conditional on computed products. It
    does not bound quadrature, collocation/exp or product error/underflow.
    """
    import numpy as np
    n, width = len(grid), len(rows)
    tile = limits["tile_points"]
    points, cols = tuple(selected_points), tuple(selected_columns)
    estimated_output = 5*8*width + len(points)*len(cols)*1024 + width*1024
    if (type(tile) is not int or not 1 <= tile <= 4096 or not n or not width
            or n*width > limits["max_values"]
            or estimated_output > limits["max_output_bytes"]
            or len(points)*len(cols) > limits.get("selected_checks", 4096)
            or len(set(points)) != len(points) or len(set(cols)) != len(cols)
            or any(type(p) is not int or not 0 <= p < n for p in points)
            or any(type(c) is not int or not 0 <= c < width for c in cols)):
        raise ValueError("grid/collocation work/tile/output/selection resource bound")
    centres = sorted({r["centre"] for r in rows})
    if centres != list(range(len(centres))):
        raise ValueError("non-complete AUX centres")
    sites = all_centres(auxiliary, len(centres), width)
    # <=32MiB native matrix plus32MiB product, bounded independently of full grid.
    if tile*width*8 > 32*1024**2:
        raise ValueError("collocation tile byte bound")
    u, compensated, correction, absolute = (np.zeros(width) for _ in range(4))
    checks = []
    count = 0
    for begin in range(0, n, tile):
        check_deadline()
        end = min(n, begin+tile)
        matrix = auxiliary.evaluate_screened(grid[begin:end, :3].tolist(), sites)
        values = np.asarray(matrix.np)
        if values.shape != (end-begin, width) or not np.isfinite(values).all():
            raise ValueError("nonfinite/wrong-size native collocation tile")
        for point in points:
            if begin <= point < end:
                for column in cols:
                    expected = oracle(grid[point, :3], rows[column])
                    native = float(values[point-begin, column])
                    checks.append(dict(point=point, xyz_hex=[float(x).hex() for x in grid[point, :3]],
                        column=column, native=native, oracle=expected,
                        native_minus_oracle=native-expected,
                        native_hex=native.hex(), oracle_hex=expected.hex()))
        products = np.multiply(values, grid[begin:end, 3, None])
        if not np.isfinite(products).all():
            raise ValueError("nonfinite weighted collocation tile")
        partial = np.sum(products, axis=0, dtype=np.float64)
        u += partial
        y = partial-correction
        next_sum = compensated+y
        correction = (next_sum-compensated)-y
        compensated = next_sum
        np.abs(products, out=products)
        absolute += np.sum(products, axis=0, dtype=np.float64)
        del values, products, matrix
        count += 1
    check_deadline()
    if not all(np.isfinite(x).all() for x in (u, compensated, absolute)):
        raise ValueError("nonfinite accumulated grid integral")
    eps = np.finfo(np.float64).eps/2
    operations = n+count+2
    gamma = operations*eps/(1-operations*eps)
    bound = np.nextafter((gamma/(1-gamma))*absolute, np.inf)
    return dict(U=u, U_compensated=compensated, absolute_sum=absolute,
        roundoff_bound=bound, tiles=count, collocation_checks=checks,
        accumulation="binary64 np.sum(axis=0) per contiguous point-major tile; ascending tile sums; "
                     "second vector Kahan across same tile sums; absolute weighted products same pass",
        error_bound_scope="nextafter(gamma/(1-gamma)*absolute_sum,+inf), gamma for Npoints+Ntiles+2; "
                          "ordinary-U summation only, computed products, normal arithmetic; "
                          "not exp/product/underflow/collocation/quadrature error or Kahan certificate")


def compare(u, q, s, rows):
    def summary(values):
        values = list(map(float, values))
        peak = max(range(len(values)), key=lambda i: abs(values[i]))
        minimum = min(range(len(values)), key=lambda i: values[i])
        maximum = max(range(len(values)), key=lambda i: values[i])
        return dict(maxabs=abs(values[peak]), l1=math.fsum(map(abs, values)),
                    l2=math.hypot(*values), signed_sum=math.fsum(values),
                    peak=dict(rows[peak], signed=values[peak]),
                    minimum=dict(rows[minimum], signed=values[minimum]),
                    maximum=dict(rows[maximum], signed=values[maximum]))
    if not len(u) == len(q) == len(s) == len(rows) or not rows:
        raise ValueError("comparison vector dimensions")
    uq = [float(a)-b for a, b in zip(u, q)]
    su = [b-float(a) for a, b in zip(u, s)]
    delta = [a-b for a, b in zip(s, q)]
    return dict(U_minus_q=summary(uq), S_minus_U=summary(su), deltaq=summary(delta),
        decomposition_maxabs=max(abs(a+b-d) for a, b, d in zip(uq, su, delta)),
        interpretation="U-q narrows whole-grid integration OR collocation; S-U combines exclusions, "
                       "ratio completeness and accumulation. No saved mask/E/V or response prediction.")


def write_vector(path, values, *, expected):
    values = list(map(float, values))
    if len(values) != expected or not all(map(math.isfinite, values)):
        raise ValueError("finite complete bounded vector required")
    data = struct.pack("<"+"d"*expected, *values)
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return dict(file=Path(path).name, dtype="<f8", shape=[expected], nbytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())
