# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Four explicit grid-only contracts; no engine imports or mutable old defaults."""
import hashlib
import math
import time

BASELINE_HASH = "a01a17a4ab55dab608883cf5a184bfc13a42a15fae15efecf0256a9df383ab29"
VECTOR_KEYS = ("U", "U_compensated", "absolute_sum", "roundoff_bound")


def declared_cases():
    return [dict(name=name, recipe=dict(radial_points=r, spherical_points=a,
        becke_smoothing=3, radius_scaling=1.0), points=n,
        expected_sha256=BASELINE_HASH if name == "baseline" else None,
        output="results/"+name)
        for name, r, a, n in (("baseline", 200, 590, 1408920),
            ("radial", 300, 590, 2116920), ("angular", 200, 974, 2325912),
            ("joint", 300, 974, 3494712))]


CASES = declared_cases()


def admit_case(case):
    if case not in declared_cases():
        raise ValueError("not one of four explicit grid cases")


def accounting(cases):
    if cases != declared_cases():
        raise ValueError("exact four cases in baseline-first order required")
    points = sum(c["points"] for c in cases)
    return dict(points=points, values=points*1146,
        per_case_values=[c["points"]*1146 for c in cases],
        tiles=sum((c["points"]+1023)//1024 for c in cases),
        max_grid_bytes=max(c["points"] for c in cases)*32,
        # canonical + native coordinates + one Python component-list transient,
        # rounded up; scientific matrices/products are separately bounded.
        grid_assembly_reservation_bytes=512*1024**2,
        tile_and_product_bytes=2*1024*1146*8,
        vector_bytes=4*5*1146*8,
        # Per-case <=4MiB JSON incl2160 checks + row/site/shell summaries;
        # vectors, basis/identity/terminal/comparison/metadata <=4MiB total.
        estimated_output_bytes=20*1024**2)


def configure(options, case):
    admit_case(case)
    for key, value in case["recipe"].items():
        setattr(options, key, value)


def layout(native, case):
    admit_case(case)
    r, a = case["recipe"]["radial_points"], case["recipe"]["spherical_points"]
    stride = (r-1)*a
    starts = [native.atom_start(i) for i in range(13)]
    if (native.npoints() != case["points"] or native.radial_points() != r
            or native.spherical_points() != a or starts != [i*stride for i in range(13)]
            or any(native.atom_npoints(i) != stride for i in range(12))):
        raise ValueError("returned grid counts/orders/per-atom layout differ")
    return dict(starts=starts, shells=r-1, angular=a)


def selection(actual):
    """Two actual radial/angular addresses per atom; regenerated, not nested.

    Fractional addresses reproduce old baseline100/137 and150/389 while
    rescaling independently for every returned layout. Coordinates come only
    from that case's grid; no claim of identical physical sample locations.
    """
    shells, angular, starts = actual["shells"], actual["angular"], actual["starts"]
    if len(starts) != 13 or starts != [i*shells*angular for i in range(13)]:
        raise ValueError("complete actual atom layout required")
    addresses = ((100*shells//199, 137*angular//590),
                 (150*shells//199, 389*angular//590))
    return [dict(atom=atom, radial=r, angular=a, index=starts[atom]+r*angular+a)
        for atom in range(12) for r, a in addresses]


def canonical_hash(grid, points):
    import numpy as np
    if (not isinstance(grid, np.ndarray) or grid.dtype != np.dtype("<f8")
            or grid.shape != (points, 4) or not grid.flags.c_contiguous
            or not np.isfinite(grid).all()):
        raise ValueError("canonical grid shape/dtype/layout/finite mismatch")
    return hashlib.sha256(memoryview(grid).cast("B")).hexdigest()


def hash_gate(case, measured, previous):
    admit_case(case)
    position = [c["name"] for c in declared_cases()].index(case["name"])
    if len(previous) != position:
        raise ValueError("baseline-first sequential hashes required")
    if case["name"] == "baseline" and measured != BASELINE_HASH:
        raise ValueError("baseline canonical hash reproduction failed before collocation")
    if measured in previous:
        raise ValueError("distinct declared grid falsely equals an earlier measured hash")


def deadline(seconds, *, clock=time.monotonic):
    started = clock()
    def check():
        if clock()-started > seconds:
            raise TimeoutError("cumulative study1500s payload deadline; no retry")
    return check


def summary(values, rows):
    values = list(map(float, values))
    if not values or len(values) != len(rows) or not all(map(math.isfinite, values)):
        raise ValueError("finite dimensioned signed comparison required")
    lo = min(range(len(values)), key=values.__getitem__)
    hi = max(range(len(values)), key=values.__getitem__)
    peak = max(range(len(values)), key=lambda i: abs(values[i]))
    return dict(minimum=dict(rows[lo], signed=values[lo]),
        maximum=dict(rows[hi], signed=values[hi]), peak=dict(rows[peak], signed=values[peak]),
        maxabs=abs(values[peak]), l2=math.hypot(*values),
        signed_sum=math.fsum(values), l1=math.fsum(map(abs, values)))


def diagnostics(u, q, s, rows, *, baseline):
    if not len(u) == len(q) == len(s) == len(rows):
        raise ValueError("comparison dimensions")
    error = [float(x)-y for x, y in zip(u, q)]
    result = dict(U_minus_q=summary(error, rows))
    for group, field in (("by_site", "site"), ("by_shell", "shell")):
        result[group] = {str(key): summary([error[i] for i in indices],
            [rows[i] for i in indices]) for key in sorted({r[field] for r in rows})
            for indices in [[i for i, r in enumerate(rows) if r[field] == key]]}
    cross = [float(y)-float(x) for x, y in zip(u, s)]
    if baseline:
        result["S_minus_U"] = summary(cross, rows)
        result["S_minus_U_label"] = "old S on reproduced baseline grid only; net partition/exclusion/accumulation"
    else:
        result["S_baseline_minus_U_cross_grid"] = summary(cross, rows)
        result["S_baseline_minus_U_label"] = (
            "cross_grid comparison only; NOT new-grid stockholder/exclusion error")
    return result


def remeasurement(measured, original):
    """Retain exact per-element signed remeasurement differences, then gate."""
    result = {}
    for key in VECTOR_KEYS:
        new, old = list(map(float, measured[key])), list(map(float, original[key]))
        if len(new) != len(old):
            raise ValueError("baseline vector dimensions")
        differences = [x-y for x, y in zip(new, old)]
        # Byte equality includes signed zero; no tolerant fallback.
        import struct
        exact = struct.pack("<"+"d"*len(new), *new) == struct.pack("<"+"d"*len(old), *old)
        result[key] = dict(exact=exact, maxabs=max(map(abs, differences)),
            l2=math.hypot(*differences), difference_hex=[x.hex() for x in differences],
            measured_hex=[x.hex() for x in new], original_hex=[x.hex() for x in old])
    return dict(all_exact=all(v["exact"] for v in result.values()), vectors=result,
                condition="all four baseline vectors byte-exact to14308581; no tolerance/fallback")


def effects(vectors):
    names = [c["name"] for c in declared_cases()]
    if set(vectors) != set(names) or len({len(v) for v in vectors.values()}) != 1:
        raise ValueError("all four complete dimensioned case vectors required")
    b, r, a, j = [vectors[n] for n in names]
    subtract = lambda x, y: [float(v)-float(w) for v, w in zip(x, y)]
    r590, r974 = subtract(r, b), subtract(j, a)
    return dict(radial_at_590=r590, radial_at_974=r974,
        angular_at_200=subtract(a, b), angular_at_300=subtract(j, r),
        interaction=subtract(r974, r590))
