# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Pure ISA accounting, not grid construction, admission or qualification.

Version1 tightens only the closed 100/200 recipe. Other recipes retain the
previous conservative Lebedev envelope; no table approximation constructs or
changes a scientific grid here. No NumPy, Psi4 or installed-runtime imports.
"""
import math


def isa_grid_rows(sites, *, radial_points, spherical_points, becke_smoothing,
                  radius_scaling, radius_policy, neighbour_policy):
    """Account for the native radial shells and rounded Lebedev sphere.

    isa_grid.cc generates n_r-1 shells per atom, with every angular point
    retained after Becke weighting. cubature.cc's selector picks the smallest
    tabulated count >= the request (clamped to5294); 194 < 200 <= 230.
    The exact branch is intentionally narrower than that cardinality proof.
    All other legal recipes keep the old min(5294,2*requested+6) envelope.
    """
    if (type(sites) is not int or sites <= 0
            or type(radial_points) is not int or radial_points < 2
            or type(spherical_points) is not int or spherical_points < 1
            or type(becke_smoothing) is not int or becke_smoothing < 0
            or not math.isfinite(radius_scaling) or radius_scaling <= 0):
        raise ValueError("finite native ISA grid dimensions required")
    closed = (radial_points == 100 and spherical_points == 200
              and becke_smoothing == 3 and radius_scaling == 1.
              and radius_policy == "native_tabulated_bragg_slater"
              and neighbour_policy == "all_sites_unscreened_full_molecular_grid")
    angular = 230 if closed else min(5294, 2*spherical_points+6)
    return sites*(radial_points-1)*angular


def q_preparation_work(sites, naux, rank, npoint):
    """Additional high-thread Q preparation/control envelope, at ALL threads.

    The preexisting 4*npoint*sites*(rank+1)^2*naux term charges the unchanged
    multiply/add column traversal. Do not discount it by workers or screening.
    Extra source-level scalar-operation allowances for partitioned_response.cc:
      * 64/point: active initialization, flags and preparation-loop indexing;
      * 32/component/point: prefactor initialization, indexing and stores;
      * 32/component/tile/point: active checks, prefactor loads and point loops;
      * 4096/batch +256/task/batch: bounded setup, task intervals, serial
        dispatch or parallel_work's <=200 worker slots/scans and finalization.
    These deliberately exceed the added straight-line operations/loop steps;
    they are not cycle, OpenMP-runtime, allocator or vendor-workspace bounds.
    Charge even excluded points, partial batches and the low-thread path which
    does not allocate/use these arrays. No thread-count-dependent admission.
    """
    if (type(sites) is not int or sites <= 0
            or type(naux) is not int or naux <= 0
            or type(rank) is not int or not 0 <= rank <= 4
            or type(npoint) is not int or npoint < 0):
        raise ValueError("finite native Q preparation dimensions required")
    block = min(4096, 1048576//naux)
    if not block:
        raise ValueError("AUX multipole row exceeds collocation scratch budget")
    components = (rank+1)**2
    tiles = min(8, (naux+63)//64)
    batches = max(1, (npoint+block-1)//block)
    return sites*(npoint*(64+32*components+32*components*tiles)
                  +batches*(4096+256*components*tiles))
