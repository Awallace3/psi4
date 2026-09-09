# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Focused ALDA row-parallel regression using one frozen small SCF fixture."""
import numpy as np
import pytest
from test_isapol_native_response import water, make, grid_and_local
from test_isapol_native_performance import threads

pytestmark = [pytest.mark.psi, pytest.mark.api]


@pytest.mark.parametrize("kernel", ["alda_slater", "alda_slater_pw92", "alda_slater_vwn"])
def test_alda_disjoint_rows_across_threads_and_block_boundaries(water, kernel):
    with threads(1):
        grid, expected = grid_and_local(water, kernel)
        serial = make(water, kernel=kernel, exact_exchange=.25, local_scale=.75, grid=grid).provider
        local = serial.local_primitive().np.copy()
        np.testing.assert_allclose(local, expected, rtol=2.e-9, atol=2.e-11)
    for n in (2, 4):
        with threads(n):
            parallel = make(water, kernel=kernel, exact_exchange=.25, local_scale=.75, grid=grid).provider
            # Existing native ALDA tolerances, not a relaxed performance gate.
            # MO collocation uses BLAS outside our region; BLAS may itself select
            # a different summation tree when Psi4's thread setting changes.
            for name in ("local_primitive", "h1", "h2"):
                np.testing.assert_allclose(getattr(parallel, name)().np, getattr(serial, name)().np,
                                           rtol=2.e-9, atol=2.e-11)
            altered = parallel.local_primitive()
            altered.np[:] = 0.
            np.testing.assert_allclose(parallel.local_primitive().np, local, rtol=2.e-9, atol=2.e-11)
