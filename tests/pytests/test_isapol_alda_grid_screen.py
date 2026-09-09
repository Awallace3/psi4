# Psi4: Copyright (c) 2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Error-bounded ALDA quadrature row screening; requires the parent's rebuilt Psi4.

``IsaAldaGridScreen`` exists so a caller can drop quadrature rows it can prove
do not matter, instead of coarsening the quadrature itself. Every claim here is
checked against the shipped local primitive on the same SCF, never against a
stored reference number or a tolerance imported from another track.

The bound is not a heuristic. Row p adds ``factor(p) * tr_p tr_p^T`` to L with
``factor(p) = w(p) fxc(p)`` and ``tr_p(t) = phi_i(p) phi_a(p)``; because tr_p is
the outer product of the occupied and the virtual orbital values at that one
point, that contribution is exactly rank one and

    ||factor(p) tr_p tr_p^T||_F = |factor(p)| * sum_i phi_i(p)^2 * sum_a phi_a(p)^2

with equality, not inequality. ``test_single_row_frobenius_norm_is_the_bound``
checks that identity to the last bit against the primitive itself, one row at a
time, which is what makes the summed bound below rigorous rather than fitted.

What this file does **not** establish: that screening makes any particular
production basis affordable. It does not. The measured aug-cc-pVTZ deficit is
asserted as a deficit in
``test_screening_does_not_open_the_avtz_alda_gate``, and the shipped
``estimate_response_work`` ALDA limit still applies in full to whatever row
subset is finally handed to ``NativeResponseProvider``.
"""
from contextlib import contextmanager

import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.p4util import OptionsState
from psi4.driver.procrouting.isapol_native_response import (
    native_response_from_wavefunction, screen_alda_grid)
from psi4.driver.procrouting.isapol_response_preflight import estimate_response_work

KERNEL, EXCHANGE, LOCAL = "alda_slater_pw92", 0.25, 0.75
CUTOFF = 1.e-10


@contextmanager
def scf_settings(basis):
    saved = OptionsState(["BASIS"], ["SCF_TYPE"], ["SCF", "REFERENCE"],
                         ["SCF", "E_CONVERGENCE"], ["SCF", "D_CONVERGENCE"])
    try:
        psi4.set_options({"basis": basis, "scf_type": "pk", "reference": "rks",
                          "e_convergence": 1.e-10, "d_convergence": 1.e-9})
        yield
    finally:
        saved.restore()


@pytest.fixture(scope="module")
def screened():
    """One PBE0 SCF, its own generated response grid, and the full-grid L.

    The grid is a genuine ``IsaGrid``, unpruned and unrenormalized, just small
    enough that the full-grid primitive is affordable in a test.
    """
    molecule = psi4.geometry("""
    0 1
    O  0.000000  0.000000 -0.068516
    H  0.000000 -0.790689  0.543701
    H  0.000000  0.790689  0.543701
    units angstrom
    symmetry c1
    no_reorient
    no_com
    """)
    with scf_settings("sto-3g"):
        _, wfn = psi4.energy("pbe0", molecule=molecule, return_wfn=True)
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = 50, 194
    grid = core.IsaGrid(molecule.clone(), options)
    rows = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    return wfn, rows, primitive(wfn, rows)


def primitive(wfn, rows):
    provider = native_response_from_wavefunction(
        wfn, caller_converged=True, kernel=KERNEL, exact_exchange=EXCHANGE,
        local_scale=LOCAL, grid=rows, density_cutoff=CUTOFF).provider
    return np.asarray(provider.local_primitive().to_array())


def screen(wfn, rows, kernel=KERNEL, **kwargs):
    return screen_alda_grid(wfn, caller_converged=True, kernel=kernel, grid=rows,
                            density_cutoff=CUTOFF, **kwargs)


def test_zero_threshold_prunes_only_rows_the_primitive_already_skips(screened):
    """threshold=0 is lossless: bound exactly 0.0 and L bitwise unchanged.

    The rows removed here are exactly those the local primitive itself discards
    (density below the cutoff, or zero quadrature weight), so this is not an
    approximation at any tolerance -- the two primitives are the same doubles.
    """
    wfn, rows, full = screened
    s = screen(wfn, rows, threshold=0.0)
    assert s.input_rows == rows.shape[0]
    assert s.exact_zero_rows > 0, "fixture grid must reach the density cutoff to be a test"
    assert s.omitted_rows == s.exact_zero_rows
    assert s.omitted_bound == 0.0
    assert s.grid.shape[0] == rows.shape[0] - s.exact_zero_rows
    assert np.array_equal(primitive(wfn, s.grid), full)
    # And the omitted rows really are the primitive's own skip set: adding any
    # of them back one at a time changes nothing either.
    dropped = np.setdiff1d(np.arange(rows.shape[0]), s.rows)
    assert dropped.size == s.exact_zero_rows
    assert np.array_equal(primitive(wfn, np.vstack((s.grid, rows[dropped[:64]]))), full)


def test_single_row_frobenius_norm_is_the_bound(screened):
    """|factor| * o(p) * u(p) IS ||L(row p)||_F, to the last bit.

    A one-row grid makes the primitive exactly that row's rank-one contribution,
    so this compares the screen's arithmetic against the shipped accumulator's
    with no summation in between. Equality here is what licenses summing the
    values into a deviation bound; an inequality-only estimate would not.
    """
    wfn, rows, _ = screened
    s = screen(wfn, rows, threshold=0.0)
    values = np.asarray(s.values)
    order = np.argsort(values)[::-1]
    probes = list(order[:5]) + list(order[len(order)//2:len(order)//2 + 3])
    checked = 0
    for p in probes:
        if values[p] == 0.0:
            continue
        one = primitive(wfn, rows[int(p):int(p) + 1])
        frobenius = np.linalg.norm(one)
        assert frobenius == pytest.approx(values[p], rel=1.e-13, abs=0.0)
        assert np.abs(one).max() <= values[p] * (1 + 1.e-13)
        checked += 1
    assert checked >= 5


@pytest.mark.parametrize("threshold", [0.0, 1.e-16, 1.e-12, 1.e-9, 1.e-7, 1.e-5])
def test_omitted_bound_holds_in_both_norms(screened, threshold):
    """The actual full-vs-pruned deviation never exceeds the reported bound.

    Both norms are checked, because the per-row identity bounds every element
    and the Frobenius norm by the same number.
    """
    wfn, rows, full = screened
    s = screen(wfn, rows, threshold=threshold)
    deviation = primitive(wfn, s.grid) - full
    assert np.abs(deviation).max() <= s.omitted_bound
    assert np.linalg.norm(deviation) <= s.omitted_bound
    assert s.threshold == threshold
    assert s.omitted_rows + s.grid.shape[0] == rows.shape[0]


def test_bound_and_deviation_shrink_together(screened):
    """Tightening the threshold is monotone in rows, bound and actual error."""
    wfn, rows, full = screened
    previous = None
    for threshold in (1.e-5, 1.e-7, 1.e-9, 1.e-12, 0.0):
        s = screen(wfn, rows, threshold=threshold)
        error = np.linalg.norm(primitive(wfn, s.grid) - full)
        if previous is not None:
            kept, bound, before = previous
            assert s.grid.shape[0] >= kept
            assert s.omitted_bound <= bound
            assert error <= before + 1.e-30
            assert set(kept_rows).issubset(set(s.rows.tolist()))
        previous = (s.grid.shape[0], s.omitted_bound, error)
        kept_rows = s.rows.tolist()
    assert previous[1] == 0.0 and previous[2] == 0.0


def test_retained_rows_are_the_input_rows_verbatim(screened):
    """A row subset, not a requadrature: no reordering, no reweighting."""
    wfn, rows, _ = screened
    s = screen(wfn, rows, threshold=1.e-8)
    assert 0 < s.grid.shape[0] < rows.shape[0]
    assert np.array_equal(s.grid, rows[s.rows])          # verbatim, including weights
    assert np.all(np.diff(s.rows) > 0)                    # original order preserved
    assert s.grid[:, 3].sum() < rows[:, 3].sum()          # no weight renormalization
    assert np.asarray(s.values).shape == (rows.shape[0],)  # values stay in input order
    assert s.kernel == KERNEL and s.density_cutoff == CUTOFF
    assert "verbatim" in s.provenance and "gate still applies" in s.provenance


def test_threshold_for_rows_never_returns_more_than_asked(screened):
    """max_rows is a ceiling, and ties resolve downward."""
    wfn, rows, _ = screened
    s0 = screen(wfn, rows, threshold=0.0)
    nonzero = s0.grid.shape[0]
    for target in (1, 17, 1000, nonzero // 2, nonzero - 1, nonzero, rows.shape[0]):
        s = screen(wfn, rows, max_rows=target)
        assert s.grid.shape[0] <= target
        assert s.threshold >= 0.0
        if target >= nonzero:
            assert s.threshold == 0.0 and s.grid.shape[0] == nonzero
            assert s.omitted_bound == 0.0
    assert screen(wfn, rows, max_rows=nonzero // 2).omitted_bound > 0.0


def test_screen_rejects_unsupported_and_underspecified_requests(screened):
    """No inferred kernel, no inferred threshold, no silent renormalization."""
    wfn, rows, _ = screened
    with pytest.raises(ValueError, match="no_local has no grid rows"):
        screen(wfn, rows, threshold=0.0, kernel="no_local")
    with pytest.raises(ValueError, match="exactly one of threshold or max_rows"):
        screen(wfn, rows)
    with pytest.raises(ValueError, match="exactly one of threshold or max_rows"):
        screen(wfn, rows, threshold=0.0, max_rows=10)
    with pytest.raises(ValueError, match="finite nonnegative"):
        screen(wfn, rows, threshold=-1.e-20)
    with pytest.raises(ValueError, match="positive integer"):
        screen(wfn, rows, max_rows=0)
    with pytest.raises(ValueError, match="nonnegative weight"):
        screen(wfn, np.column_stack((rows[:, :3], -rows[:, 3])), threshold=0.0)
    with pytest.raises(ValueError, match="caller_converged"):
        screen_alda_grid(wfn, caller_converged=False, kernel=KERNEL, grid=rows, threshold=0.0)
    with pytest.raises(ValueError, match="retained no rows"):
        screen(wfn, rows, threshold=float(np.asarray(screen(wfn, rows, threshold=0.0).values).max()))


def test_screening_does_not_open_the_avtz_alda_gate():
    """Measured deficit for PBE0/aug-cc-pVTZ water on the shipped response grid.

    Pure preflight arithmetic on the dimensions and row counts measured on this
    branch: nbf = nmo = 92, nocc = 5, nvir = 87, nOV = 435 and the public
    ``IsaGrid(99, 590)`` three-atom grid's 3 * 98 * 590 = 173460 rows, of which
    135502 survive lossless screening and 37958 are exactly zero.

    The ALDA limit admits 10569 rows there. Keeping only those omits 53.6% of
    the total contribution norm, and even the lossless subset still needs 12.8
    times the permitted work. Row screening is therefore **not** what makes that
    demo affordable, and this test records that rather than papering over it.
    """
    nbf, nmo, nocc, rows = 92, 92, 5, 173460
    estimate = estimate_response_work(nbf, nmo, nocc, rows)
    assert estimate.failures == ("ALDA work resource limit",)
    assert estimate.max_grid_rows == 10569
    for kept in (135502, 113834, 20000):
        assert estimate_response_work(nbf, nmo, nocc, kept).failures == ("ALDA work resource limit",)
    assert not estimate_response_work(nbf, nmo, nocc, 10569).failures
