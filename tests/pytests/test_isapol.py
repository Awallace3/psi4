"""Parity tests for libisapol against CamCASP 6.0.051.

The whole point of this module is that it reproduces another code's arithmetic, so
the tolerances here are deliberately absurd: element data and grid coordinates are
compared *bit for bit*, and quadrature weights to within one unit in the last place.
If one of these starts failing by a few ulp, do not relax the tolerance -- something
re-associated a floating-point expression.  That has already happened twice: the
element table's single-precision Fortran literals, and gfortran contracting
`r*x + c` into an FMA under `-march=native`.

Reference data lives in data_isapol/; its README says how to regenerate it.
"""

import pathlib

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

DATA = pathlib.Path(__file__).parent / "data_isapol"


def _ulps(got, want):
    """Distance from `got` to `want` in representable doubles."""
    g = np.asarray(got, dtype=np.float64).view(np.int64)
    w = np.asarray(want, dtype=np.float64).view(np.int64)
    # map to a monotone ordering across the sign boundary
    g = np.where(g < 0, np.int64(-(2**63)) - g, g)
    w = np.where(w < 0, np.int64(-(2**63)) - w, w)
    return np.abs(g - w)


@pytest.fixture(scope="module")
def atomprop():
    rows = (DATA / "camcasp_atomprop.dat").read_text().split("\n")
    maxz = int(rows[0])
    out = {}
    for line in rows[1 : maxz + 2]:
        f = line.split()
        out[int(f[0])] = (f[1], *(float(x) for x in f[2:7]))
    return out


@pytest.fixture(scope="module")
def refgrid():
    return np.load(DATA / "camcasp_grid_h2o.npz")


@pytest.fixture(scope="module")
def h2o(refgrid):
    xyz = refgrid["xyz"]
    mol = psi4.geometry(
        "units bohr\nno_com\nno_reorient\nsymmetry c1\n"
        + "\n".join(f"{Z} {x} {y} {z}" for Z, (x, y, z) in zip(refgrid["Z"], xyz))
    )
    mol.update_geometry()
    return mol


# ---------------------------------------------------------------- gate 5: tables

def test_element_symbols(atomprop):
    for Z, row in atomprop.items():
        assert psi4.core.isapol_element_symbol(Z).strip() == row[0].strip(), f"Z={Z}"


@pytest.mark.parametrize(
    "column, getter",
    [
        (1, "isapol_slater_radius"),
        (2, "isapol_vdw_radius_bondi"),
        (3, "isapol_vdw_radius_grimme"),
        (4, "isapol_c6_grimme"),
        (5, "isapol_covalent_radius"),
    ],
)
def test_atomprop_bit_identical(atomprop, column, getter):
    """CamCASP stores these as *single-precision* Fortran literals.

    Transcribing them as `double` gets you 4e-8 relative error, which is eight
    orders of magnitude too large.  Hence the exact comparison.
    """
    fn = getattr(psi4.core, getter)
    for Z, row in atomprop.items():
        want = row[column]
        if getter == "isapol_slater_radius" and want == 0.0:
            # CamCASP has no Slater radius past Z=54; the grid must say so loudly
            # rather than divide by zero.
            with pytest.raises(Exception):
                fn(Z)
            continue
        assert fn(Z) == want, f"Z={Z} {getter}"


# -------------------------------------------------------------- gate 5b: the grid

def test_grid_matches_camcasp(refgrid, h2o):
    opts = psi4.core.IsaGridOptions()
    opts.radial_points = int(refgrid["n_r"])
    opts.spherical_points = int(refgrid["n_a"])
    opts.becke_smoothing = int(refgrid["k_mu"])
    opts.radius_scaling = float(refgrid["rscale"])
    grid = psi4.core.IsaGrid(h2o, opts)

    n_r, n_a = int(refgrid["n_r"]), int(refgrid["n_a"])
    assert grid.natom() == 3
    # n_r is CamCASP's radial count, and the Euler-MacLaurin map emits n_r - 1 shells
    assert grid.npoints() == 3 * (n_r - 1) * n_a
    assert grid.npoints() == len(refgrid["grid"])
    assert [grid.atom_start(a) for a in range(4)] == list(refgrid["starts"])

    ref = refgrid["grid"]
    got = np.column_stack([grid.x(), grid.y(), grid.z(), grid.w()])

    # Psi4's Lebedev generators enumerate the octahedral sign patterns in a
    # different order than CamCASP's gen_oh, so compare shell by shell after
    # sorting.  Nothing physical depends on the order.
    def canonical(block):
        key = np.round(block[:, :3], 12) + 0.0  # +0.0 folds -0.0 onto 0.0
        return block[np.lexsort((key[:, 2], key[:, 1], key[:, 0]))]

    weight_ulps = []
    for lo in range(0, len(ref), n_a):
        r = canonical(ref[lo : lo + n_a])
        g = canonical(got[lo : lo + n_a])
        assert np.array_equal(r[:, :3], g[:, :3]), f"coordinates differ at point {lo}"
        weight_ulps.append(_ulps(g[:, 3], r[:, 3]).max())

    # The residual is the association of the 4*pi factor, which Psi4 folds into the
    # tabulated Lebedev weight and CamCASP folds into the radial one.  Two roundings
    # in a different order, hence up to 2 ulp.
    assert max(weight_ulps) <= 2, f"weights differ by {max(weight_ulps)} ulp"

    got_sum = np.asarray(grid.w()).sum()
    want_sum = ref[:, 3].sum()
    assert _ulps(got_sum, want_sum) <= 2, f"{got_sum!r} != {want_sum!r}"


def test_grid_option_defaults_are_the_camcasp_module_defaults():
    """CamCASP ``src/parameters.f90:189-197`` declares the grid module defaults.

        par_num_radial_points    = 80
        par_num_angular_points   = 590
        par_becke_smoothing      = 3
        par_radius_scaling       = 1.0_dp
        par_integration_grid_type = 1   (Lebedev)

    ``src/types_secondary.F90:136-138`` initialises ``NumRadPoints``,
    ``NumAngPoints``, ``BeckeSmoothPar``, ``RadiusScaling`` and ``GridType`` from
    them, and ``num_integration_grid.F90:438-442`` copies them straight into the
    ``atom_grids`` module before ``make_grid``.  These are declared model
    parameters, not tolerances: a grid declared at any other value is a
    different model.  Only Lebedev (GridType 1) is implemented here, which is
    why there is no grid-type knob to pin.
    """
    opts = psi4.core.IsaGridOptions()
    assert opts.radial_points == 80
    assert opts.spherical_points == 590
    assert opts.becke_smoothing == 3
    assert opts.radius_scaling == 1.0


def test_grid_reproduces_the_reference_protocol_summary(h2o):
    """The isa-pol protocol's own ``SET GRID { Angular 400 / Radial 100 }``.

    CamCASP's reference run of this molecule (water.out, "Integration grid
    summary") reports

        Total number of points =       128898
        Number of angular points          434
        Number of radial  points          100
        Atom     Total   Angular    Radial
        O1           42966       434       100

    so 400 is rounded *up* to the tabulated Lebedev size 434 by ``Lbdv()``, and
    each atom carries 42966 = 99 x 434 points because the Euler-MacLaurin map
    emits ``n_r - 1`` shells.  The ``refgrid`` fixture already compares the
    points themselves one by one, but only on a small (8/110) grid; this pins
    the shape of the production grid against the reference's own published
    summary.
    """
    opts = psi4.core.IsaGridOptions()
    opts.radial_points = 100
    opts.spherical_points = 400
    grid = psi4.core.IsaGrid(h2o, opts)

    assert grid.spherical_points() == 434
    assert grid.radial_points() == 100
    assert grid.npoints() == 128898
    assert [grid.atom_npoints(a) for a in range(3)] == [42966, 42966, 42966]
    assert [grid.atom_start(a) for a in range(4)] == [0, 42966, 85932, 128898]

    # num_integration_grid.F90:433 defaults every site's radius to
    # AtomProp(Z)%Rslater, and rscale multiplies it, so alpha is the tabulated
    # Bragg-Slater radius itself at the declared RadiusScaling = 1.
    for a, Z in enumerate((8, 1, 1)):
        assert grid.alpha(a) == psi4.core.isapol_slater_radius(Z)


def test_grid_integrates_atomic_gaussians(h2o):
    """A reference-free check that the assembled grid is actually a quadrature.

    Sum of normalised s-Gaussians on the three nuclei integrates to 3.  This
    exercises the Becke partition in the way the ISA solver does -- each atomic
    sub-grid contributes only its own share -- and would catch a partition that is
    subtly not a partition of unity, which the point-by-point comparisons above
    would not.
    """
    opts = psi4.core.IsaGridOptions()
    opts.radial_points = 60
    opts.spherical_points = 302
    grid = psi4.core.IsaGrid(h2o, opts)

    xyz = np.column_stack([grid.x(), grid.y(), grid.z()])
    w = np.asarray(grid.w())
    assert np.isfinite(w).all()
    assert (w >= 0).all()  # points buried inside a neighbour legitimately get 0

    nuc = h2o.geometry().np
    alpha = 1.7
    f = np.zeros(len(w))
    for R in nuc:
        d2 = ((xyz - R) ** 2).sum(axis=1)
        f += (alpha / np.pi) ** 1.5 * np.exp(-alpha * d2)

    # Tolerance is set by the quadrature, not by parity: the Euler-MacLaurin
    # radial map is only so good on a tight Gaussian sitting on hydrogen's
    # 0.94 a0 Slater radius.
    assert np.isclose(np.dot(w, f), h2o.natom(), rtol=0, atol=1e-6)


def test_vdw_radius_tables_are_distinct():
    """MODULE radii is double precision; AtomProp's copy is float32-rounded.

    Both cite Bondi (1964) and agree to seven digits, but only one of them is
    what the lattice generator uses.  If this ever starts passing as an equality,
    someone has collapsed the two tables and broken fit-point parity.
    """
    for Z in (1, 6, 7, 8, 17):
        double = psi4.core.isapol_vdw_radius(Z)
        single = psi4.core.isapol_vdw_radius_bondi(Z)
        assert double != single
        assert abs(double - single) < 1e-6 * double

    # Elements MODULE radii leaves out fall back on vdwdef rather than throwing.
    assert psi4.core.isapol_vdw_radius(4) == 2.5
    assert psi4.core.isapol_vdw_radius(92) == 2.5
    assert psi4.core.isapol_vdw_radius(0) == 0.0
