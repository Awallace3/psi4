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

import math
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


# ---------------------------------------------------------------------------
# Gate 3 -- Casimir imaginary-frequency quadrature
# ---------------------------------------------------------------------------


def _casimir_reference():
    """{(omega0, n_freq): {k: (omega, tm1sq, weight)}} from the committed fixture."""
    ref = {}
    for line in (DATA / "camcasp_casimir_freq.dat").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        omega0, n_freq, k, omega, tm1sq, weight = line.split()
        ref.setdefault((float(omega0), int(n_freq)), {})[int(k)] = (
            float(omega),
            float(tm1sq),
            float(weight),
        )
    return ref


CASIMIR_REF = _casimir_reference()


@pytest.mark.parametrize("key", sorted(CASIMIR_REF))
def test_casimir_grid_bit_identical(key):
    """Every frequency, Jacobian and weight matches CamCASP's `frequencies` exactly.

    The quadrature is 6 lines of arithmetic over a literal table, so there is no
    excuse for anything less than bit parity here, and getting it means downstream
    C_n differences can never be blamed on the frequency grid.
    """
    omega0, n_freq = key
    grid = psi4.core.CasimirGrid(n_freq, omega0)

    assert grid.n_freq() == n_freq
    assert grid.omega0() == omega0

    for k, (omega, tm1sq, weight) in CASIMIR_REF[key].items():
        assert _ulps(grid.omega(k), omega) == 0, f"omega({k})"
        assert _ulps(grid.tm1sq(k), tm1sq) == 0, f"tm1sq({k})"
        assert _ulps(grid.weight(k), weight) == 0, f"weight({k})"


def test_casimir_grid_static_point():
    """Index 0 is the static point: zero frequency, and outside the quadrature."""
    grid = psi4.core.CasimirGrid(10)
    assert grid.omega(0) == 0.0
    assert grid.weight(0) == 0.0
    assert grid.cp_weight(0) == 0.0
    assert grid.wsq(0) == 0.0
    assert len(grid.omegas()) == 11


def test_casimir_grid_rejects_bad_n_freq():
    for n in (0, 3, 12):
        with pytest.raises(RuntimeError):
            psi4.core.CasimirGrid(n)
    with pytest.raises(RuntimeError):
        psi4.core.CasimirGrid(10).omega(11)


def test_casimir_grid_rejects_bad_omega0():
    for omega0 in (0.0, -0.3, math.nan, math.inf, -math.inf):
        with pytest.raises(RuntimeError, match="omega0"):
            psi4.core.CasimirGrid(10, omega0)


def test_grid_rejects_bad_atom_index(h2o):
    opts = psi4.core.IsaGridOptions()
    opts.radial_points = 4
    opts.spherical_points = 6
    grid = psi4.core.IsaGrid(h2o, opts)
    n = grid.natom()
    assert grid.atom_start(n) == grid.npoints()
    assert grid.atom_start(n - 1) + grid.atom_npoints(n - 1) == grid.npoints()
    assert grid.alpha(n - 1) > 0.0
    for get, bad in ((grid.atom_start, (-1, n + 1)), (grid.atom_npoints, (-1, n)), (grid.alpha, (-1, n))):
        for A in bad:
            with pytest.raises(RuntimeError):
                get(A)


def test_casimir_grid_ordering():
    """Frequencies come out ascending, and symmetric about omega0 in log scale.

    The two halves of the Gauss-Legendre rule map to omega0(1-t)/(1+t) and
    omega0(1+t)/(1-t), which are reciprocals scaled by omega0^2, so the pairing is
    exact.  This is what makes the unpacking in the constructor checkable without
    reference data.
    """
    grid = psi4.core.CasimirGrid(10)
    w = np.array([grid.omega(k) for k in range(1, 11)])
    assert (np.diff(w) > 0).all()
    assert np.allclose(w[:5] * w[::-1][:5], grid.omega0() ** 2, rtol=1e-15, atol=0)
    assert np.allclose([grid.weight(k) for k in range(1, 6)],
                       [grid.weight(k) for k in range(10, 5, -1)], rtol=0, atol=0)


def test_casimir_polder_single_pole():
    """C_6 for a one-term Unsold model, against its closed form.

    alpha(iw) = a w1^2 / (w1^2 + w^2) gives C_6 = (3/2) a_A a_B w1A w1B / (w1A + w1B).
    `cp_weight` carries the Gauss-Legendre weight, the Jacobian of the omega mapping
    and the 1/(2 pi) of the Casimir-Polder formula, so the isotropic C_6 is
    6 * sum_k cp_weight(k) alpha_A alpha_B -- this test pins that factor down.
    """
    grid = psi4.core.CasimirGrid(10)
    w = np.array([grid.omega(k) for k in range(1, 11)])
    cw = np.array([grid.cp_weight(k) for k in range(1, 11)])

    for (aA, w1A), (aB, w1B) in [((1.38, 0.85), (1.38, 0.85)), ((10.6, 0.55), (1.38, 0.85))]:
        alpha_A = aA * w1A**2 / (w1A**2 + w**2)
        alpha_B = aB * w1B**2 / (w1B**2 + w**2)
        exact = 1.5 * aA * aB * w1A * w1B / (w1A + w1B)
        # 1e-4 is the accuracy of a 10-point rule on this integrand, not a parity
        # tolerance; the grid itself is bit-identical to CamCASP (test above).
        assert np.isclose(6.0 * np.dot(cw, alpha_A * alpha_B), exact, rtol=1e-4, atol=0)


# --- Fit points: the random cloud the point-response refinement samples on ----
#
# The generator and the cloud are both bit-exact and stay that way: the
# refinement fits to the potential at these points, so a single extra or missing
# deviate shifts every later point and every fitted coefficient.
#
# The full 256-deep reference stream of each seed is not committed.  What is kept
# is the part that constrains the generator most per line: the head of each
# stream, which pins the lag-table initialization, and far checkpoints, which pin
# the lag indexing at draws a prefix comparison cannot reach.


PRAND_HEAD = {  # first eight dprand() draws after sdprnd(seed)
    0: [
        0.7913183967560748, 0.3741928192300996, 0.44861432892185005, 0.7428031178186686,
        0.6519068277859982, 0.8119357718628364, 0.09214209014210396, 0.8206511920147144,
    ],
    1: [
        0.6249940256745002, 0.4739767236995389, 0.33774460100239345, 0.9689091568136964,
        0.9894501642012402, 0.370424110430112, 0.8350173508391068, 0.6246675242825115,
    ],
    7: [
        0.6270499875570739, 0.07276176427472626, 0.6755453525578009, 0.4372540904407638,
        0.14793210299769322, 0.6505653461173164, 0.6730832975142309, 0.5388973351375128,
    ],
    9999: [
        0.7168473754051381, 0.2223250760738269, 0.8897056205590392, 0.5964859634692803,
        0.4657688706665557, 0.7045823942811531, 0.10825906528783459, 0.4158048498115648,
    ],
}

#: (seed, n, value): the n-th draw of the same stream, n up to 1e5.
PRAND_FAR = [
    (0, 1000, 0.3558536943794186),
    (1, 1000, 0.8229176297805547),
    (7, 1000, 0.5286973905111498),
    (9999, 1000, 0.7794555606863224),
    (0, 10000, 0.34760313364902484),
    (1, 10000, 0.3919020645028068),
    (7, 10000, 0.2389805735660362),
    (9999, 10000, 0.5977261887690788),
    (0, 100000, 0.4218999418199747),
    (1, 100000, 0.6455312107296698),
    (7, 100000, 0.16019401789973986),
    (9999, 100000, 0.20506340327750427),
]


@pytest.mark.parametrize("seed", sorted(PRAND_HEAD))
def test_maclaren_stream_head_bit_identical(seed):
    """The start of each reference stream, exactly, straight out of sdprnd."""
    want = PRAND_HEAD[seed]
    got = psi4.core.MaclarenRng(seed).take(len(want))
    assert np.array_equal(got, want), (seed, got, want)


@pytest.mark.parametrize("seed,n,want", PRAND_FAR)
def test_maclaren_far_stream_bit_identical(seed, n, want):
    """The n-th draw, n up to 1e5, so a lag-index bug cannot hide in a prefix."""
    rng = psi4.core.MaclarenRng(seed)
    assert rng.take(n)[-1] == want


def test_maclaren_reseed_restarts_stream():
    rng = psi4.core.MaclarenRng(1)
    first = rng.take(50)
    rng.take(500)
    rng.seed(1)
    assert np.array_equal(rng.take(50), first)
    with pytest.raises(ValueError, match="nonnegative"):
        rng.take(-1)
    assert rng.take(0).size == 0
    assert np.array_equal(rng.take(50), psi4.core.MaclarenRng(1).take(100)[50:])


def test_maclaren_seed_folds_like_fortran():
    """sdprnd takes mod(abs(iseed), 10000), so these three are the same stream."""
    a = psi4.core.MaclarenRng(7).take(20)
    for equivalent in (-7, 10007, 20007):
        assert np.array_equal(psi4.core.MaclarenRng(equivalent).take(20), a)


def test_maclaren_range():
    """Uniform on (0, 1), never exactly 0 or 1: dprand adds `tiny` to guarantee it."""
    x = psi4.core.MaclarenRng(1).take(200000)
    assert x.min() > 0.0 and x.max() < 1.0
    # 200k draws of a generator this old should still look flat.  The bound is
    # five binomial standard deviations, so this is a smoke test for a broken
    # range reduction, not a claim about the generator's statistical quality.
    nbin = 20
    counts, _ = np.histogram(x, bins=nbin, range=(0.0, 1.0))
    expected = len(x) / nbin
    sigma = np.sqrt(expected * (1.0 - 1.0 / nbin))
    assert np.abs(counts - expected).max() < 5.0 * sigma


@pytest.fixture(scope="module")
def reffit():
    return np.load(DATA / "camcasp_fit_points.npz")


def _fit_molecule(geom):
    mol = psi4.geometry(
        "units bohr\nno_com\nno_reorient\nsymmetry c1\n"
        + "\n".join(f"{int(Z)} {x} {y} {z}" for Z, x, y, z in geom)
    )
    mol.update_geometry()
    return mol


@pytest.mark.parametrize("name, npoints", [("h2o", 2000), ("hcl", 2000), ("ref500", 500)])
def test_fit_points_bit_identical(reffit, name, npoints):
    """``ref500`` is the cloud the H2O_props reference case's own refinement was fitted on.

    `tests/H2O_props/psi4/H2O-avtz.clt` declares `Options Tests`, and the
    `properties` run type's generated `SET Lattice` block therefore asks for
    `Random 500` rather than the production `Random 2000`
    (cluster_file_interface.F90::write_camcasp_1).

    The stored cloud comes from `oracle/make_lattice_oracle.sh`'s `latticedump`,
    i.e. from CamCASP's own `generate_lattice` and `random.f90`; it was also
    checked against those routines linked directly out of a built CamCASP, and
    all three routes agree bitwise.
    """
    mol = _fit_molecule(reffit[f"{name}_geom"])
    options = psi4.core.FitPointsOptions()
    options.npoints = npoints
    options.seed = 1
    points = psi4.core.FitPoints(mol, options)
    assert points.npoints() == npoints

    # The cube the candidates are drawn from has to match first: it is built from
    # the van der Waals radii and the centre of geometry, and if it is off by an
    # ulp every accept/reject decision downstream is suspect.
    assert points.dmax() == reffit[f"{name}_dmax"]
    assert np.array_equal(points.centre(), reffit[f"{name}_centre"])

    got = np.column_stack([points.x(), points.y(), points.z()])
    assert np.array_equal(got, reffit[f"{name}_points"])


def test_fit_points_lie_in_the_shell(reffit):
    """Every accepted point is outside 2 R_vdW of all atoms and inside 4 of one."""
    geom = reffit["h2o_geom"]
    mol = _fit_molecule(geom)
    options = psi4.core.FitPointsOptions()
    options.npoints = 500
    points = psi4.core.FitPoints(mol, options)

    xyz = np.column_stack([points.x(), points.y(), points.z()])
    radii = np.array([psi4.core.isapol_vdw_radius(int(Z)) for Z, *_ in geom])
    d = np.linalg.norm(xyz[:, None, :] - geom[None, :, 1:], axis=2)
    assert (d >= 2.0 * radii).all()
    assert (d < 4.0 * radii).any(axis=1).all()


def test_fit_points_seed_changes_the_cloud(reffit):
    mol = _fit_molecule(reffit["h2o_geom"])
    clouds = []
    for seed in (1, 2):
        options = psi4.core.FitPointsOptions()
        options.npoints = 100
        options.seed = seed
        p = psi4.core.FitPoints(mol, options)
        clouds.append(np.column_stack([p.x(), p.y(), p.z()]))
    assert not np.array_equal(*clouds)


def test_fit_points_rejects_empty_shell(reffit):
    mol = _fit_molecule(reffit["h2o_geom"])
    options = psi4.core.FitPointsOptions()
    options.lolim = 4.0
    options.hilim = 2.0
    with pytest.raises(RuntimeError, match="hilim must exceed lolim"):
        psi4.core.FitPoints(mol, options)
    options = psi4.core.FitPointsOptions()
    options.npoints = 0
    with pytest.raises(RuntimeError, match="npoints must be positive"):
        psi4.core.FitPoints(mol, options)


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
