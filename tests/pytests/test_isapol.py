"""Parity tests for libisapol against CamCASP 6.0.051.

The whole point of this module is that it reproduces another code's arithmetic, so
the tolerances here are deliberately absurd: element data and grid coordinates are
compared *bit for bit*, and quadrature weights to within one unit in the last place.
If one of these starts failing by a few ulp, do not relax the tolerance -- something
re-associated a floating-point expression.  See psi4/src/psi4/libisapol/SPEC.md
sections 3.5.3 and 3.5.5, which document the two times that has already happened.

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
    """CamCASP stores these as *single-precision* Fortran literals (SPEC.md 3.5.3).

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


def test_slater_radius_rejects_heavy_elements():
    with pytest.raises(Exception):
        psi4.core.isapol_slater_radius(55)


# -------------------------------------------------------------- gate 5b: the grid

def test_grid_shape(refgrid, h2o):
    opts = psi4.core.IsaGridOptions()
    opts.radial_points = int(refgrid["n_r"])
    opts.spherical_points = int(refgrid["n_a"])
    grid = psi4.core.IsaGrid(h2o, opts)

    n_r, n_a = int(refgrid["n_r"]), int(refgrid["n_a"])
    assert grid.natom() == 3
    # n_r is CamCASP's radial count, and the Euler-MacLaurin map emits n_r - 1 shells
    assert grid.npoints() == 3 * (n_r - 1) * n_a
    assert grid.npoints() == len(refgrid["grid"])
    assert [grid.atom_start(a) for a in range(4)] == list(refgrid["starts"])


def test_grid_matches_camcasp(refgrid, h2o):
    opts = psi4.core.IsaGridOptions()
    opts.radial_points = int(refgrid["n_r"])
    opts.spherical_points = int(refgrid["n_a"])
    opts.becke_smoothing = int(refgrid["k_mu"])
    opts.radius_scaling = float(refgrid["rscale"])
    grid = psi4.core.IsaGrid(h2o, opts)

    ref = refgrid["grid"]
    got = np.column_stack([grid.x(), grid.y(), grid.z(), grid.w()])
    n_a = int(refgrid["n_a"])

    # Psi4's Lebedev generators enumerate the octahedral sign patterns in a
    # different order than CamCASP's gen_oh, so compare shell by shell after
    # sorting.  Nothing physical depends on the order (SPEC.md 3.5.5).
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


def test_grid_weights_sum(refgrid, h2o):
    opts = psi4.core.IsaGridOptions()
    opts.radial_points = int(refgrid["n_r"])
    opts.spherical_points = int(refgrid["n_a"])
    grid = psi4.core.IsaGrid(h2o, opts)
    got = np.asarray(grid.w()).sum()
    want = refgrid["grid"][:, 3].sum()
    assert _ulps(got, want) <= 2, f"{got!r} != {want!r}"


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
# Gate 3 -- Casimir imaginary-frequency quadrature (SPEC.md 9.1)
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
    for n in (0, 1, 3, 9, 12):
        with pytest.raises(RuntimeError):
            psi4.core.CasimirGrid(n)
    with pytest.raises(RuntimeError):
        psi4.core.CasimirGrid(10).omega(11)


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
# Gates 1 and 2 of SPEC.md 15.  Both are bit-exact and stay that way: the
# refinement fits to the potential at these points, so a single extra or missing
# deviate shifts every later point and every fitted coefficient.


def _prand_reference():
    stream, check = {}, []
    for line in (DATA / "camcasp_prand.dat").read_text().splitlines():
        if line.startswith("#"):
            continue
        kind, seed, i, value = line.split()
        if kind == "stream":
            stream.setdefault(int(seed), []).append(float(value))
        else:
            check.append((int(seed), int(i), float(value)))
    return {s: np.array(v) for s, v in stream.items()}, check


PRAND_STREAM, PRAND_CHECK = _prand_reference()


@pytest.mark.parametrize("seed", sorted(PRAND_STREAM))
def test_maclaren_stream_bit_identical(seed):
    want = PRAND_STREAM[seed]
    got = psi4.core.MaclarenRng(seed).take(len(want))
    assert np.array_equal(got, want)


@pytest.mark.parametrize("seed,n,want", PRAND_CHECK)
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


@pytest.mark.parametrize("name", ["h2o", "hcl"])
def test_fit_points_bit_identical(reffit, name):
    mol = _fit_molecule(reffit[f"{name}_geom"])
    options = psi4.core.FitPointsOptions()
    options.npoints = 2000
    options.seed = 1
    points = psi4.core.FitPoints(mol, options)

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
    with pytest.raises(RuntimeError):
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


# --- Recoupling tables: the anisotropic dispersion coefficients ---------------
#
# Gate 4 of SPEC.md 15.  CamCASP's c6code.f90 ... c12code.f90 are machine-generated
# Fortran fragments; `oracle/parse_cncode.py` reduces them to the fixture read here
# and to the table libisapol compiles in, so the two cannot drift apart silently.
# On top of that parity check, the invariants below are derived from the theory
# rather than from CamCASP, and would catch a table that was self-consistently wrong.


def _recoupling_reference():
    """{(n, L1, L2, J): [(p, q, r, s, la, lap, lb, lbp, ip), ...]} from the fixture."""
    blocks, key = {}, None
    for line in (DATA / "camcasp_recoupling.dat").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if fields[0] == "block":
            key = tuple(int(f) for f in fields[1:5])
            assert key not in blocks
            blocks[key] = []
            assert int(fields[5]) >= 1
        else:
            blocks[key].append(tuple(int(f) for f in fields))
    return blocks


RECOUPLING_REF = _recoupling_reference()

# casimir.f90:112-123, with CamCASP's padding to width 3 stripped.
COMPONENT_LABELS = [
    lab
    for L in range(9)
    for lab in [f"{L}0"] + [f"{L}{K}{cs}" for K in range(1, L + 1) for cs in "cs"]
]


def _cxx_block(key):
    return psi4.core.isapol_recoupling_block(*key)


def _as_tuple(term):
    return (term.p, term.q, term.r, term.s, term.la, term.lap, term.lb, term.lbp, term.ipow)


def test_recoupling_blocks_match_fixture():
    """Every tabulated block, and nothing else, with its terms in CamCASP's order."""
    got = {(n, L1, L2, J): terms for n, L1, L2, J, terms in psi4.core.isapol_recoupling_blocks()}
    assert sorted(got) == sorted(RECOUPLING_REF)
    assert sum(len(t) for t in got.values()) == 4673
    for key, want in RECOUPLING_REF.items():
        assert [_as_tuple(t) for t in got[key]] == want, key


@pytest.mark.parametrize("n", range(6, 13))
def test_recoupling_coefficients_bit_identical(n):
    """The double CamCASP would compute, `(p d0/q d0) * sqrt(r d0/s d0)`, exactly."""
    for key, want in RECOUPLING_REF.items():
        if key[0] != n:
            continue
        got = _cxx_block(key)
        assert len(got) == len(want)
        for term, (p, q, r, s, *_) in zip(got, want):
            expected = (float(p) / float(q)) * np.sqrt(float(r) / float(s))
            assert term.coefficient == expected, (key, p, q, r, s)


def test_recoupling_absent_blocks_are_empty():
    """A block CamCASP does not tabulate is a vanishing coefficient, not an error."""
    assert _cxx_block((6, 0, 1, 1)) == []       # forbidden by parity: J must match n
    assert _cxx_block((6, 3, 3, 0)) == []       # C_6 has no octopole polarizabilities
    assert _cxx_block((6, 2, 2, 3)) == []       # tabulated as a comment, no terms
    assert _cxx_block((7, 8, 8, 16)) == []      # far above what C_7 can reach
    for key in RECOUPLING_REF:
        assert _cxx_block(key) != []


def test_recoupling_rejects_bad_order():
    for n in (5, 13, 0, -1):
        with pytest.raises(Exception):
            _cxx_block((n, 0, 0, 0))
    with pytest.raises(Exception):
        _cxx_block((6, -1, 0, 0))


def test_recoupling_conserves_the_order():
    """n = la + la' + lb + lb' + 2: each multipole rank costs one power of 1/R."""
    for (n, _, _, _), terms in RECOUPLING_REF.items():
        for (_, _, _, _, la, lap, lb, lbp, _) in terms:
            assert la + lap + lb + lbp + 2 == n
            assert 1 <= min(la, lap, lb, lbp)
            assert max(la, lap, lb, lbp) <= psi4.core.ISAPOL_MAX_POLARIZABILITY_RANK


def test_recoupling_respects_the_triangle_rule():
    for (n, L1, L2, J) in RECOUPLING_REF:
        assert abs(L1 - L2) <= J <= L1 + L2
        assert J % 2 == n % 2, "J and n have the same parity"
        assert max(L1, L2) <= psi4.core.ISAPOL_MAX_DISPERSION_RANK


def test_recoupling_imaginary_factor_follows_the_rank_parity():
    """i appears exactly when L1 + L2 + J is odd, which is what makes the sum real."""
    for (_, L1, L2, J), terms in RECOUPLING_REF.items():
        for term in terms:
            assert term[8] == (L1 + L2 + J) % 2


def test_recoupling_is_symmetric_under_exchange():
    """Swapping A and B swaps (la, la') with (lb, lb') and changes nothing else."""
    for (n, L1, L2, J), terms in RECOUPLING_REF.items():
        mirror = RECOUPLING_REF[(n, L2, L1, J)]
        swapped = [(p, q, r, s, lb, lbp, la, lap, ip)
                   for (p, q, r, s, la, lap, lb, lbp, ip) in terms]
        assert sorted(swapped) == sorted(mirror), (n, L1, L2, J)


def test_recoupling_c6_closed_forms():
    """The eight C_6 blocks, against the closed forms in SPEC.md 9.3."""
    want = {
        (0, 0, 0): 2.0,
        (0, 2, 2): -np.sqrt(2.0),
        (1, 1, 0): 1.0,
        (1, 1, 2): 2.0,
        (2, 0, 2): -np.sqrt(2.0),
        (2, 2, 0): 1.0 / 5.0,
        (2, 2, 2): 2.0 / 7.0,
        (2, 2, 4): 108.0 / 35.0,
    }
    got = {}
    for L1 in range(5):
        for L2 in range(5):
            for J in range(9):
                block = _cxx_block((6, L1, L2, J))
                if not block:
                    continue
                assert len(block) == 1, "every C_6 block is a single dipole-dipole term"
                term = block[0]
                assert (term.la, term.lap, term.lb, term.lbp, term.ipow) == (1, 1, 1, 1, 0)
                got[(L1, L2, J)] = term.coefficient
    assert got == want


def test_recoupling_reproduces_the_isotropic_c6():
    """C_6(00, 00, 0) is the ordinary isotropic C_6, for a one-term Unsold model.

    `recouple` (casimir.f90) builds alpha_00(11) with the real Clebsch-Gordan
    coefficient -1/sqrt(3) on each of the three diagonal dipole components, so
    alpha_00(11) = -sqrt(3) alpha_iso.  The block coefficient is 2 and `cp_weight`
    carries the 1/(2 pi), so C_6(00, 00, 0) must come out as (3/pi) int alpha^2 --
    the same number `test_casimir_polder_single_pole` checks the grid against.
    """
    grid = psi4.core.CasimirGrid(10, psi4.core.ISAPOL_OMEGA0)
    w = np.array([grid.omega(k) for k in range(1, 11)])
    cw = np.array([grid.cp_weight(k) for k in range(1, 11)])

    block = _cxx_block((6, 0, 0, 0))
    assert len(block) == 1

    for (aA, w1A), (aB, w1B) in [((1.38, 0.85), (1.38, 0.85)), ((10.6, 0.55), (1.38, 0.85))]:
        alpha_A = -np.sqrt(3.0) * aA * w1A**2 / (w1A**2 + w**2)
        alpha_B = -np.sqrt(3.0) * aB * w1B**2 / (w1B**2 + w**2)
        c6 = block[0].coefficient * np.dot(cw, alpha_A * alpha_B)
        exact = 1.5 * aA * aB * w1A * w1B / (w1A + w1B)
        # 1e-4 is the accuracy of the 10-point rule on this integrand, as above.
        assert np.isclose(c6, exact, rtol=1e-4, atol=0)


def test_component_indexing():
    for t, label in enumerate(COMPONENT_LABELS, start=1):
        L = int(label[0])
        assert psi4.core.isapol_component_label(t) == label
        assert psi4.core.isapol_component_rank(t) == L
        assert psi4.core.isapol_component_first(L) <= t <= psi4.core.isapol_component_last(L)
    for L in range(psi4.core.ISAPOL_MAX_DISPERSION_RANK + 1):
        first = psi4.core.isapol_component_first(L)
        last = psi4.core.isapol_component_last(L)
        assert (first, last) == (L * L + 1, (L + 1) ** 2)
        assert last - first + 1 == 2 * L + 1
    for t in (0, -1, len(COMPONENT_LABELS) + 1):
        with pytest.raises(Exception):
            psi4.core.isapol_component_rank(t)
