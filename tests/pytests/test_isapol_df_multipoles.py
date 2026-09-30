# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""The DF-centre distributed-multipole rule in closed form, gated.

The DF-centre rule is CamCASP's ``DistPolAlgorithm = DF``, which charges every
auxiliary function *wholly* to the centre it sits on.  The closed form is
checked against an independent grid oracle: the same rule integrated natively
by ``IsaPartitionedMultipoles`` with ``auxiliary_sites=[a]`` and a unit
stockholder ratio.  Only the molecular quadrature separates the two, which
``test_the_closed_form_matches_the_grid_oracle_...`` shows by refining the grid
and watching the gap fall.  Every number here is cheap and SCF-free.
"""
from types import SimpleNamespace
import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting.isapol_basis import BasisRecipe, ShellRecipe
from psi4.driver.procrouting import isapol_df_multipoles as dfm

#: The shipped Racah evaluator recovers exactly; these are the measured residuals
#: of the monomial least squares at ranks 0..3, seed 20260910.
HARMONIC_RESIDUALS = (6.661338147750939e-16, 1.3322676295501878e-15,
                      1.0658141036401503e-14, 2.3092638912203256e-14)
#: cc-pVDZ-JKFIT water AUX: 42 shells, 131 Cartesian functions, 3 sites.
NFUNCTION = 131
#: GAMINT convention check and the closed-form charge row, both on that AUX.
CONVENTION = (6.661338147750939e-16, 2.643465579626019)
CHARGE_ROW_ERROR = 7.105427357601002e-15
CHARGE_ROW_MAGNITUDE = 10.197510192046527
#: Grid-vs-closed-form maximum relative difference per rank, by (radial, spherical).
#: One rule, quadrature only: every entry falls when the grid is refined.
GRID_CONVERGENCE = {
    (50, 110): (5.263600e-05, 1.102143e-04, 1.018353e-04, 1.305225e-03),
    (75, 302): (2.27e-07, 3.64e-07, 3.00e-07, 1.37e-06),
    (99, 590): (9.95e-09, 4.62e-09, 9.00e-10, 9.24e-09),
}


@pytest.fixture(scope='module')
def declared():
    """An AUX recipe and its sites with no SCF at all: the rule needs neither."""
    core.be_quiet()
    wfn = core.Wavefunction.build(psi4.geometry('O\nH 1 1\nH 1 1 2 100\nsymmetry c1'), 'cc-pvdz')
    mol = wfn.molecule()
    centres = tuple((mol.x(i), mol.y(i), mol.z(i)) for i in range(mol.natom()))
    aux = core.BasisSet.build(mol, 'DF_BASIS_SCF', 'cc-pVDZ-JKFIT', puream=0)
    shells = []
    for j in range(aux.nshell()):
        s = aux.shell(j)
        shells.append(ShellRecipe(int(aux.shell_to_center(j)), int(s.am),
            tuple(s.exp(k) for k in range(s.nprimitive)), tuple(s.coef(k) for k in range(s.nprimitive))))
    auxiliary = BasisRecipe('cc-pVDZ-JKFIT Cartesian molecular AUX', 'Psi4 shipped basis',
                            'Cartesian', centres, tuple(shells))
    sites = tuple(SimpleNamespace(label=f'{mol.symbol(i)}{i+1}', origin=c) for i, c in enumerate(centres))
    return wfn, SimpleNamespace(auxiliary=auxiliary, sites=sites)


@pytest.fixture(scope='module')
def analytic(declared):
    _, recipe = declared
    return dfm.analytic_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3)


def _grid(wfn, radial, spherical):
    options = core.IsaGridOptions()
    options.radial_points, options.spherical_points = radial, spherical
    grid = core.IsaGrid(wfn.molecule().clone(), options)
    return np.column_stack((grid.x(), grid.y(), grid.z())), np.asarray(grid.w())


def _grid_oracle(recipe, rank, points, weights):
    """DF-centre Q on the grid: site ``a`` collocates only its own AUX centre."""
    unit = np.ones(points.shape[0]).tolist()
    declared = []
    for i, site in enumerate(recipe.sites):
        samples = core.IsaMultipoleSamples()
        samples.points, samples.weights = points.tolist(), weights.tolist()
        samples.shape, samples.shape_sum = unit, unit
        samples.auxiliary_sites = [i]
        item = core.IsaMultipoleSite()
        item.label, item.origin, item.rank = site.label, list(site.origin), rank
        item.samples = samples
        declared.append(item)
    return core.IsaPartitionedMultipoles(recipe.auxiliary.build('MolecularAux'), declared,
                                         'DF-centre grid oracle', 0.)


def test_harmonic_expansion_recovers_the_shipped_racah_evaluator():
    """The closed form's only non-elementary input, recovered and then gated.

    The Cartesian monomial coefficients of the Racah regular harmonics are not
    hard-coded: they are least-squares recovered from the shipped
    ``core.isa_regular_multipoles`` at a declared seed, so the closed form and
    the grid oracle are expanding the *same* harmonics by construction.  The
    residual is therefore a precondition, not a diagnostic.
    """
    expansion = dfm.harmonic_expansion(3)
    assert expansion.rank == 3 and expansion.seed == dfm.HARMONIC_SEED
    assert expansion.residuals == HARMONIC_RESIDUALS
    assert expansion.residual_max == max(HARMONIC_RESIDUALS) < dfm.HARMONIC_TOLERANCE
    assert [expansion.block(l).shape for l in range(4)] == [(1, 1), (3, 3), (6, 5), (10, 7)]
    assert [len(dfm.monomials(l)) for l in range(4)] == [1, 3, 6, 10]
    with pytest.raises(RuntimeError, match='monomial recovery failed'):
        dfm.harmonic_expansion(3, tolerance=1.e-18)


def test_the_cartesian_convention_is_verified_against_the_built_basis(declared):
    """That the recipe's shell list and the built basis agree function by function.

    The closed form walks ``CARTESIAN_POWERS`` in GAMINT order over the recipe's
    own shells; if the built basis ordered or normalized its columns differently
    the Q would be silently wrong, so the two are compared on random points.
    """
    _, recipe = declared
    built = recipe.auxiliary.build('MolecularAux')
    error, magnitude = dfm.verify_cartesian_convention(recipe.auxiliary, built)
    assert (error, magnitude) == CONVENTION
    assert error < dfm.CONVENTION_TOLERANCE*magnitude
    with pytest.raises(RuntimeError, match='GAMINT Cartesian convention check failed'):
        dfm.verify_cartesian_convention(recipe.auxiliary, built, tolerance=1.e-18)


def test_the_charge_row_is_checked_against_an_independent_gaussian_moment(declared, analytic):
    """``R_00 = 1``, so the rank-0 row is just ``int chi_k`` on each function's own centre.

    That derivation shares no code with the harmonic expansion, which is what
    makes it a real check on the closed form rather than a restatement of it.
    """
    _, recipe = declared
    rows = dfm.closed_form_charge_rows(recipe.auxiliary, 3)
    assert rows.shape == (3, NFUNCTION)
    assert analytic.values.shape == (3*16, NFUNCTION)
    np.testing.assert_allclose(analytic.values[[0, 16, 32]], rows, rtol=0, atol=1.e-13)
    assert analytic.diagnostics['charge_row_error'] == CHARGE_ROW_ERROR
    assert analytic.diagnostics['charge_row_magnitude'] == CHARGE_ROW_MAGNITUDE
    assert analytic.diagnostics['quadrature_defect'] == 'none; nothing is sampled'
    with pytest.raises(RuntimeError, match='charge row disagrees with the direct'):
        dfm.analytic_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3,
                                          charge_tolerance=1.e-18)


def test_the_closed_form_matches_the_grid_oracle_up_to_quadrature(declared, analytic):
    """Refine the grid and the gap falls: there is no second model here.

    Each rank is followed separately because a single maximum would hide which
    moment carries the quadrature error -- it is spread over all four rows, and
    worst in relative terms on rank 3, not concentrated in the charge row.
    """
    wfn, recipe = declared
    m, previous = 16, None
    for (radial, spherical), recorded in GRID_CONVERGENCE.items():
        points, weights = _grid(wfn, radial, spherical)
        grid = _grid_oracle(recipe, 3, points, weights)
        assert grid.negative_ratios == [0, 0, 0]
        assert grid.excluded_denominators == [0, 0, 0]
        values = grid.values.np
        measured = []
        for l in range(4):
            rows = [i*m + l*l + t for i in range(3) for t in range(2*l+1)]
            difference = float(np.abs(values[rows]-analytic.values[rows]).max())
            measured.append(difference/float(np.abs(analytic.values[rows]).max()))
        # The recorded table, to the digits it records, and strictly decreasing.
        np.testing.assert_allclose(measured, recorded, rtol=5.e-2, atol=0)
        if previous is not None:
            assert all(a < b for a, b in zip(measured, previous))
        previous = measured
    assert max(previous) < 1.e-8


@pytest.mark.parametrize('kwargs,pattern', [
    (dict(rank=0), 'rank must be an integer 1 to 4, not 0'),
    (dict(rank=5), 'rank must be an integer 1 to 4, not 5'),
    (dict(rank=True), 'rank must be an integer 1 to 4, not True'),
    (dict(rank=3.), 'rank must be an integer 1 to 4, not 3.0'),
    (dict(rank=3, nsite=2), 'needs one site per auxiliary centre'),
    (dict(rank=3, displace=True), 'centre order must match site order exactly'),
])
def test_the_rule_refuses_anything_but_an_exact_site_to_centre_correspondence(
        declared, kwargs, pattern):
    """A displaced or miscounted site would make this a different, undeclared rule.

    "Every auxiliary function wholly on its own centre" has no meaning if the
    site is not *at* the centre, so the correspondence is exact rather than
    within a tolerance.
    """
    _, recipe = declared
    sites = list(recipe.sites)[:kwargs.pop('nsite', None) or len(recipe.sites)]
    if kwargs.pop('displace', False):
        moved = core.IsaMultipoleSite()
        moved.label, moved.origin, moved.rank = sites[0].label, [0., 0., 1.], 3
        sites = [moved] + sites[1:]
    with pytest.raises(ValueError, match=pattern):
        dfm.analytic_df_centre_multipoles(recipe.auxiliary, sites, **kwargs)


def test_a_supplied_expansion_of_the_wrong_rank_is_refused(declared):
    _, recipe = declared
    with pytest.raises(ValueError, match='Supplied harmonic expansion is rank 2'):
        dfm.analytic_df_centre_multipoles(recipe.auxiliary, recipe.sites, 3,
                                          expansion=dfm.harmonic_expansion(2))
