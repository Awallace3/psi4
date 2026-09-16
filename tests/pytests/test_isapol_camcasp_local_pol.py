# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Closed oracle for the Casimir-Polder step, on committed literals.

The reference case is CamCASP `examples/properties/H2O`: weight type 4, prefix
`H2O_aTZ`, PBE0/aug-cc-pVTZ through DALTON, CKS propagator, a constrained-NN
density fit at `Eta = 0.0005, Lambda = 1000`, rank-4 distributed polarizabilities,
and a PFIT refinement onto the 17-variable `.pdef` model.  It is *not* the
`tests/H2O_props` case that `camcasp_cn_pot_h2o_l2h1.json` carries, and the two
must not be conflated: they differ in the weight type, in the SCF back end, and
-- decisively for any local tensor -- in the `H2O.axes` declaration (bond axes
here, `z global Z` there).

*Why literals suffice for the quantitative part.*  The step under test reduces
the reference's refined local tensors to per-rank isotropic scalars and hands
those to our own `isa_isotropic_dispersion`.  The reduction throws away
everything except `alpha_bar_l = tr(A_l)/(2l + 1)`, so the entire input to our
code is the 33 numbers in `SCALARS` -- three sites by three ranks by eleven
frequencies, with the model's zero ranks omitted.  Committing those instead of
the 1,271-line fixture loses no coverage of ours at all: the printed C6/C8/C10
below must still come back out of our quadrature and pair assembly.

*What only the fixture can do*, and so lives in
`agent_scratch/pytests/test_isapol_camcasp_local_pol_oracle.py`: rebuilding all
2,673 printed tensor entries from the 17 `.pdef` variables, the full structural
sweep over them, and the guard test that re-derives every literal here from the
fixture so a regenerated capture cannot leave stale numbers committed.

*What no version of this file asserts.*  Nothing here compares a Psi4-computed
polarizability to the reference's.  The reference's own two routes to the
molecular polarizability disagree by 0.26% (`INTERNAL_INCONSISTENCY`), and its
refinement lattice is 500 `Seed 1` random points we cannot reproduce without
CamCASP's RNG, so its refined tensors cannot be matched exactly by construction.
The molecular isotropic total is complete only at n = 6.
"""
import numpy as np
import pytest
from psi4 import core

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

#: The reference's declared quadrature: `Quad 10`, `Beta 0.5`, hence `f11` in its
#: own pol-file name -- the static point plus ten dynamic nodes.
QUAD, BETA = 10, 0.5

#: Printed to seven or eight significant figures throughout, so a relative
#: agreement near 1e-7 is the tightest statement the reference data supports.
PRINTED = 1.e-6

SITES = [('O', 'O', [0.0, 0.0, 0.0]),
         ('H1', 'H', [-1.45365196, 0.0, -1.12168732]),
         ('H2', 'H', [1.45365196, 0.0, -1.12168732])]

#: `alpha_bar_l = tr(A_l)/(2l + 1)` of the reference's refined local tensors, at
#: all eleven grid frequencies, index 0 the static point.  This is the whole input
#: our dispersion assembly sees.  The `.pdef` gives oxygen rank 2 and each hydrogen
#: rank 1, so O rank 3 and H ranks 2, 3 are identically zero and are not listed;
#: H2 is a declared `COPY` of H1 in *local* axes, so it repeats H1 exactly.
SCALARS = {
    ('O', 1): [6.509418566666667, 6.508479433333334, 6.481486966666666, 6.3229547,
               5.812577133333334, 4.7269188, 3.1804547, 1.6868987666666666,
               0.6462111066666667, 0.13141871666666669, 0.005380400066666667],
    ('O', 2): [23.2918814, 23.289271799999998, 23.2140196, 22.763072799999996,
               21.2340924, 17.827700800000002, 12.979121999999998, 7.8504881,
               3.2668528, 0.649356594, 0.0236903486],
    ('H', 1): [1.3810829333333334, 1.3808881, 1.3753015333333334, 1.3429815999999999,
               1.2433183666666665, 1.03868023, 0.71782938, 0.34710575666666665,
               0.09400381066666667, 0.011749210666666668, 0.0003682661633333333],
}

#: The reference's printed isotropic dispersion coefficients, one block per
#: unordered site-type pair.  Odd orders vanish identically for this model; an
#: order a block does not declare is listed in `ADMISSIBLE` and is not evidence
#: either way.
REFERENCE_CN = {'O O': {6: 21.59463, 8: 422.2789, 10: 3894.995},
                'H O': {6: 4.590924, 8: 44.65951},
                'H H': {6: 0.9829859}}
ADMISSIBLE = {'O O': [6, 8, 10], 'H O': [6, 8], 'H H': [6]}
#: Ordered pairs of sites that map onto each printed type-pair block.
MULTIPLICITY = {'O O': 1, 'H O': 4, 'H H': 4}
MOLECULAR_C6 = 43.890269599999996
#: Only n = 6 is a complete molecular total: under `L2` with rank 1 on hydrogen
#: the H-O and H-H blocks structurally cannot carry C10 at all.
COMPLETE_ORDERS = [6]

#: Molecular isotropic C6 of the *other* fixture's Psi4 back-end row, i.e. the same
#: property from the same code under a different declared protocol.
OTHER_CASE_C6 = 46.617408

#: The reference's own `00(l l)` recoupled rows from the same `_casimir.out`,
#: dynamic nodes 1..10 only.  Rebuilding these from `SCALARS` tests that our
#: `(-1)^l sqrt(2l + 1)` convention is the reference's, sign included.
RECOUPLED = {
    ('O', 1): [-11.273, -11.2263, -10.9517, -10.0677, -8.18726, -5.50871, -2.92179,
               -1.11927, -0.227624, -0.00931913],
    ('O', 2): [52.0764, 51.9081, 50.8998, 47.4809, 39.864, 29.0222, 17.5542, 7.3049,
               1.45201, 0.0529732],
    ('H1', 1): [-2.39177, -2.38209, -2.32611, -2.15349, -1.79905, -1.24332, -0.601205,
                -0.162819, -0.0203502, -0.000637856],
    ('H2', 1): [-2.39177, -2.38209, -2.32611, -2.15349, -1.79905, -1.24332, -0.601205,
                -0.162819, -0.0203502, -0.000637856],
}

#: The two declared bond frames, local-to-global columns.  Oxygen is the identity.
FRAMES = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
          [[-0.610905425288665, 0.0, -0.7917035817481662],
           [0.0, 0.9999999999999999, 0.0],
           [0.7917035817481661, 0.0, -0.610905425288665]],
          [[0.610905425288665, 0.0, 0.7917035817481662],
           [0.0, -0.9999999999999999, 0.0],
           [0.7917035817481661, 0.0, -0.610905425288665]]]

#: The static H1 dipole-dipole block in *local* axes, xyz ordered, rebuilt at
#: capture time from the four declared H1 variables.  H2's is bit-identical.
H1_DIPOLE_BLOCK = [[1.2967801, 0.0, -0.26821221],
                   [0.0, 1.206554, 0.0],
                   [-0.26821221, 0.0, 1.6399147]]

#: The reference's two routes to its own static molecular isotropic polarizability:
#: translating its rank-4 distributed tensor, versus rotating and summing its
#: refined local tensors.  The gap is the refinement's own distortion.
INTERNAL_INCONSISTENCY = {'distributed': 9.24735726744245, 'refined': 9.271584433333333,
                          'absolute': 0.02422716589088303, 'relative': 0.002619901577305833}

#: Residuals of the one rank-4 reference quantity, `H2O_aTZ_NL4_static.pol`, at
#: 3 sites x 25 components.  The charge-flow sum rules close to 7.7e-7 rather than
#: to zero: a direct measurement of the reference's constrained-NN fit quality, and
#: the level any comparison against it is limited to.
DISTRIBUTED_RESIDUALS = {'asymmetry_maxabs': 5.807088143683359e-11,
                         'charge_charge_total': -3.11178860457062e-12,
                         'sum_over_a_maxabs': 7.675323379086052e-07,
                         'sum_over_b_maxabs': 7.675286010089266e-07}
#: Which translation sign convention sends the C2v-forbidden xz element to zero.
#: Measured at capture time, not assumed.
FORBIDDEN_XZ = {'plus': 7.03403762042365e-08, 'minus': 2.0687942772212864e-06}


def scalars(label):
    """The (11, 3) per-rank isotropic table of one site, zeros where the model has none."""
    key = 'O' if label == 'O' else 'H'
    return np.array([[SCALARS.get((key, rank), [0.0]*(QUAD + 1))[k] for rank in (1, 2, 3)]
                     for k in range(QUAD + 1)])


def reference_model(grid):
    """The reference's refined local polarizabilities as an isotropic model."""
    sites = []
    for label, _type, origin in SITES:
        site = core.IsaIsotropicSite()
        site.label, site.origin, site.ranks = label, origin, [1, 2, 3]
        site.polarizabilities = core.Matrix.from_array(scalars(label))
        sites.append(site)
    return core.IsaIsotropicModel([grid.omega(k) for k in range(grid.n_freq() + 1)],
                                  sites, 'committed literals from CamCASP H2O_aTZ_ref_wt4_L2_0f10.pol')


def test_casimir_grid_matches_the_declared_quadrature():
    """`CasimirGrid(10, .5)` is the reference's `Quad 10 / Beta 0.5`, all 11 nodes.

    `n_freq()` is the Gauss-Legendre *order*; the grid carries `n_freq + 1`
    frequencies with index 0 the static point.  The reference names its own file
    `..._f11_NL4`, so 11 is its count too.  The nodes come in reciprocal pairs
    about `omega0` by construction of the `omega0 (1 -+ t)/(1 +- t)` map, and that
    is checked here rather than the tabulated numbers being restated.
    """
    grid = core.CasimirGrid(QUAD, BETA)
    assert grid.n_freq() == QUAD == 10
    assert grid.omega0() == BETA
    frequencies = [grid.omega(k) for k in range(grid.n_freq() + 1)]
    assert len(frequencies) == QUAD + 1 == 11
    assert frequencies[0] == 0. and grid.cp_weight(0) == 0.
    for k in range(1, QUAD//2 + 1):
        assert frequencies[k]*frequencies[QUAD - k + 1] == pytest.approx(BETA**2, rel=1e-14)
        assert grid.cp_weight(k) > 0. and grid.cp_weight(QUAD - k + 1) > 0.


def test_isotropic_dispersion_reproduces_the_reference_rows():
    """The closed oracle: reference local polarizabilities in, its own C_n out.

    Both sides of this comparison are the reference's, so what is under test is
    ours: the Casimir-Polder weights, the `(-1)^l` recoupling convention behind
    the isotropic reduction, the order bookkeeping `n = 2(l_a + l_b + 1)`, and the
    pair assembly.  Site labels are mapped to the reference's printed *type*
    pairs, since one printed block stands for every ordered pair of those types.
    """
    grid = core.CasimirGrid(QUAD, BETA)
    model = reference_model(grid)
    weights = [grid.cp_weight(k) for k in range(grid.n_freq() + 1)]
    result = core.isa_isotropic_dispersion(model, model, weights, 12)

    types = {label: kind for label, kind, _ in SITES}
    labels = [label for label, _, _ in SITES]
    compared, worst = 0, 0.
    for pair in result.pairs:
        ta, tb = types[labels[pair.site_a]], types[labels[pair.site_b]]
        # One printed block per unordered type pair; the case prints `H O`, never
        # `O H`, so the lookup is canonicalized rather than the reverse assumed absent.
        key = f'{ta} {tb}' if f'{ta} {tb}' in REFERENCE_CN else f'{tb} {ta}'
        for coefficient in pair.coefficients:
            expected = REFERENCE_CN[key].get(coefficient.order)
            if expected is None:
                # Odd orders vanish identically for this model, and an order no
                # printed block declares is not evidence either way.
                assert coefficient.order % 2 == 1 or coefficient.order not in ADMISSIBLE[key]
                continue
            assert not coefficient.missing_rank_pairs, (key, coefficient.order)
            relative = abs(coefficient.value - expected)/abs(expected)
            worst = max(worst, relative)
            assert relative < PRINTED, (key, coefficient.order, coefficient.value, expected)
            compared += 1
    # Six distinct reference numbers, each reached from one or four ordered site
    # pairs; O-O C6/C8/C10, H-O C6/C8 and H-H C6.
    assert compared == sum(len(orders)*MULTIPLICITY[key] for key, orders in REFERENCE_CN.items()) == 15
    assert worst < 3.e-7, worst


def test_molecular_isotropic_c6_is_complete_and_higher_orders_are_not():
    """`C6(OO) + 4 C6(HO) + 4 C6(HH)`; C8 and C10 are structurally partial here."""
    assert COMPLETE_ORDERS == [6]
    total = sum(MULTIPLICITY[key]*orders[6] for key, orders in REFERENCE_CN.items())
    assert total == pytest.approx(MOLECULAR_C6, rel=1e-12)
    # Only the O-O block admits n = 10, so a "molecular" C10 is just that block.
    assert [key for key, orders in ADMISSIBLE.items() if 10 in orders] == ['O O']
    assert MULTIPLICITY['O O'] == 1


def test_reference_family_spread_bounds_what_a_match_can_mean():
    """Two reference protocols, same property, 6% apart.

    `tests/H2O_props` weight 3 with the Psi4 back end gives 46.617408; this case
    -- weight 4, DALTON, bond axes, `Eta = 0.0005` -- gives 43.890270.  Neither is
    wrong; the gap is the reference pipeline's own sensitivity to declarations,
    and it is larger than any tolerance a native chain could be held to against a
    single row of it.

    Both normalizations of that gap are asserted, because a one-sided percentage
    invites being requoted against the wrong denominator: 2.727138 absolute is
    6.21% of this case's total and 5.85% of the other's.
    """
    assert MOLECULAR_C6 < OTHER_CASE_C6
    absolute = OTHER_CASE_C6 - MOLECULAR_C6
    assert absolute == pytest.approx(2.727138, rel=1e-6)
    assert absolute/MOLECULAR_C6 == pytest.approx(.0621354, rel=1e-5)
    assert absolute/OTHER_CASE_C6 == pytest.approx(.0585004, rel=1e-5)


def test_the_copy_holds_in_local_axes_and_not_in_global_ones():
    """Independent confirmation, from the reference's own output, of the frame rule.

    `H2 H2 COPY H1 H1` is a statement about *local* tensors: the two printed
    blocks are bit-identical, and the fit therefore has one hydrogen variable set
    rather than two.  In global axes they are not identical -- the two declared
    bond frames differ by the molecular plane's x mirror, so the globalized blocks
    are related by `diag(-1, 1, 1)` and differ in their xz element by twice it.
    A refinement that pinned the two hydrogens together on *global* anchors would
    therefore be left with that difference as an irreducible residual, which is
    what `RefinementModel.copy_anchor_discrepancy` measures and why the
    constrained-NN refinement is run on declared frames.
    """
    local = np.array(H1_DIPOLE_BLOCK)
    # Nonzero in local axes, so the component exists to be mis-shared at all.
    assert local[0, 2] != 0.

    frames = np.array(FRAMES)
    blocks = [frame @ local @ frame.T for frame in frames[1:]]
    mirror = np.diag([-1., 1., 1.])
    assert np.allclose(blocks[1], mirror @ blocks[0] @ mirror, atol=1e-12, rtol=0)
    difference = np.max(np.abs(blocks[0] - blocks[1]))
    assert difference == pytest.approx(2.*abs(blocks[0][0, 2]), rel=1e-12)
    assert difference == pytest.approx(.46794962, rel=1e-7)
    # Both declared frames are *proper* rotations, so the global x mirror cannot be
    # the whole story: it is paired with a local y flip, `F2 = M F1 diag(1, -1, 1)`,
    # exactly.  That local flip is invisible above because the local dipole block
    # decouples y, which is why the globalized blocks are related by M alone.
    for frame in frames:
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-15, rtol=0)
        assert np.linalg.det(frame) == pytest.approx(1., abs=1e-15)
    np.testing.assert_array_equal(frames[2], mirror @ frames[1] @ np.diag([1., -1., 1.]))
    assert local[1, 0] == local[1, 2] == 0.


def test_reference_refinement_moves_its_own_molecular_polarizability():
    """The refinement's own distortion, measured on a partition-invariant number.

    The molecular polarizability does not depend on how the response is
    distributed, so translating the reference's rank-4 distributed tensor and
    rotating-and-summing its refined local tensors must agree.  They differ by
    0.024 (0.26%), which is the same order as the anchor movement our own
    constrained-NN refinement makes, and it bounds how exactly the refined local
    tensors can be reproduced by anything.
    """
    gap = INTERNAL_INCONSISTENCY
    assert gap['refined'] - gap['distributed'] == pytest.approx(gap['absolute'], rel=1e-12)
    # Normalized on the distributed route, which is the partition-invariant one.
    assert gap['absolute']/gap['distributed'] == pytest.approx(gap['relative'], rel=1e-12)
    assert .002 < gap['relative'] < .003
    # The refined route is a rank-1 trace of the same local tensors this file
    # reduces, so its isotropic value must be the rank-1 scalar sum at omega = 0.
    assert sum(scalars(label)[0, 0] for label, _, _ in SITES) == pytest.approx(gap['refined'], rel=1e-14)


def test_rank4_distributed_reference_is_constrained_but_not_exactly():
    """How well the one rank-4 reference quantity's own constraints hold.

    Its charge-flow sum rules close to 7.7e-7 rather than to zero: that is a
    direct measurement of the reference's `Eta = 0.0005, Lambda = 1000`
    constrained-NN fit quality, and it is the level any comparison against it is
    limited to.  The translation sign convention is *measured*, by which choice
    sends the C2v-forbidden xz element to zero, not assumed.
    """
    assert DISTRIBUTED_RESIDUALS['asymmetry_maxabs'] < 1.e-10
    assert abs(DISTRIBUTED_RESIDUALS['charge_charge_total']) < 1.e-11
    for key in ('sum_over_a_maxabs', 'sum_over_b_maxabs'):
        assert 1.e-8 < DISTRIBUTED_RESIDUALS[key] < 1.e-6, key
    assert FORBIDDEN_XZ['plus'] < .05*FORBIDDEN_XZ['minus']
    assert FORBIDDEN_XZ['plus'] < 1.e-7


def test_recoupling_convention_is_the_references_own_printed_convention():
    """`alpha^{00}(l l) = (-1)^l sqrt(2 l + 1) alpha_bar_l`, checked against the source.

    The closed dispersion oracle above reduces the reference's local tensors to
    per-rank isotropic scalars, which is only legitimate if our recoupling
    convention is the reference's.  That is not asserted here but measured: the
    reference prints its own `00(l l)` recoupled rows in the same
    `_casimir.out`, and rebuilding them from the scalars reproduces every one at
    printed precision.  Sign and magnitude are both tested -- a convention
    differing by `(-1)^l` would pass on rank 2 and fail on rank 1.

    The printed rows carry the **dynamic** nodes only, 10 columns for `Quad 10`,
    so this covers indices 1..10 and says nothing about the static point; the
    static point is covered instead by the local/distributed agreement test.
    """
    compared, worst = 0, 0.
    for (label, rank), printed in RECOUPLED.items():
        assert len(printed) == QUAD
        ours = ((-1.)**rank)*np.sqrt(2*rank + 1)*scalars(label)[1:, rank - 1]
        assert np.sign(ours) @ np.sign(printed) == len(printed), (label, rank)
        for k, value in enumerate(printed):
            worst = max(worst, abs(ours[k]/value - 1.))
            compared += 1
    # O rank 1 and 2, and rank 1 on each hydrogen, over 10 dynamic nodes each.
    assert compared == 4*QUAD == 40
    assert worst < 1.e-5, worst
    # Rank 1 on oxygen is negative in the recoupled form and positive as a
    # scalar; that sign is the whole content of the `(-1)^l` factor.
    assert RECOUPLED[('O', 1)][0] < 0. < RECOUPLED[('O', 2)][0]
