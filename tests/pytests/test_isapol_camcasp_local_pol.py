# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Closed oracle for the Casimir-Polder step, on committed literals.

The reference case is CamCASP `examples/properties/H2O`: weight type 4, prefix
`H2O_aTZ`, PBE0/aug-cc-pVTZ through DALTON, CKS propagator, a constrained-NN
density fit at `Eta = 0.0005, Lambda = 1000`, rank-4 distributed polarizabilities,
and a PFIT refinement onto the 17-variable `.pdef` model.  It is *not* the
`tests/H2O_props` case of `test_isapol_primary_casimir_reference.py`, and the
two must not be conflated: they differ in the weight type, in the SCF back end, and
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

*What is not here.*  The reference-only measurements once asserted beside
this oracle (its printed `(-1)^l sqrt(2l + 1)` recoupled rows, the bond-frame
COPY rule, the 7.7e-7 closure of its rank-4 sum rules) execute no Psi4 code;
they are recorded in `data_isapol/README.md`.  Nothing here compares a
Psi4-computed polarizability to the reference's: its own two routes to the
molecular polarizability disagree by 0.26%.  The molecular isotropic total
is complete only at n = 6.
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
#: `C6(OO) + 4 C6(HO) + 4 C6(HH)` of the printed blocks.
MOLECULAR_C6 = 43.890269599999996


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
    # The molecular C6 is the sum over all nine ordered site pairs of our output.
    c6 = [p.coefficients[0] for p in result.pairs]
    assert all(x.order == 6 and x.complete for x in c6) and len(c6) == 9
    assert abs(sum(x.value for x in c6)/MOLECULAR_C6 - 1.) < PRINTED
