# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Closed oracle for the Casimir-Polder step, and what the reference bounds.

`data_isapol/camcasp_local_pol_h2o_atz_wt4.json` decodes a **second** CamCASP
water case, `examples/properties/H2O`: weight type 4, prefix `H2O_aTZ`, PBE0/
aug-cc-pVTZ through DALTON, CKS propagator, a constrained-NN density fit at
`Eta = 0.0005, Lambda = 1000`, rank-4 distributed polarizabilities, and a PFIT
refinement onto the 17-variable `.pdef` model.  `oracle/read_local_pol.py` built
it out of printed output data and the case's own input declarations alone.

It is not the `tests/H2O_props` case that `camcasp_cn_pot_h2o_l2h1.json` carries,
and the two must not be conflated.  They differ in the weight type, in the SCF
back end, and -- decisively for any local tensor -- in the `H2O.axes`
declaration: bond axes here, `z global Z` there.  Their molecular isotropic C6
differs by 6.21%, which is measured in
`test_reference_family_spread_bounds_what_a_match_can_mean` and is the honest
bound on what agreement with "the" reference number can mean.

*What this file does assert quantitatively.*  The case prints local
polarizabilities and, from those same numbers, dispersion coefficients.  That
closes a loop we can walk independently: reduce the printed local tensors to
per-rank isotropic scalars, hand them to our own `isa_isotropic_dispersion` on
the declared `Quad 10 / Beta 0.5` Casimir grid, and the printed C6/C8/C10 must
come back.  Nothing of ours is fitted here and no SCF runs -- the input side is
entirely the reference's -- so this is a test of our Casimir-Polder quadrature
and dispersion assembly against reference data, and it holds to the reference's
printed precision.  The `(-1)^l sqrt(2 l + 1)` recoupling convention that
reduction rests on is not assumed either: the same file prints its own
`00(l l)` recoupled rows, and
`test_recoupling_convention_is_the_references_own_printed_convention` rebuilds
every one of them from the `.pdef` variables, sign included.

*What it does not assert.*  Nothing here compares a Psi4-computed polarizability
to the reference's.  The reference's own two routes to the molecular
polarizability -- translating its rank-4 distributed tensor, and rotating and
summing its refined local tensors -- disagree with each other by 0.26%
(`test_reference_refinement_moves_its_own_molecular_polarizability`), which is
the reference refinement's own distortion; and its refinement lattice is 500
`Seed 1` random points that we cannot reproduce without CamCASP's RNG, so its
refined tensors cannot be matched exactly by construction.  The molecular
isotropic totals are complete only at n = 6: under `L2` with rank 1 on hydrogen,
the H-O and H-H blocks structurally cannot carry C10 at all, so a molecular C10
from this model is partial and is labelled as such.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from psi4 import core

FIXTURE = json.loads(
    (Path(__file__).parent/'data_isapol/camcasp_local_pol_h2o_atz_wt4.json').read_text())

#: Racah component order of the printed local tensors, `(rank + 1) ** 2` of them.
COMPONENTS = FIXTURE['model']['components']

#: The reference's declared quadrature: `Quad 10`, `Beta 0.5`, hence `f11` in its
#: own pol-file name -- the static point plus ten dynamic nodes.
QUAD, BETA = FIXTURE['grid']['dynamic_nodes'], FIXTURE['grid']['beta']

#: Printed to seven or eight significant figures throughout, so a relative
#: agreement near 1e-7 is the tightest statement the reference data supports.
PRINTED = 1.e-6

#: Molecular isotropic C6 of the other fixture's Psi4 back-end row, i.e. the same
#: property from the same code under a different declared protocol.
OTHER_CASE_C6 = 46.617408


def local_tensors(index):
    """Rebuild the printed 9x9 local tensors of one frequency from the `.pdef`.

    The fixture carries the 17 declared variables rather than 2673 numbers; the
    decoder asserted at capture time that this rebuild reproduces the printed
    file exactly, and `reconstruction_error` records it.
    """
    model, refined = FIXTURE['model'], FIXTURE['refined_local']
    source = {tuple(to): tuple(frm) for to, frm in model['copies']}
    out = []
    for label in refined['site_labels']:
        owner = source.get((label, label), (label, label))[0]
        tensor = np.zeros((len(COMPONENTS), len(COMPONENTS)))
        for site, ca, cb, name in model['variables']:
            if site != owner:
                continue
            row, col = COMPONENTS.index(ca), COMPONENTS.index(cb)
            tensor[row, col] = tensor[col, row] = refined['values'][name][index]
        out.append(tensor)
    return np.array(out)


def isotropic_scalars(tensor, ranks=(1, 2, 3)):
    """`alpha_bar_l = tr(A_l) / (2 l + 1)` per rank; a rank the model omits is 0."""
    out = np.zeros(len(ranks))
    for j, rank in enumerate(ranks):
        block = slice(rank*rank, (rank + 1)**2)
        if block.stop <= tensor.shape[0]:
            out[j] = np.trace(tensor[block, block])/(2*rank + 1)
    return out


def reference_model(grid):
    """The reference's own refined local polarizabilities as an isotropic model."""
    refined = FIXTURE['refined_local']
    origins = {s['label']: s['xyz'] for s in FIXTURE['input']['clt']['sites']}
    frequencies = [grid.omega(k) for k in range(grid.n_freq() + 1)]
    scalars = np.array([[isotropic_scalars(t) for t in local_tensors(k)]
                        for k in range(len(frequencies))])
    sites = []
    for index, label in enumerate(refined['site_labels']):
        site = core.IsaIsotropicSite()
        site.label, site.origin, site.ranks = label, origins[label], [1, 2, 3]
        site.polarizabilities = core.Matrix.from_array(scalars[:, index])
        sites.append(site)
    return core.IsaIsotropicModel(
        frequencies, sites, 'decoded CamCASP ' + refined['source']), scalars


def test_fixture_is_the_declared_pdef_model():
    """17 free numbers per frequency: 13 on O, 4 on H1, and H2 a COPY of H1."""
    model, refined = FIXTURE['model'], FIXTURE['refined_local']
    assert model['parameter_count'] == 17
    assert sum(1 for v in model['variables'] if v[0] == 'O') == 13
    assert sum(1 for v in model['variables'] if v[0] == 'H1') == 4
    assert model['copies'] == [[['H2', 'H2'], ['H1', 'H1']]]
    assert refined['reconstruction_error'] == 0.
    assert FIXTURE['dispersion']['site_rank_limits'] == {'O': 2, 'H1': 1, 'H2': 1}
    # The rank limits show up in the reference's own output as exact zeros, which
    # is the same shape our `anchor[1:, 1:]` construction produces.
    for index in range(FIXTURE['grid']['frequency_count']):
        tensors = local_tensors(index)
        assert np.all(tensors[:, 0, :] == 0.) and np.all(tensors[:, :, 0] == 0.)
        assert np.all(tensors[1:, 4:, :] == 0.)
        assert isotropic_scalars(tensors[1])[1] == 0.


def dipole_block(tensor):
    """The Cartesian xyz dipole-dipole block of one Racah local tensor."""
    cartesian = {'10': 2, '11c': 0, '11s': 1}
    block = np.zeros((3, 3))
    for ca, ia in cartesian.items():
        for cb, ib in cartesian.items():
            block[ia, ib] = tensor[COMPONENTS.index(ca), COMPONENTS.index(cb)]
    return block


def globalized_dipole_blocks(index):
    """Each site's dipole block rotated into global axes by its declared frame."""
    frames = np.array(FIXTURE['input']['frames'])
    return [frame @ dipole_block(tensor) @ frame.T
            for tensor, frame in zip(local_tensors(index), frames)]


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
    assert FIXTURE['refined_local']['copy_local_discrepancy'] == 0.
    tensors = local_tensors(0)
    assert np.array_equal(tensors[1], tensors[2])
    # Nonzero in local axes, so the component exists to be mis-shared at all.
    assert FIXTURE['refined_local']['values']['H1_10_11c_A'][0] != 0.

    mirror = np.diag([-1., 1., 1.])
    blocks = globalized_dipole_blocks(0)
    assert np.allclose(blocks[2], mirror @ blocks[1] @ mirror, atol=1e-12, rtol=0)
    difference = np.max(np.abs(blocks[1] - blocks[2]))
    assert difference == pytest.approx(2.*abs(blocks[1][0, 2]), rel=1e-12)
    assert difference == pytest.approx(.46794962, rel=1e-7)


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
    assert len(frequencies) == FIXTURE['grid']['frequency_count'] == 11
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
    model, _ = reference_model(grid)
    weights = [grid.cp_weight(k) for k in range(grid.n_freq() + 1)]
    result = core.isa_isotropic_dispersion(model, model, weights, 12)

    types = {s['label']: s['type'] for s in FIXTURE['input']['clt']['sites']}
    labels = FIXTURE['refined_local']['site_labels']
    reference = FIXTURE['dispersion']['isotropic']
    admissible = FIXTURE['dispersion']['admissible_orders']
    compared, worst = 0, 0.
    for pair in result.pairs:
        ta, tb = types[labels[pair.site_a]], types[labels[pair.site_b]]
        # One printed block per unordered type pair; the case prints `H O`, never
        # `O H`, so the lookup is canonicalized rather than the reverse assumed absent.
        key = f'{ta} {tb}' if f'{ta} {tb}' in reference else f'{tb} {ta}'
        for coefficient in pair.coefficients:
            expected = reference.get(key, {}).get(str(coefficient.order))
            if expected is None or expected == 0.:
                # Odd orders vanish identically for this model, and an order no
                # printed block declares is not evidence either way.
                assert coefficient.order % 2 == 1 or coefficient.order not in admissible[key]
                continue
            assert not coefficient.missing_rank_pairs, (key, coefficient.order)
            relative = abs(coefficient.value - expected)/abs(expected)
            worst = max(worst, relative)
            assert relative < PRINTED, (key, coefficient.order, coefficient.value, expected)
            compared += 1
    # Six distinct reference numbers, each reached from four ordered site pairs
    # for the hydrogens; O-O C6/C8/C10, H-O C6/C8 and H-H C6.
    assert compared == 1 + 1 + 1 + 4 + 4 + 4
    assert worst < 3.e-7, worst


def test_molecular_isotropic_c6_is_complete_and_higher_orders_are_not():
    """`C6(OO) + 4 C6(HO) + 4 C6(HH)`; C8 and C10 are structurally partial here."""
    dispersion = FIXTURE['dispersion']
    assert dispersion['type_multiplicity'] == {'O O': 1, 'H O': 4, 'H H': 4}
    assert dispersion['admissible_orders'] == {'O O': [6, 8, 10], 'H O': [6, 8], 'H H': [6]}
    assert dispersion['molecular_isotropic_complete_orders'] == [6]
    assert dispersion['molecular_isotropic']['6'] == pytest.approx(43.8902696, rel=1e-12)
    # Recorded so the partial totals cannot later be quoted as molecular values.
    assert dispersion['molecular_isotropic']['10'] == dispersion['isotropic']['O O']['10']


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
    ours = FIXTURE['dispersion']['molecular_isotropic']['6']
    assert ours < OTHER_CASE_C6
    absolute = OTHER_CASE_C6 - ours
    assert absolute == pytest.approx(2.727138, rel=1e-6)
    assert absolute/ours == pytest.approx(.0621354, rel=1e-5)
    assert absolute/OTHER_CASE_C6 == pytest.approx(.0585004, rel=1e-5)


def test_reference_refinement_moves_its_own_molecular_polarizability():
    """The refinement's own distortion, measured on a partition-invariant number.

    The molecular polarizability does not depend on how the response is
    distributed, so translating the reference's rank-4 distributed tensor and
    rotating-and-summing its refined local tensors must agree.  They differ by
    0.024 (0.26%), which is the same order as the anchor movement our own
    constrained-NN refinement makes, and it bounds how exactly the refined local
    tensors can be reproduced by anything.
    """
    total = sum(globalized_dipole_blocks(0))
    recorded = FIXTURE['refined_local']
    assert np.allclose(total, np.array(recorded['molecular_alpha_static']), atol=1e-12, rtol=0)
    assert np.trace(total)/3. == pytest.approx(recorded['molecular_alpha_isotropic'][0], rel=1e-12)

    gap = FIXTURE['internal_inconsistency']
    assert gap['distributed'] == pytest.approx(9.24735727, rel=1e-8)
    assert gap['refined'] == pytest.approx(9.27158443, rel=1e-8)
    assert gap['absolute'] == pytest.approx(.0242272, rel=1e-5)
    assert .002 < gap['relative'] < .003


def test_rank4_distributed_reference_is_constrained_but_not_exactly():
    """The only rank-4 reference quantity, and how well its own constraints hold.

    75 = 3 sites x 25 components, i.e. rank 4, at full double precision.  Its
    charge-flow sum rules close to 7.7e-7 rather than to zero: that is a direct
    measurement of the reference's `Eta = 0.0005, Lambda = 1000` constrained-NN
    fit quality, and it is the level any comparison against it is limited to.
    The translation sign convention is *measured*, by which choice sends the
    C2v-forbidden xz element to zero, not assumed.
    """
    distributed = FIXTURE['distributed_static']
    assert distributed['rank'] == 4
    assert distributed['dimension'] == 75 == distributed['site_count']*25
    assert distributed['residuals']['asymmetry_maxabs'] < 1.e-10
    assert abs(distributed['residuals']['charge_charge_total']) < 1.e-11
    for key in ('sum_over_a_maxabs', 'sum_over_b_maxabs'):
        assert 1.e-8 < distributed['residuals'][key] < 1.e-6, key
    assert distributed['translation_convention'] == 'plus'
    forbidden = distributed['forbidden_xz_by_convention']
    assert forbidden['plus'] < .05*forbidden['minus']
    assert forbidden['plus'] < 1.e-7


def test_recoupling_convention_is_the_references_own_printed_convention():
    """`alpha^{00}(l l) = (-1)^l sqrt(2 l + 1) alpha_bar_l`, checked against the source.

    The closed dispersion oracle above reduces the reference's local tensors to
    per-rank isotropic scalars, which is only legitimate if our recoupling
    convention is the reference's.  That is not asserted here but measured: the
    reference prints its own `00(l l)` recoupled rows in the same
    `_casimir.out`, and rebuilding them from the `.pdef` variables reproduces
    every one at printed precision.  Sign and magnitude are both tested -- a
    convention differing by `(-1)^l` would pass on rank 2 and fail on rank 1.

    The printed rows carry the **dynamic** nodes only, 10 columns for `Quad 10`,
    so this covers indices 1..10 and says nothing about the static point; the
    static point is covered instead by the local/distributed agreement tests.
    """
    recoupled = FIXTURE['recoupled_isotropic']
    indices = recoupled['dynamic_indices']
    assert indices == list(range(1, QUAD + 1))
    assert len(indices) == FIXTURE['grid']['frequency_count'] - 1

    scalars = {label: np.array([isotropic_scalars(local_tensors(k)[index])
                                for k in indices])
               for index, label in enumerate(FIXTURE['refined_local']['site_labels'])}
    compared, worst = 0, 0.
    for label, rows in recoupled['rows'].items():
        for name, printed in rows.items():
            rank = int(name[3])
            assert name == f'00({rank}{rank})'
            ours = ((-1.)**rank)*np.sqrt(2*rank + 1)*scalars[label][:, rank - 1]
            assert np.sign(ours) @ np.sign(printed) == len(printed), (label, name)
            for k, value in enumerate(printed):
                worst = max(worst, abs(ours[k]/value - 1.))
                compared += 1
    # O rank 1 and 2, and rank 1 on each hydrogen, over 10 dynamic nodes each.
    assert compared == 4*QUAD == 40
    assert worst < 1.e-5, worst
    # Rank 1 on oxygen is negative in the recoupled form and positive as a
    # scalar; that sign is the whole content of the `(-1)^l` factor.
    assert recoupled['rows']['O']['00(11)'][0] < 0. < recoupled['rows']['O']['00(22)'][0]
