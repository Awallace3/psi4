# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""End-to-end comparison against the reference localized `Cn` potential.

`data_isapol/camcasp_cn_pot_h2o_l2h1.json` is the decoded reference result for
water at PBE0/aVTZ with a GRAC-corrected asymptote, localized by LW under the
`Limit 2` / `H-Limit 1` model, in all three of its shipped SCF back-end rows.
`oracle/read_cn_pot.py` built it out of printed output data alone.

Two things are compared here and they are reported separately.

*The isotropic track.*  The reference's `00 00 0` row is the isotropic site-site
`C_n`, which is what a native isotropic dispersion chain produces too, so it is
compared quantitatively.  The unit of comparison is deliberately the **molecular
total**, because that is the only number in the comparison that is partition
invariant; the site-resolved splits are reported as well, and they disagree
badly, which is the point of reporting them.

*The recoupled track.*  Every other reference row is a recoupled Stone component
`C_n^{l_a k_a, l_b k_b, j}`.  The native anisotropic product is
`orientation_resolved_scalars_not_recoupled_components`, a different
representation, so those 377 nonzero rows are an explicitly **uncompared** track
and the fixture does not even carry their values.  That is asserted rather than
left implicit, so the coverage claim cannot quietly drift.

Nothing here asserts a tolerance against the reference.  Our chain cannot yet
declare the reference's own per-site rank limits -- the LW workspace is uniform,
so `{O: 2, H: 1}` is rejected before any compute -- and it has no PFIT
refinement, so it is a *differently declared model*.  What the long test does
assert is where our totals sit relative to the reference family's own internal
spread, which is a bound the model mismatch cannot explain away.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_native as n
from psi4.driver.procrouting import isapol_oeprop as o
from psi4.driver.procrouting.isapol_lw import AnisotropicDispersion
from psi4.driver.procrouting.isapol_response_preflight import estimate_response_work

FIXTURE = json.loads((Path(__file__).parent/'data_isapol/camcasp_cn_pot_h2o_l2h1.json').read_text())

#: The reference input's own GRAC shift: I.P. 12.62063 eV over CamCASP's own
#: 27.21136 eV/Eh, minus HOMO 0.3989 Eh.  Only the `psi4` back-end row declares it.
SHIFT = .06490004527520865
MATCHED_AUX = 'aug-cc-pVTZ-JKFIT'
TRACED = 1.e3  # the traced constrained-NN penalty

#: `n = 2*(l_a + l_b + 1)` over the declared rank limits, `Limit 2` and `H-Limit 1`.
ADMISSIBLE = {'O O': (6, 8, 10), 'H O': (6, 8), 'H H': (6,)}


def _multiplicity(row):
    """Ordered site pairs each printed type-pair block stands for.

    One block per site TYPE pair; the reference model declares one variable set
    per type and `COPY`s the rest, so the O-H block owns four ordered pairs.
    """
    counts = {}
    for site in row['input']['sites']:
        counts[site['type']] = counts.get(site['type'], 0) + 1
    out = {}
    for pair in row['isotropic']:
        a, b = pair.split()
        out[pair] = counts[a]*counts[b]*(1 if a == b else 2)
    return out


def reference_total(row, order):
    """The reference's molecular isotropic `C_n`, summed over ordered site pairs."""
    multiplicity = _multiplicity(row)
    return sum(multiplicity[pair]*values.get(str(order), 0.)
               for pair, values in row['isotropic'].items())


def native_by_type(properties):
    """Native isotropic coefficients folded onto reference type pairs, per pair.

    Folded rather than summed so the result is directly the printed reference
    quantity: `{('H', 'O'): {6: <per ordered pair>}}`.
    """
    totals, counts = {}, {}
    for pair in properties.dispersion.pairs:
        key = tuple(sorted(''.join(c for c in label if c.isalpha())
                           for label in (pair.label_a, pair.label_b)))
        for coefficient in pair.coefficients:
            totals.setdefault(key, {}).setdefault(coefficient.order, 0.)
            totals[key][coefficient.order] += float(coefficient.value)
            counts[key, coefficient.order] = counts.get((key, coefficient.order), 0) + 1
    return {' '.join(key): {order: value/counts[key, order] for order, value in orders.items()}
            for key, orders in totals.items()}


def native_total(properties, order):
    return sum(float(c.value) for pair in properties.dispersion.pairs
               for c in pair.coefficients if c.order == order)


def test_three_backends_declare_one_localization_model_and_one_axis_system():
    """Everything that defines the localization is byte-identical across the rows."""
    assert FIXTURE['backends'].keys() == {'dalton', 'nwchem', 'psi4'}
    assert 'MIT License' in FIXTURE['notice'] and 'Anthony Stone' in FIXTURE['notice']
    shared = FIXTURE['shared_header']
    assert shared == {'Axes file': 'H2O.axes', 'Pol file format': 'NEW', 'Limit': '2',
                      'WSM-Limit': '2', 'H-Limit': '1', 'Isotropic?': 'False',
                      'Model file': 'H2O.pdef', 'Pol Cutoff': '0.0001',
                      'Loc algorithm': 'LW', 'Weight': '3', 'Weight coeff': '0.001',
                      'SVD threshold': '0.0', 'NoRefine?': 'False'}
    digests = set()
    for row in FIXTURE['backends'].values():
        assert row['header'] == shared  # nothing at all differs in the header
        assert row['input']['basis'] == 'aVTZ'
        digests.add(row['axes_sha256'])
    assert len(digests) == 1
    # Two local frames, O on the global axes; this is the frame the recoupled
    # components below are expressed in, and it is why they are not comparable.
    assert FIXTURE['axes'].split('\n')[1:3] == ['  H1  z global Z x from H2 to H1',
                                                '  H2  z global Z x from H1 to H2']


def test_the_family_spread_is_not_pure_scf_code_noise():
    """The three rows declare three different asymptotic corrections.

    Read carelessly this is "the same calculation in three SCF codes", and the
    spread across it would then be reference noise.  It is not: the `.clt` files
    differ in `SCFcode` *and* `HOMO`, so at the shared I.P. they are three
    different GRAC shifts, and DALTON's is twice ours.  Only the `psi4` row is
    the input our own chain reproduces, so it is the only defensible yardstick,
    and the nwchem/psi4 pair bounds how much a shift of this size is worth.
    """
    shifts = {name: row['input']['grac_shift'] for name, row in FIXTURE['backends'].items()}
    assert len({row['input']['ip_ev'] for row in FIXTURE['backends'].values()}) == 1
    assert len(set(shifts.values())) == 3
    assert shifts['psi4'] == SHIFT
    assert shifts['dalton'] == pytest.approx(2.*shifts['psi4'], rel=2e-2)
    assert shifts['nwchem'] == pytest.approx(shifts['psi4'], rel=2e-2)
    c6 = {name: reference_total(row, 6) for name, row in FIXTURE['backends'].items()}
    assert c6['psi4'] == pytest.approx(46.617408, abs=1e-6)
    assert c6['nwchem'] == pytest.approx(45.730744, abs=1e-6)
    assert c6['dalton'] == pytest.approx(45.050294, abs=1e-6)
    # The pair that differs only slightly in shift: 1.9% on the molecular C6.
    assert .015 < c6['psi4']/c6['nwchem'] - 1. < .022
    assert .03 < c6['psi4']/c6['dalton'] - 1. < .05  # the doubled shift, for scale


def test_reference_prints_exactly_the_admissible_isotropic_orders():
    """`n = 2(l_a + l_b + 1)` under `Limit 2` and `H-Limit 1`, and nothing else.

    This is the truncation our uniform-rank chain cannot yet declare: it admits
    rank 3 on every site, hence `{6, 8, 10, 12}` on every pair, where the
    reference admits `{6, 8}` on O-H and `{6}` alone on H-H.
    """
    for row in FIXTURE['backends'].values():
        for pair, values in row['isotropic'].items():
            nonzero = tuple(sorted(int(k) for k, v in values.items() if v != 0.))
            assert nonzero == ADMISSIBLE[pair], (pair, values)
            zero = sorted(int(k) for k, v in values.items() if v == 0.)
            assert all(order % 2 for order in zero), (pair, zero)
            assert zero == row['census'][pair]['isotropic_orders_printed_zero']
            assert '12' not in values  # nothing above C10 is printed at this limit


def test_the_recoupled_track_is_a_labelled_census_and_deliberately_uncompared():
    """377 nonzero recoupled rows, counted and not compared.

    The reason is representational, not a tolerance: what we produce for the
    anisotropy is orientation-resolved scalars, not the reference's recoupled
    components, so the fixture withholds the values on purpose.
    """
    assert AnisotropicDispersion.kind == 'orientation_resolved_scalars_not_recoupled_components'
    for row in FIXTURE['backends'].values():
        census = row['census']
        assert {p: c['recoupled_rows_nonzero'] for p, c in census.items()} == \
               {'O O': 258, 'H O': 86, 'H H': 33}
        assert sum(c['recoupled_rows_nonzero'] for c in census.values()) == 377
        for entry in census.values():
            assert entry['recoupled_rows'] == entry['recoupled_rows_nonzero']
            assert entry['printed_zero'] < entry['printed_entries']
            assert entry['indices'][0] == '00'
        # Unlike the isotropic row, the recoupled rows do NOT vanish at the odd
        # orders wherever the model carries them, so the uncompared track is not
        # a subset of the compared one and cannot be dismissed as zeros.
        assert census['O O']['nonzero_rows_per_order'] == \
               {'6': 16, '7': 46, '8': 95, '9': 118, '10': 103}
        assert census['H O']['nonzero_rows_per_order'] == {'6': 23, '7': 33, '8': 53}
        assert census['H H']['nonzero_rows_per_order'] == {'6': 33}
        assert max(census['O O']['j_values']) == 8 and census['H H']['j_values'] == [0, 2, 4]
    # The census is a census: no value of any recoupled component is carried, so
    # no test downstream can start quoting one against an orientation-resolved scalar.
    for row in FIXTURE['backends'].values():
        assert 'rows' not in row and 'anisotropic' not in row
        for entry in row['census'].values():
            assert set(entry) == {'indices', 'isotropic_orders_printed_zero', 'j_values',
                                  'nonzero_rows_per_order', 'orders', 'printed_entries',
                                  'printed_zero', 'recoupled_rows', 'recoupled_rows_nonzero'}
    # And it is identical in all three rows: the SCF back-end moves values, never
    # which components the declared model has.
    assert len({json.dumps(row['census'], sort_keys=True)
                for row in FIXTURE['backends'].values()}) == 1


@pytest.fixture(scope='module')
def chains():
    """The reference input's own state, in the one AUX at which it is accepted.

    PBE0/aug-cc-pVTZ with the reference GRAC shift, matched Cartesian AUX (see
    `test_isapol_matched_auxiliary`), and the two response routes worth
    comparing: the fit-free `direct_ov` and the traced constrained-NN penalty.
    The per-site rank limits of the reference model are NOT declarable here, so
    both rows are uniform rank 3.
    """
    core.be_quiet()
    threads = core.get_num_threads()
    core.set_num_threads(min(8, os.cpu_count() or 1))
    water = psi4.geometry('0 1\nO 0. 0. 0.\nH -1.45365196 0. -1.12168732\n'
                          'H 1.45365196 0. -1.12168732\nunits bohr\nsymmetry c1\nno_com\nno_reorient\n')
    psi4.set_options({'basis': 'aug-cc-pvtz', 'reference': 'rks', 'scf_type': 'pk',
                      'e_convergence': 1e-10, 'd_convergence': 1e-10,
                      'dft_radial_points': 99, 'dft_spherical_points': 590, 'dft_alpha': .25,
                      'dft_grac_shift': SHIFT, 'dft_grac_alpha': .5, 'dft_grac_beta': 40.,
                      'dft_grac_x_func': 'XC_GGA_X_LB', 'dft_grac_c_func': 'XC_LDA_C_VWN'})
    try:
        _, wfn = psi4.energy('pbe0', molecule=water, return_wfn=True)
        options = core.IsaGridOptions()
        options.radial_points, options.spherical_points = 99, 590
        grid = core.IsaGrid(wfn.molecule().clone(), options)
        response_grid = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
        quadrature = n.Quadrature.from_casimir(core.CasimirGrid(10, .5))
        estimate_response_work(wfn.basisset().nbf(), wfn.nmo(), wfn.nalpha(),
                               response_grid.shape[0], algorithm='shared_sweep').require_pass()
        recipe = o.generated_recipe(wfn, aux_basis=MATCHED_AUX)
        shared = dict(bonds=((1, 0), (2, 0)), frames=None, caller_converged=True,
                      kernel='alda_slater_pw92', exact_exchange=.25, local_scale=.75,
                      response_grid=response_grid, frequencies=quadrature.frequencies,
                      quadrature=quadrature, pair_self=True, response_algorithm='shared_sweep',
                      scf_correction='FIXED_GRAC', expected_grac_shift=SHIFT)
        out = {'direct_ov': n.native_properties(wfn, recipe, response_basis='direct_ov', **shared)}
        out['lambda1000'] = n.native_properties(
            wfn, recipe, response_basis='fitted_auxiliary', ov_charge_penalty=TRACED,
            response_context=out['direct_ov'].context, **shared)
        for properties in out.values():
            assert not properties.failures, [f.message for f in properties.failures]
        return out
    finally:
        core.set_num_threads(threads)


@pytest.mark.long
def test_molecular_isotropic_c6_lands_inside_the_reference_family_spread(chains):
    """The one partition-invariant number agrees to better than the family's own spread.

    Both routes land inside the 1.9% the reference itself moves between the two
    rows whose GRAC shifts nearly agree, and the constrained-NN route -- the
    traced one -- is the closer of the two against the same-SCF-code row.  That
    is a bound, not a tolerance: the models still differ in rank limits and in
    refinement, so no `approx` is claimed against the reference anywhere.
    """
    reference = {name: reference_total(row, 6) for name, row in FIXTURE['backends'].items()}
    spread = reference['psi4']/reference['nwchem'] - 1.
    defect = {k: native_total(p, 6)/reference['psi4'] - 1. for k, p in chains.items()}
    assert native_total(chains['direct_ov'], 6) == pytest.approx(46.897125, abs=1e-5)
    assert native_total(chains['lambda1000'], 6) == pytest.approx(46.768291, abs=1e-5)
    for value in defect.values():
        assert 0. < value < spread
    assert defect['lambda1000'] < defect['direct_ov']
    # Our chain is closest to the row that declares our own GRAC shift, and
    # further from each row the further that row's shift is from ours.
    for route, properties in chains.items():
        total = native_total(properties, 6)
        against = {name: abs(total/value - 1.) for name, value in reference.items()}
        assert min(against, key=against.get) == 'psi4', (route, against)
        assert against['psi4'] < against['nwchem'] < against['dalton']


@pytest.mark.long
def test_site_resolved_splits_are_far_outside_that_spread(chains):
    """Where the disagreement actually is, and that it is not a tolerance question.

    The molecular total agreeing to a fraction of a percent while O-O grows past
    +150% at n=10 and H-H sits 46% low at n=6 is the signature of a different
    partition of the same molecular response, not of a converging one.  Both
    remaining differences point there: the reference refines with PFIT, and its
    `H-Limit 1` model does not carry the rank-3 hydrogen column ours does.
    """
    reference = FIXTURE['backends']['psi4']['isotropic']
    for route, properties in chains.items():
        native = native_by_type(properties)
        assert native.keys() == reference.keys()
        assert native['H H'][6]/reference['H H']['6'] - 1. < -.4
        assert -.2 < native['H O'][6]/reference['H O']['6'] - 1. < -.1
        assert .2 < native['H O'][8]/reference['H O']['8'] - 1. < .3
        assert .3 < native['O O'][6]/reference['O O']['6'] - 1. < .4
        assert .2 < native['O O'][8]/reference['O O']['8'] - 1. < .3
        assert native['O O'][10]/reference['O O']['10'] - 1. > 1.5
        # The defect is worst at the highest order the reference prints, and the
        # sign pattern -- O-O too large, H-H too small -- is the same in both
        # routes, so it is not the response fit: it is the partition.
        defects = {(pair, k): native[pair][k]/reference[pair][str(k)] - 1.
                   for pair, orders in (('O O', (6, 8, 10)), ('H O', (6, 8))) for k in orders}
        assert max(defects, key=defects.get) == ('O O', 10), (route, defects)
        # And the native chain carries orders the reference model does not print.
        for pair in ('H O', 'H H'):
            assert set(native[pair]) > {int(k) for k in reference[pair] if reference[pair][k]}


@pytest.mark.long
def test_native_chain_is_a_differently_declared_model_than_the_reference(chains):
    """What is structurally different, recorded next to the numbers above.

    Per-site rank limits are not declarable: the LW workspace is uniform, so the
    reference's `{O: 2, H: 1}` model is rejected before any compute rather than
    silently approximated.  This is asserted so the comparison above can never
    be mistaken for a parity claim.
    """
    for properties in chains.values():
        assert MATCHED_AUX in properties.model and 'ALDA' in properties.model
        assert properties.local.metadata.residual_policy == 'production'
        assert properties.local.metadata.residual_tolerance == 1.e-6
        assert {c.order for pair in properties.dispersion.pairs
                for c in pair.coefficients} == {6, 8, 10, 12}
        assert len(properties.dispersion.pairs) == 9
