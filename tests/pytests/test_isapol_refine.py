"""psi4's PFIT refinement stage against a CamCASP ``pfit`` run.

The reference numbers at the bottom of this file are the ``Print Parameters``
output of CamCASP 6.0's ``src/pfit`` program, run on a formatted-``Lattice``
input (``pfit.f90``/``points.f90::read_data``) that carries exactly the sites,
local axes, ``.pdef`` polarizability model with its ``COPY`` symmetry, point
cloud, point-to-point responses and penalty anchors/strengths that the psi4
refinement below is built from.  No CamCASP source is executed, linked or
vendored by psi4; the numbers are data produced by running upstream's own
program, and the model construction they check is transcribed from
``src/tools/process_data.F90::write_pfit_local_symm``.

``pfit`` prints its fitted parameters with ``f15.8``, so the oracle resolves
only to 5e-9 absolute.  That print rounding, not the algebra, sets the
tolerances here: the largest deviation observed across the three cases is
4.98e-9, i.e. every parameter agrees with the reference to the last printed
digit.

The inputs are built from dyadic rationals and integer direction vectors, using
only ``+ - * /`` and ``sqrt``, so they are bit-identical on any IEEE-754
platform and do not depend on a random-number stream.
"""
import hashlib
import math

import numpy as np
import pytest

import psi4
from psi4 import core
from psi4.driver.procrouting import isapol_logging as _lg
from psi4.driver.procrouting import isapol_lw as _lw
from psi4.driver.procrouting import isapol_refine as R

pytestmark = [pytest.mark.smoke]

IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
#: H1's local frame is the C2v image of H2's, the signed permutation
#: diag(-1,-1,1).  This is what the reference case's own ``H2O.axes`` builds --
#: ``H1  z global Z x from H2 to H1`` at this geometry, tracked as
#: ``data_isapol/orient_local/H2O.axes`` -- and it is the frame that makes the
#: two hydrogens share one set of variables with no sign changes.  Leaving both
#: in global axes instead is what
#: ``test_a_copy_equivalence_reports_how_far_its_sites_disagree`` measures.
H1_FRAME = ((-1., 0., 0.), (0., -1., 0.), (0., 0., 1.))
#: ~/gits/CamCASP/tests/H2O_props/psi4/H2O-avtz.clt, bohr.
O_ORIGIN = (0., 0., 0.)
H1_ORIGIN = (-1.45365196, 0., -1.12168732)
H2_ORIGIN = (1.45365196, 0., -1.12168732)

#: ranks, point count, damping, mask, weight type, weight coefficient
CASES = {
    'l2h1': dict(rank_o=2, rank_h=1, npoint=40, damping=0.0, masked=False,
                 weight_type=3, weight_coefficient=1.0e-3),
    'rank4': dict(rank_o=4, rank_h=1, npoint=30, damping=0.0, masked=True,
                  weight_type=3, weight_coefficient=1.0e-3),
    'damped': dict(rank_o=2, rank_h=1, npoint=40, damping=1.5, masked=False,
                   weight_type=5, weight_coefficient=2.0e-3),
}


def dyadic(n, offset):
    """An ``n`` x ``n`` symmetric positive-definite matrix of dyadic rationals."""
    b = np.array([[(((3 * i + 5 * k + offset) % 19) - 9) / 8.0 for k in range(n)]
                  for i in range(n)])
    return b @ b.T / 8.0 + np.eye(n) * (n / 4.0)


def kept(n):
    """CamCASP keeps a component pair only if the reference site's anchor
    exceeds the cutoff.  This mask zeroes most off-diagonal anchors exactly, so
    the ``abs(pol) > cutoff`` branch of the model builder has to drop them."""
    mask = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (5 * i + 3 * j) % 4 == 0:
                mask[i, j] = True
    return mask | mask.T


def points(count):
    """Fit points on integer directions at dyadic radii, 4.0 to 8.0 bohr."""
    out = []
    for i in range(count):
        v = np.array([((i * 7) % 13) - 6, ((i * 11) % 17) - 8, ((i * 5) % 11) - 5],
                     dtype=float)
        if not v.any():
            v = np.array([3., -2., 1.])
        radius = 4.0 + ((i * 3) % 9) / 2.0
        out.append(v * (radius / np.sqrt(v @ v)))
    return np.array(out)


_CACHE = {}


def case(name):
    """Everything a refinement needs, plus the model that generated its data.

    The data is the forward map of a *different* local model, so the fit is a
    genuine compromise between the data term and the penalty term rather than a
    fixed point at the anchors.
    """
    if name in _CACHE:
        return _CACHE[name]
    spec = CASES[name]
    sites = (R.RefinementSite('O', 'O', O_ORIGIN, IDENTITY, spec['rank_o']),
             R.RefinementSite('H1', 'H', H1_ORIGIN, H1_FRAME, spec['rank_h']),
             R.RefinementSite('H2', 'H', H2_ORIGIN, IDENTITY, spec['rank_h']))
    no, nh = (spec['rank_o'] + 1) ** 2, (spec['rank_h'] + 1) ** 2
    anchor_o = dyadic(no, 7)
    if spec['masked']:
        anchor_o = np.where(kept(no), anchor_o, 0.0)
    anchor_h = dyadic(nh, 11)
    anchors = [anchor_o, anchor_h, anchor_h.copy()]
    source = [anchor_o + dyadic(no, 3) / 8.0, anchor_h + dyadic(nh, 5) / 8.0]
    source = [source[0], source[1], source[1]]

    model = R.refinement_model(sites, anchors, cutoff=1.0e-4,
                               weight_type=spec['weight_type'],
                               weight_coefficient=spec['weight_coefficient'],
                               provenance=f'test_isapol_refine case {name}')
    pts = points(spec['npoint'])
    fields = R.channel_fields(pts, model, damping=spec['damping'])
    response = R.point_to_point_response(fields, model, source)
    built = dict(name=name, model=model, points=pts, fields=fields,
                 damping=spec['damping'], anchors=anchors, source=source,
                 response=response, targets=R.pack_lower_triangle(response))
    _CACHE[name] = built
    return built


_SOLVED = {}


def solve(built):
    """The refinement of a case, solved once and reused; the rank-4 case is a
    101-parameter fit over 465 point pairs and is not cheap."""
    if built['name'] not in _SOLVED:
        _SOLVED[built['name']] = R.refine(
            built['model'], built['points'], built['targets'],
            fields=built['fields'], damping=built['damping'],
            target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
            source_id='test_isapol_refine',
            generation_record='forward map of a perturbed local model')
    return _SOLVED[built['name']]


# --------------------------------------------------------------- transcription

def test_component_rank_matches_index_to_rank():
    """``index_to_rank`` (process_data.F90:1878-1892) on a one-based index."""
    expected = [0] + [1] * 3 + [2] * 5 + [3] * 7 + [4] * 9
    assert [R.component_rank(i) for i in range(25)] == expected
    assert len(R.COMPONENT_NAMES) == 25
    assert R.COMPONENT_NAMES[:5] == ('00', '10', '11c', '11s', '20')
    assert R.COMPONENT_NAMES[-1] == '44s'
    with pytest.raises(ValueError):
        R.component_rank(25)
    with pytest.raises(ValueError):
        R.component_rank(-1)


@pytest.mark.parametrize('weight_type,rank1,rank2,expected', [
    (0, 0, 0, 0.0),
    (1, 0, 0, 2.0e-3),
    (1, 4, 4, 2.0e-3),
    (2, 0, 0, 2.0e-3 / 4.0),
    (3, 0, 0, 2.0e-3 / 10.0),
    (4, 1, 1, 2.0e-3),
    (4, 1, 2, 0.0),
    (5, 1, 1, 2.0e-3),
    (5, 2, 1, 2.0e-3 * 10.0e-3),
    (6, 1, 0, 2.0e-3),
    (6, 0, 3, 2.0e-3 * 10.0e-2),
])
def test_penalty_weight_matches_weights(weight_type, rank1, rank2, expected):
    """``weights`` (process_data.F90:1810-1876), alpha = 3 so |a|+1 = 4."""
    got = R.penalty_weight(weight_type=weight_type, weight_coefficient=2.0e-3,
                           alpha=3.0, frequency=0.0, rank1=rank1, rank2=rank2)
    assert got == pytest.approx(expected, rel=0.0, abs=1.0e-18)


def test_penalty_weight_frequency_scaling():
    static = R.penalty_weight(weight_type=1, weight_coefficient=1.0e-3, alpha=1.0,
                              frequency=0.0, rank1=1, rank2=1)
    dynamic = R.penalty_weight(weight_type=1, weight_coefficient=1.0e-3, alpha=1.0,
                               frequency=0.5, rank1=1, rank2=1)
    assert dynamic == pytest.approx(static / 1.25, rel=1.0e-15)


def test_penalty_weight_rejects_illegal_inputs():
    with pytest.raises(ValueError):
        R.penalty_weight(weight_type=7, weight_coefficient=1.0e-3, alpha=1.0,
                         frequency=0.0, rank1=0, rank2=0)
    with pytest.raises(ValueError):
        R.penalty_weight(weight_type=1, weight_coefficient=-1.0e-3, alpha=1.0,
                         frequency=0.0, rank1=0, rank2=0)
    with pytest.raises(ValueError):
        R.penalty_weight(weight_type=1, weight_coefficient=1.0e-3, alpha=1.0,
                         frequency=-0.5, rank1=0, rank2=0)


# ----------------------------------------------------------------- model shape

def test_l2h1_model_shape():
    """The reference protocol: O to rank 2, H to rank 1, C2v-equivalent H sites."""
    model = case('l2h1')['model']
    assert model.channel_count == 9 + 4 + 4
    assert model.site_types == ('O', 'H')
    assert model.reference_sites == (0, 1)
    assert model.equivalent_sites == ((0,), (1, 2))
    assert model.channel_offsets == (0, 9, 13)
    # upper triangles of a 9x9 and a 4x4 block, the latter shared by two sites
    assert model.parameter_count == 45 + 10
    assert model.nonsymmetric_parameter_count == 45 + 2 * 10
    assert model.parameter_labels[0] == 'O_00_00_A'
    assert model.parameter_labels[44] == 'O_22s_22s_A'
    assert model.parameter_labels[45] == 'H1_00_00_A'
    assert model.parameter_labels[-1] == 'H1_11s_11s_A'
    assert len(model.anchors) == len(model.strengths) == model.parameter_count
    assert model.anchor_sha256 and model.provenance
    # H1's frame is the C2v image of H2's, so the two hydrogens' *local* tensors
    # are the same array and the COPY declaration costs nothing at all.
    assert model.copy_anchor_discrepancy == 0.0


#: Racah ``00,10,11c,11s`` sign pattern of the mirror that maps one hydrogen of a
#: planar molecule onto the other: in-plane ``11c`` flips, ``z`` and the
#: out-of-plane ``11s`` do not.
MIRROR = np.diag([1., 1., -1., 1.])


def test_a_copy_equivalence_reports_how_far_its_sites_disagree():
    """``copy_anchor_discrepancy``: what a COPY declaration costs in the wrong frame.

    A COPY equivalence is written in each site's *own* local axes, so declaring
    one commits the caller to sites whose local tensors coincide.  Leave two
    mirror-image sites in global axes and they do not: the ``10,11c`` coupling
    carries opposite signs.  CamCASP writes the reference site's value to every
    site of the type regardless, so this module does too, and reports the
    distance rather than symmetrizing the anchors or refusing the model.  A
    penalty strong enough to pin the parameters to the anchors then misses the
    *equivalent* site's own anchors by precisely that distance -- which is the
    measurable statement that the frames, not the fit, are what was wrong.
    """
    built = case('l2h1')
    anchor_o, anchor_h = built['anchors'][0], built['anchors'][1]
    mirrored = MIRROR @ anchor_h @ MIRROR
    global_axes = (R.RefinementSite('O', 'O', O_ORIGIN, IDENTITY, 2),
                   R.RefinementSite('H1', 'H', H1_ORIGIN, IDENTITY, 1),
                   R.RefinementSite('H2', 'H', H2_ORIGIN, IDENTITY, 1))
    model = R.refinement_model(global_axes, [anchor_o, anchor_h, mirrored],
                               cutoff=1.0e-4, weight_coefficient=1.0e12,
                               provenance='mirror-image sites left in global axes')

    # The variable list is unchanged -- a sign flip does not move |anchor| past
    # the cutoff -- so the two models differ in the anchors alone.
    assert model.parameter_labels == built['model'].parameter_labels
    expected = max(abs(anchor_h[i, j] - mirrored[i, j])
                   for i in range(4) for j in range(i, 4)
                   if abs(anchor_h[i, j]) > 1.0e-4)
    assert expected > 0.0
    assert model.copy_anchor_discrepancy == expected
    assert model.copy_anchor_discrepancy == 2.0 * abs(anchor_h[1, 2])

    # Pinned hard, so the data term is irrelevant and the parameters sit on the
    # anchors of the *reference* hydrogen, H1.
    result = R.refine(model, built['points'], built['targets'], damping=0.0,
                      target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                      source_id='test_isapol_refine',
                      generation_record='pinned mirror-image COPY declaration')
    assert result.status == core.IsaPfitStatus.Solved
    assert result.anchor_shift_maxabs < 1.0e-6
    assert np.max(np.abs(result.refined_tensors[1] - anchor_h)) < 1.0e-6
    assert np.max(np.abs(result.refined_tensors[2] - mirrored)) == pytest.approx(
        model.copy_anchor_discrepancy, abs=1.0e-6)
    # Same anchors in each site's own local axes: nothing to report, and the
    # equivalent site is then reproduced as exactly as the reference one.
    matched = R.refinement_model(global_axes, [anchor_o, anchor_h, anchor_h.copy()],
                                 cutoff=1.0e-4, provenance='matched local anchors')
    assert matched.copy_anchor_discrepancy == 0.0


def test_cutoff_drops_components_only_by_the_reference_site():
    """A component pair survives only if the *reference* site's anchor clears
    the cutoff (process_data.F90:2109-2118); a large value elsewhere in the type
    does not rescue it."""
    built = case('rank4')
    model = built['model']
    anchor_o = built['anchors'][0]
    survivors = sum(1 for i in range(25) for j in range(i, 25)
                    if abs(anchor_o[i, j]) > 1.0e-4)
    assert model.parameter_count == survivors + 10
    assert model.parameter_count < 325 + 10
    assert model.channel_count == 25 + 4 + 4
    for label, entries in zip(model.parameter_labels, model.parameter_entries):
        site, row, col = entries[0]
        assert row <= col
        if site == 0:
            assert abs(anchor_o[row, col]) > 1.0e-4
    # the two hydrogens are one variable each, occupying both sites
    hydrogen = [e for lab, e in zip(model.parameter_labels, model.parameter_entries)
                if lab.startswith('H1_')]
    assert len(hydrogen) == 10
    assert all(tuple(s for s, _, _ in e) == (1, 2) for e in hydrogen)


def test_a_large_value_at_an_equivalent_site_does_not_rescue_a_component():
    """The cutoff test reads ``pol(RefSiteIndx,...)`` only, so a component the
    reference site screens out is absent from the model no matter how large the
    equivalent site's value is."""
    sites = (R.RefinementSite('O', 'O', O_ORIGIN, IDENTITY, 1),
             R.RefinementSite('H1', 'H', H1_ORIGIN, H1_FRAME, 1),
             R.RefinementSite('H2', 'H', H2_ORIGIN, IDENTITY, 1))
    reference = np.eye(4)
    equivalent = np.eye(4)
    equivalent[0, 3] = equivalent[3, 0] = 12.5
    model = R.refinement_model(sites, [np.eye(4), reference, equivalent],
                              cutoff=1.0e-4, provenance='cutoff asymmetry')
    assert 'H1_00_11s_A' not in model.parameter_labels
    assert 'H1_00_00_A' in model.parameter_labels
    # and the reverse ordering keeps it, because the reference site is first
    swapped = (sites[0],
               R.RefinementSite('H2', 'H', H2_ORIGIN, IDENTITY, 1),
               R.RefinementSite('H1', 'H', H1_ORIGIN, H1_FRAME, 1))
    model = R.refinement_model(swapped, [np.eye(4), equivalent, reference],
                              cutoff=1.0e-4, provenance='cutoff asymmetry')
    assert 'H2_00_11s_A' in model.parameter_labels


def test_forward_map_is_symmetric_and_packs_consistently():
    built = case('l2h1')
    response = built['response']
    assert np.max(np.abs(response - response.T)) < 1.0e-15
    packed = built['targets']
    npoint = len(built['points'])
    assert packed.shape == (npoint * (npoint + 1) // 2,)
    for i in range(npoint):
        for j in range(i + 1):
            assert packed[i * (i + 1) // 2 + j] == response[i, j]


def test_anchors_are_a_fixed_point_of_the_refinement():
    """With the anchors' own forward map as data, both terms of the objective
    are minimized at the anchors, so the refinement must not move them."""
    built = case('l2h1')
    model, anchors = built['model'], built['anchors']
    response = R.point_to_point_response(built['fields'], model, anchors)
    result = R.refine(model, built['points'], R.pack_lower_triangle(response),
                      fields=built['fields'],
                      target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                      source_id='test_isapol_refine',
                      generation_record='forward map of the anchors')
    assert result.status == core.IsaPfitStatus.Solved
    assert result.anchor_shift_maxabs < 1.0e-9
    assert result.diagnostics.data_rms < 1.0e-12
    for refined, anchor in zip(result.refined_tensors, anchors):
        assert np.max(np.abs(refined - anchor)) < 1.0e-9
        assert np.max(np.abs(refined - refined.T)) == 0.0


def test_refined_tensors_share_one_block_per_type():
    result = solve(case('l2h1'))
    assert result.status == core.IsaPfitStatus.Solved
    assert np.array_equal(result.refined_tensors[1], result.refined_tensors[2])
    assert result.refinement_status == 'PFIT_refined_against_point_to_point_response'
    assert 'anchors' in result.penalty_convention


def test_excluded_components_are_zero_not_carried_over():
    """Cutoff-excluded components are absent from the model, so the refined
    tensor holds exact zeros there rather than the anchor value."""
    built = case('rank4')
    result = solve(built)
    anchor_o, refined_o = built['anchors'][0], result.refined_tensors[0]
    zeros = [(i, j) for i in range(25) for j in range(25)
             if abs(anchor_o[i, j]) <= 1.0e-4]
    assert zeros
    for i, j in zeros:
        assert refined_o[i, j] == 0.0


def test_isotropic_scalars_reduce_each_rank_block():
    """``alpha_l = trace(alpha_ll)/(2l+1)``, per site, rank1 upward.

    This is the reduction the isotropic Casimir-Polder sum consumes, and it is
    the same one ``isapol_lw`` applies to the *unrefined* localized tensors; the
    point of having it here is that a refined model can be fed to the dispersion
    kernel without the caller re-deriving the packing.  The expectation below is
    recomputed from ``refined_tensors`` independently of the helper.
    """
    built = case('l2h1')
    result = solve(built)
    ranks, scalars = R.isotropic_scalars(result)
    assert ranks == ((1, 2), (1,), (1,))
    for site_ranks, site_scalars, tensor in zip(ranks, scalars, result.refined_tensors):
        assert len(site_scalars) == len(site_ranks)
        for l, value in zip(site_ranks, site_scalars):
            block = np.asarray(tensor)[l * l:(l + 1) ** 2, l * l:(l + 1) ** 2]
            assert block.shape == (2 * l + 1, 2 * l + 1)
            assert value == np.trace(block) / (2 * l + 1)
    # Rank0 is a refinement variable but not a polarizability the sum runs
    # over, so no site reports it even though every tensor carries the block.
    assert all(0 not in site_ranks for site_ranks in ranks)
    assert result.refined_tensors[0][0, 0] != 0.0
    # COPY-equivalent sites share one block, hence one set of scalars.
    assert scalars[1] == scalars[2]
    # The refinement moved the model: the scalars are not the anchors'.
    anchor_o = built['anchors'][0]
    assert scalars[0][0] != np.trace(anchor_o[1:4, 1:4]) / 3.0


def test_isotropic_scalars_requires_a_refinement_result():
    with pytest.raises(ValueError):
        R.isotropic_scalars(solve(case('l2h1')).refined_tensors)


# ---------------------------------------------- dispersion from refined tensors

QUADRATURE = _lw.Provenance('CasimirGrid(4, 0.5)', '0' * 64, 'test_isapol_refine',
                            'Gauss-Legendre imaginary-frequency nodes, static point first')


def refined_nodes(name='l2h1', n_freq=4, *, sites=None, declared=lambda k: None, **overrides):
    """One refinement per ``CasimirGrid(n_freq, 0.5)`` node, static point first.

    The anchors (hence the variable set) are the case's at every node; the data
    are the forward map of the case's source model scaled by ``1/(1+omega^2)``,
    so the refined scalars genuinely vary along the grid.  ``sites`` replaces
    the case's sites (e.g. relabeled); ``declared(k)`` is node k's
    ``declared_variables``.
    """
    built = case(name)
    spec = dict(CASES[name], **overrides)
    grid = core.CasimirGrid(n_freq, 0.5)
    nodes = []
    for k in range(n_freq + 1):
        omega = grid.omega(k)
        model = R.refinement_model(sites or built['model'].sites, built['anchors'],
                                   frequency_au=omega, cutoff=1.0e-4,
                                   weight_type=spec['weight_type'],
                                   weight_coefficient=spec['weight_coefficient'],
                                   provenance=f'test_isapol_refine case {name}',
                                   declared_variables=declared(k))
        source = [t / (1.0 + omega * omega) for t in built['source']]
        targets = R.pack_lower_triangle(R.point_to_point_response(built['fields'], model, source))
        nodes.append(R.refine(model, built['points'], targets, fields=built['fields'],
                              target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                              source_id='test_isapol_refine',
                              generation_record=f'scaled forward map, node {k}'))
    return tuple(nodes), [grid.cp_weight(k) for k in range(n_freq + 1)]


def oracle_coefficient(nodes, weights, a, b, order, ranks):
    """``binomial(2la+2lb, 2la) sum_f w_f alpha_a,la(f) alpha_b,lb(f)`` from the tensors."""
    def scalar(node, site, l):
        block = np.asarray(node.refined_tensors[site])[l * l:(l + 1) ** 2, l * l:(l + 1) ** 2]
        return np.trace(block) / (2 * l + 1)
    value, missing = 0.0, []
    for la in range(1, order // 2 - 1):
        lb = order // 2 - 1 - la
        if la not in ranks[a] or lb not in ranks[b]:
            missing.append((la, lb))
            continue
        integral = 0.0
        for node, w in zip(nodes, weights):
            if w != 0.0:
                integral += w * scalar(node, a, la) * scalar(node, b, lb)
        value += math.comb(2 * la + 2 * lb, 2 * la) * integral
    return value, tuple(missing)


@pytest.fixture(scope='module')
def l2h1_nodes():
    return refined_nodes()


def test_refined_dispersion_matches_an_independent_contraction(l2h1_nodes):
    """Every ordered site pair and order, against the closed form on the tensors."""
    nodes, weights = l2h1_nodes
    record = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                            quadrature_provenance=QUADRATURE)
    assert record.labels == ('O', 'H1', 'H2')
    assert record.frequencies == tuple(n.model.frequency_au for n in nodes)
    assert record.frequencies[0] == 0.0 and record.cp_weights[0] == 0.0
    assert record.site_ranks == ((1, 2), (1,), (1,))
    assert record.solver_status == ('Solved',) * len(nodes)
    assert [(p.site_a, p.site_b) for p in record.pairs] == [(a, b) for a in range(3) for b in range(3)]
    digest = hashlib.sha256(''.join(n.model.anchor_sha256 for n in nodes).encode()).hexdigest()
    assert record.anchor_sha256 == digest
    for pair in record.pairs:
        assert [c.order for c in pair.coefficients] == [6, 8, 10, 12]
        for c in pair.coefficients:
            value, missing = oracle_coefficient(nodes, weights, pair.site_a, pair.site_b,
                                                c.order, record.site_ranks)
            assert c.value == pytest.approx(value, rel=1e-13, abs=0.0)
            assert c.missing_rank_pairs == missing
            assert c.unrestricted_complete == (not missing)
    by_pair = {(p.label_a, p.label_b): p.coefficients for p in record.pairs}
    # Only O carries rank 2: O-O C8 is complete, O-H C8 is not, C12 never is.
    assert by_pair['O', 'O'][1].unrestricted_complete
    assert not by_pair['O', 'H1'][1].unrestricted_complete
    assert not any(p.coefficients[3].unrestricted_complete for p in record.pairs)
    # COPY-equivalent hydrogens share one variable set, hence identical rows.
    assert [c.value for c in by_pair['O', 'H1']] == [c.value for c in by_pair['O', 'H2']]


def test_refined_dispersion_declared_ranks_truncate_rather_than_zero_fill(l2h1_nodes):
    nodes, weights = l2h1_nodes
    full = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                          quadrature_provenance=QUADRATURE, max_order=8)
    dipole = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                            quadrature_provenance=QUADRATURE, max_order=8,
                                            site_ranks=((1,), (1,), (1,)))
    assert [c.order for c in full.pairs[0].coefficients] == [6, 8]
    assert dipole.site_ranks == ((1,), (1,), (1,))
    assert full.pairs[0].coefficients[0].value == dipole.pairs[0].coefficients[0].value
    assert full.pairs[0].coefficients[1].unrestricted_complete
    assert dipole.pairs[0].coefficients[1].missing_rank_pairs == ((1, 2), (2, 1))
    assert dipole.pairs[0].coefficients[1].value == 0.0


def test_refined_dispersion_report_and_optional_qcvariables(l2h1_nodes):
    """Reporting changes no number; incomplete orders publish only as INCOMPLETE."""
    nodes, weights = l2h1_nodes
    lines = []
    log = _lg.StageLog(2, writer=lines.append)
    mol = psi4.geometry('units bohr\nsymmetry c1\nno_com\nno_reorient\n'
                        'O 0 0 0\nH -1.45365196 0 -1.12168732\nH 1.45365196 0 -1.12168732\n')
    wfn = core.Wavefunction.build(mol, 'sto-3g')
    record = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                            quadrature_provenance=QUADRATURE, log=log, wfn=wfn)
    quiet = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                           quadrature_provenance=QUADRATURE)
    assert record.pairs == quiet.pairs
    text = ''.join(lines)
    for heading in ('Casimir-Polder quadrature', 'Atomic (same-site) isotropic dispersion',
                    'Pairwise isotropic dispersion coefficients', 'Rank pairs absent',
                    'Rank pairs entering each order', 'Sum over all ordered site pairs',
                    'provenance: ' + QUADRATURE.description):
        assert heading in text, heading
    assert [name for name, _ in log.stages] == [
        'isotropic dispersion from refined tensors (Casimir-Polder)']

    def total(order):
        value = 0.0  # in pair order; builtin sum() compensates and differs by an ulp
        for p in record.pairs:
            value += p.coefficients[(order - 6) // 2].value
        return value

    stem = 'ATOMIC REFINED DISPERSION'
    assert wfn.variable(f'{stem} C6 TOTAL') == total(6)
    assert wfn.variable(f'{stem} C12 TOTAL INCOMPLETE') == total(12)
    assert not wfn.has_variable(f'{stem} C12 TOTAL')
    assert wfn.variable(f'{stem} C8 O O') == record.pairs[0].coefficients[1].value
    assert wfn.has_variable(f'{stem} C8 O H1 INCOMPLETE')
    assert not wfn.has_variable(f'{stem} C8 O H1')
    assert wfn.variable('ATOM O C6 REFINED DISPERSION COEFFICIENT') == \
        record.pairs[0].coefficients[0].value
    assert wfn.has_variable('ATOM H1 C8 REFINED DISPERSION COEFFICIENT INCOMPLETE')
    assert wfn.variable(f'{stem} SITE PAIRS') == 9.0
    assert wfn.variable(f'{stem} MAX ORDER') == 12.0
    assert wfn.variable(f'{stem} QUADRATURE NODES') == len(weights)
    np.testing.assert_array_equal(np.asarray(wfn.array_variable(f'{stem} CP WEIGHTS')),
                                  [weights])
    np.testing.assert_array_equal(np.asarray(wfn.array_variable(f'{stem} QUADRATURE FREQUENCIES')),
                                  [record.frequencies])
    assert wfn.variable(f'{stem} ANCHOR SHIFT MAXABS') == record.anchor_shift_maxabs


def test_refined_dispersion_reaches_a_complete_c12_at_rank_four():
    """O at rank 4 supplies (1,4) and (4,1): the O-O C12 sum closes, O-H does not."""
    nodes, weights = refined_nodes('rank4', n_freq=2)
    record = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                            quadrature_provenance=QUADRATURE)
    o_o, o_h = record.pairs[0].coefficients[3], record.pairs[1].coefficients[3]
    assert o_o.unrestricted_complete and o_o.included_rank_pairs == ((1, 4), (2, 3), (3, 2), (4, 1))
    assert o_h.missing_rank_pairs == ((1, 4), (2, 3), (3, 2))
    assert o_h.included_rank_pairs == ((4, 1),)
    value, _ = oracle_coefficient(nodes, weights, 0, 0, 12, record.site_ranks)
    assert o_o.value == pytest.approx(value, rel=1e-13, abs=0.0)


def _water_wfn():
    mol = psi4.geometry('units bohr\nsymmetry c1\nno_com\nno_reorient\n'
                        'O 0 0 0\nH -1.45365196 0 -1.12168732\nH 1.45365196 0 -1.12168732\n')
    wfn = core.Wavefunction.build(mol, 'sto-3g')
    wfn.set_variable('UNRELATED SENTINEL', 1.5)
    return wfn


def _dispersion_keys(wfn):
    return {k for k in wfn.variables() if 'REFINED DISPERSION' in k}


def test_reused_wavefunction_never_holds_both_suffix_variants(l2h1_nodes):
    """Complete -> INCOMPLETE -> complete on one wavefunction: only current keys survive."""
    nodes, weights = l2h1_nodes
    wfn = _water_wfn()
    keys = ('ATOMIC REFINED DISPERSION C6 TOTAL', 'ATOMIC REFINED DISPERSION C6 O O',
            'ATOM O C6 REFINED DISPERSION COEFFICIENT')
    # Declaring only rank 2 on O drops the (1,1) term from every O-containing C6.
    for site_ranks, complete in ((None, True), (((2,), (1,), (1,)), False), (None, True)):
        record = R.refined_isotropic_dispersion(nodes, cp_weights=weights, max_order=6,
                                                quadrature_provenance=QUADRATURE,
                                                site_ranks=site_ranks, wfn=wfn)
        total = 0.0
        for p in record.pairs:
            total += p.coefficients[0].value
        values = (total, record.pairs[0].coefficients[0].value, record.pairs[0].coefficients[0].value)
        assert record.pairs[0].coefficients[0].unrestricted_complete is complete
        for key, value in zip(keys, values):
            current, opposite = (key, key + ' INCOMPLETE') if complete else (key + ' INCOMPLETE', key)
            assert wfn.variable(current) == value, current
            assert not wfn.has_variable(opposite), opposite
        assert wfn.variable('UNRELATED SENTINEL') == 1.5
    assert not any(k.endswith('INCOMPLETE') and 'C6' in k for k in _dispersion_keys(wfn))


@pytest.mark.parametrize('labels,match', [(('O', 'h', 'H'), 'collide'),
                                          (('O', 'H 1', 'H2'), 'whitespace')])
def test_publication_refuses_aliasing_labels_but_the_model_keeps_them(labels, match):
    built = case('l2h1')
    sites = tuple(R.RefinementSite(label, s.site_type, s.origin_bohr, s.frame, s.rank_limit)
                  for label, s in zip(labels, built['model'].sites))
    nodes, weights = refined_nodes(n_freq=2, sites=sites)
    wfn = _water_wfn()
    wfn.set_variable('ATOMIC REFINED DISPERSION C6 TOTAL', -7.0)
    lines = []
    log = _lg.StageLog(2, writer=lines.append)
    with pytest.raises(ValueError, match=match):
        R.refined_isotropic_dispersion(nodes, cp_weights=weights, quadrature_provenance=QUADRATURE,
                                       log=log, wfn=wfn)
    # Refused before any log stage or write; prior wavefunction data untouched.
    assert lines == [] and log.stages == ()
    assert _dispersion_keys(wfn) == {'ATOMIC REFINED DISPERSION C6 TOTAL'}
    assert wfn.variable('ATOMIC REFINED DISPERSION C6 TOTAL') == -7.0
    assert wfn.variable('UNRELATED SENTINEL') == 1.5
    record = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                            quadrature_provenance=QUADRATURE)
    assert record.labels == labels and len(record.pairs) == 9


def test_declared_and_cutoff_derived_nodes_are_different_models(l2h1_nodes):
    nodes, weights = l2h1_nodes
    labels = nodes[0].model.parameter_labels
    mixed, _ = refined_nodes(declared=lambda k: labels if k == 1 else None)
    with pytest.raises(ValueError, match='declared model'):
        R.refined_isotropic_dispersion(mixed, cp_weights=weights, quadrature_provenance=QUADRATURE)
    explicit, _ = refined_nodes(declared=lambda k: labels)
    assert all(n.model.declared_variables == labels for n in explicit)
    derived = R.refined_isotropic_dispersion(nodes, cp_weights=weights,
                                             quadrature_provenance=QUADRATURE)
    replayed = R.refined_isotropic_dispersion(explicit, cp_weights=weights,
                                              quadrature_provenance=QUADRATURE)
    assert replayed.pairs == derived.pairs


def test_failed_contraction_closes_its_stage_as_failed(l2h1_nodes):
    """Reversed nodes are refused by the kernel; the log says so and nothing is published."""
    nodes, weights = l2h1_nodes
    lines = []
    log = _lg.StageLog(1, writer=lines.append)
    wfn = _water_wfn()
    with pytest.raises(ValueError, match='strictly increasing'):
        R.refined_isotropic_dispersion(nodes[::-1], cp_weights=weights[::-1],
                                       quadrature_provenance=QUADRATURE, log=log, wfn=wfn)
    text = ''.join(lines)
    assert 'Stage FAILED: isotropic dispersion from refined tensors (Casimir-Polder)' in text
    assert 'strictly increasing' in text
    assert log.stages == ()
    assert _dispersion_keys(wfn) == set() and wfn.variable('UNRELATED SENTINEL') == 1.5
    log.stage('next')
    log.stage_end()
    R.refined_isotropic_dispersion(nodes, cp_weights=weights, quadrature_provenance=QUADRATURE,
                                   log=log)
    text = ''.join(lines)
    assert [name for name, _ in log.stages] == [
        'next', 'isotropic dispersion from refined tensors (Casimir-Polder)']
    assert text.count('Stage complete: isotropic dispersion') == 1
    assert text.index('Stage FAILED') < text.index('Stage complete: next') \
        < text.index('Stage complete: isotropic dispersion')


def _replace_node(nodes, index, node):
    return nodes[:index] + (node,) + nodes[index + 1:]


@pytest.mark.parametrize('fault,match', [
    ('empty', 'non-empty sequence'),
    ('tensors', 'non-empty sequence'),
    ('repeated', 'distinct'),
    ('sites', 'identical declared site set'),
    ('penalty', 'weight_coefficient differs'),
    ('provenance', 'Provenance'),
    ('order7', 'max_order'),
    ('order14', 'max_order'),
    ('order_bool', 'max_order'),
    ('weights_short', 'one CP weight'),
    ('weights_negative', 'non-negative'),
    ('weights_nan', 'finite'),
    ('weights_zero', 'not all zero'),
    ('static_weight', 'static node'),
    ('ranks_count', 'one rank tuple per site'),
    ('ranks_above_limit', 'exceeds the refinement rank limit'),
    ('ranks_order', 'strictly increasing'),
    ('ranks_type', 'integer ranks'),
])
def test_refined_dispersion_refusals(l2h1_nodes, fault, match):
    nodes, weights = l2h1_nodes
    kwargs = dict(cp_weights=list(weights), quadrature_provenance=QUADRATURE)
    if fault == 'empty':
        nodes = ()
    elif fault == 'tensors':
        nodes = tuple(n.refined_tensors for n in nodes)
    elif fault == 'repeated':
        nodes = _replace_node(nodes, 2, nodes[1])
    elif fault == 'sites':
        nodes = _replace_node(nodes, 1, solve(case('rank4')))
    elif fault == 'penalty':
        nodes = _replace_node(nodes, 1, refined_nodes(weight_coefficient=2.0e-3)[0][1])
    elif fault == 'provenance':
        kwargs['quadrature_provenance'] = QUADRATURE.description
    elif fault.startswith('order'):
        kwargs['max_order'] = {'order7': 7, 'order14': 14, 'order_bool': True}[fault]
    elif fault == 'weights_short':
        kwargs['cp_weights'] = weights[:-1]
    elif fault == 'weights_negative':
        kwargs['cp_weights'][1] = -1.0
    elif fault == 'weights_nan':
        kwargs['cp_weights'][1] = float('nan')
    elif fault == 'weights_zero':
        kwargs['cp_weights'] = [0.0] * len(weights)
    elif fault == 'static_weight':
        kwargs['cp_weights'][0] = 1.0
    else:
        kwargs['site_ranks'] = {'ranks_count': ((1,), (1,)),
                                'ranks_above_limit': ((1, 2), (1, 2), (1,)),
                                'ranks_order': ((2, 1), (1,), (1,)),
                                'ranks_type': ((1.0,), (1,), (1,))}[fault]
    with pytest.raises(ValueError, match=match):
        R.refined_isotropic_dispersion(nodes, **kwargs)


# ------------------------------------------------------- declared variable list

def test_declared_variables_replay_the_cutoff_list_exactly():
    """A ``.pdef`` that names exactly what the cutoff would have kept.

    ``declared_variables`` is how a supplied ``.pdef`` variable list enters the
    model instead of being derived from ``cutoff``.  Handed the cutoff's own
    list back, it must reproduce the cutoff-derived model term for term --
    labels, entries, anchors and strengths -- so that the only thing the option
    can change is which variables exist, never what a variable means.
    """
    built = case('rank4')
    derived = built['model']
    replayed = R.refinement_model(derived.sites, built['anchors'], cutoff=1.0e-4,
                                  weight_type=3, weight_coefficient=1.0e-3,
                                  provenance='declared replay of the cutoff list',
                                  declared_variables=derived.parameter_labels)
    assert replayed.parameter_labels == derived.parameter_labels
    assert replayed.parameter_entries == derived.parameter_entries
    assert replayed.anchors == derived.anchors
    assert replayed.strengths == derived.strengths
    assert replayed.nonsymmetric_parameter_count == derived.nonsymmetric_parameter_count
    # Every declared variable cleared the cutoff, so none is unpenalized, and
    # the declared list is what records that this model was declared at all --
    # the anchor hash cannot, because the supplied tensors are the same ones.
    assert replayed.unpenalized_variables == ()
    assert replayed.declared_variables == tuple(derived.parameter_labels)
    assert derived.declared_variables == ()
    assert replayed.anchor_sha256 == derived.anchor_sha256


def test_declared_variables_add_free_parameters_the_cutoff_dropped():
    """The extra variables of a ``.pdef`` with no ``Penalties`` line for them.

    ``output_2``'s benzene ``.pdef`` names more variables than its ``Penalties``
    block anchors; the surplus ones are free parameters.  This is the state
    ``declared_variables`` has to be able to express: anchor 0.0, strength 0.0,
    listed in ``unpenalized_variables``, and placed in component-scan order
    rather than in the order they were declared.  The cutoff is unchanged and
    still recorded -- it is what decides which variables are *anchored* -- so
    this is a different declared model, not a loosened cutoff.
    """
    built = case('rank4')
    derived = built['model']
    dropped = tuple(f'O_{R.COMPONENT_NAMES[i]}_{R.COMPONENT_NAMES[j]}_A'
                    for i in range(25) for j in range(i, 25)
                    if f'O_{R.COMPONENT_NAMES[i]}_{R.COMPONENT_NAMES[j]}_A'
                    not in derived.parameter_labels)
    assert len(dropped) > 3
    # declared last, so the component-scan ordering below is a real assertion
    extra = dropped[:3]
    wide = R.refinement_model(derived.sites, built['anchors'], cutoff=1.0e-4,
                              weight_type=3, weight_coefficient=1.0e-3,
                              provenance='declared list with free parameters',
                              declared_variables=tuple(derived.parameter_labels) + extra)
    assert wide.parameter_count == derived.parameter_count + 3
    assert wide.unpenalized_variables == extra
    assert wide.cutoff == derived.cutoff
    assert wide.weight_coefficient == derived.weight_coefficient
    positions = [wide.parameter_labels.index(name) for name in extra]
    assert positions == sorted(positions)
    assert max(positions) < wide.parameter_labels.index(derived.parameter_labels[-1])
    for index in positions:
        assert wide.anchors[index] == 0.0
        assert wide.strengths[index] == 0.0
    # Unpenalized means unpenalized: the surplus parameters answer to the data
    # term alone, so they move off zero and the data residual falls.
    solved = R.refine(wide, built['points'], built['targets'], damping=built['damping'],
                      target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                      source_id='test_isapol_refine',
                      generation_record='declared variable list with free parameters')
    assert solved.status == core.IsaPfitStatus.Solved
    assert all(solved.parameters[index] != 0.0 for index in positions)
    assert solved.diagnostics.data_sse < solve(built).diagnostics.data_sse


def test_declared_variables_refuse_to_omit_a_penalized_variable():
    """A declared list short of a variable the cutoff keeps would leave that
    anchor unfitted while the model still advertised the ``Penalties`` block as
    covering the fit, so it is refused and the missing names are reported."""
    built = case('rank4')
    derived = built['model']
    with pytest.raises(ValueError, match='declared_variables omits'):
        R.refinement_model(derived.sites, built['anchors'], cutoff=1.0e-4,
                           provenance='short declared list',
                           declared_variables=derived.parameter_labels[1:])


@pytest.mark.parametrize('declared,pattern', [
    ((), 'at least one'),
    (('O_00_00_A', 'O_00_00_A'), 'must not repeat'),
    (('O_00_00',), 'is not a <site>'),
    (('X_00_00_A',), 'names no single reference site'),
    # H2 is an equivalent site; the COPY declaration means it carries no
    # variables of its own, so naming it is a refusal, not a synonym for H1.
    (('H2_00_00_A',), 'names no single reference site'),
    (('O_00_zz_A',), 'does not name two multipole components'),
    (('O_11c_10_A',), 'is below the diagonal'),
    (('H1_00_20_A',), "outside site H1's rank limit"),
])
def test_declared_variables_reject_malformed_names(declared, pattern):
    built = case('l2h1')
    with pytest.raises(ValueError, match=pattern):
        R.refinement_model(built['model'].sites, built['anchors'], cutoff=1.0e-4,
                           provenance='malformed declared list',
                           declared_variables=declared)


# ---------------------------------------------------------------- input guards

def test_refine_requires_declared_provenance():
    built = case('l2h1')
    with pytest.raises(ValueError):
        R.refine(built['model'], built['points'], built['targets'],
                 fields=built['fields'],
                 target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                 source_id='   ', generation_record='nonblank')
    with pytest.raises(ValueError):
        R.refine(built['model'], built['points'], built['targets'],
                 fields=built['fields'],
                 target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                 source_id='nonblank', generation_record='')


def test_native_direct_origin_requires_its_own_representation():
    built = case('l2h1')
    common = dict(fields=built['fields'], source_id='native',
                  generation_record='native point-charge response')
    with pytest.raises(ValueError):
        R.refine(built['model'], built['points'], built['targets'],
                 target_origin=core.IsaPfitTargetOrigin.NativeDirectActualPointResponse,
                 response_representation='fitted_density_coefficients', **common)
    with pytest.raises(ValueError):
        # a native direct target may not name an auxiliary basis
        R.refine(built['model'], built['points'], built['targets'],
                 target_origin=core.IsaPfitTargetOrigin.NativeDirectActualPointResponse,
                 response_representation='native_point_charge_ov_operators',
                 auxiliary_basis_id='aug-cc-pVTZ-jkfit', **common)


def test_an_undetermined_refinement_raises_rather_than_returning_parameters():
    """Unpenalized variables (weight type 0) and one point: rank deficient."""
    built = case('l2h1')
    model = R.refinement_model(built['model'].sites, built['anchors'], weight_type=0,
                               provenance='unpenalized')
    with pytest.raises(RuntimeError, match='unavailable'):
        R.refine(model, built['points'][:1], built['targets'][:1],
                 target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                 source_id='test_isapol_refine', generation_record='one point')


def test_rejects_a_fit_point_sitting_on_a_site():
    model = case('l2h1')['model']
    with pytest.raises(ValueError):
        R.channel_fields(np.array([[0., 0., 0.], [4., 0., 0.]]), model)


def test_rejects_an_improper_frame():
    with pytest.raises(ValueError):
        R.RefinementSite('O', 'O', O_ORIGIN,
                         ((1., 0., 0.), (0., 1., 0.), (0., 0., -1.)), 1)
    with pytest.raises(ValueError):
        R.RefinementSite('O', 'O', O_ORIGIN,
                         ((1., 0., 0.), (0., 1., 0.), (0., 0., 2.)), 1)


# -------------------------------------------------------------- CamCASP oracle

@pytest.mark.parametrize('name', sorted(CASES))
def test_matches_camcasp_pfit(name):
    """Every fitted parameter agrees with CamCASP ``pfit`` to its last printed
    digit, for the reference L2H1 shape, for a rank-4 model with cutoff-excluded
    components, and with Tang-Toennies damped T functions."""
    built = case(name)
    result = solve(built)
    assert result.status == core.IsaPfitStatus.Solved
    reference = PFIT_PARAMETERS[name]
    assert list(built['model'].parameter_labels) == list(reference['labels'])
    assert result.diagnostics.numerical_rank == built['model'].parameter_count
    np.testing.assert_allclose(result.parameters, reference['parameters'],
                               rtol=0.0, atol=1.0e-8)
    assert result.diagnostics.data_rms == pytest.approx(reference['data_rms'],
                                                        rel=0.0, abs=1.0e-8)
    assert result.diagnostics.data_max_residual == pytest.approx(
        reference['data_max_residual'], rel=0.0, abs=1.0e-8)


#: CamCASP 6.0 ``pfit`` ``Print Parameters`` output (``f15.8``) for the three
#: cases above, together with the residual statistics from the same run.
PFIT_PARAMETERS = {
    'l2h1': dict(
        data_rms=0.00010149, data_max_residual=0.00066321,
        labels=(
            'O_00_00_A', 'O_00_10_A', 'O_00_11c_A', 'O_00_11s_A', 'O_00_20_A', 'O_00_21c_A',
            'O_00_21s_A', 'O_00_22c_A', 'O_00_22s_A', 'O_10_10_A', 'O_10_11c_A', 'O_10_11s_A',
            'O_10_20_A', 'O_10_21c_A', 'O_10_21s_A', 'O_10_22c_A', 'O_10_22s_A', 'O_11c_11c_A',
            'O_11c_11s_A', 'O_11c_20_A', 'O_11c_21c_A', 'O_11c_21s_A', 'O_11c_22c_A',
            'O_11c_22s_A', 'O_11s_11s_A', 'O_11s_20_A', 'O_11s_21c_A', 'O_11s_21s_A',
            'O_11s_22c_A', 'O_11s_22s_A', 'O_20_20_A', 'O_20_21c_A', 'O_20_21s_A',
            'O_20_22c_A', 'O_20_22s_A', 'O_21c_21c_A', 'O_21c_21s_A', 'O_21c_22c_A',
            'O_21c_22s_A', 'O_21s_21s_A', 'O_21s_22c_A', 'O_21s_22s_A', 'O_22c_22c_A',
            'O_22c_22s_A', 'O_22s_22s_A', 'H1_00_00_A', 'H1_00_10_A', 'H1_00_11c_A',
            'H1_00_11s_A', 'H1_10_10_A', 'H1_10_11c_A', 'H1_10_11s_A', 'H1_11c_11c_A',
            'H1_11c_11s_A', 'H1_11s_11s_A',
        ),
        parameters=(
            3.06392933, -0.07995206, -0.23135574, -0.28119038, -0.09656678, 0.17658923,
            0.45959083, -0.15173498, -0.08010063, 2.87729787, 0.13507317, -0.05214656,
            -0.28502118, -0.21527048, -0.10870750, 0.44637505, 0.43286874, 3.04681709,
            0.27116960, -0.25716485, -0.25337246, -0.21634005, 0.15732531, 0.20281994,
            3.13038763, -0.20020656, -0.26039315, -0.28525433, -0.08694206, 0.00135555,
            2.85527148, 0.25595168, -0.05484409, -0.25358175, -0.34174271, 2.79972463,
            0.17382448, -0.20102784, -0.24182409, 2.69135449, -0.11141236, -0.10563603,
            2.71552327, 0.40205973, 2.75430254, 1.35956794, -0.03454813, -0.04006384,
            -0.13789243, 1.39455422, 0.20162343, -0.03532288, 1.32068252, -0.02307904,
            1.40840472,
        ),
    ),
    'rank4': dict(
        data_rms=0.00075544, data_max_residual=0.00605403,
        labels=(
            'O_00_00_A', 'O_00_20_A', 'O_00_22s_A', 'O_00_32c_A', 'O_00_40_A', 'O_00_42s_A',
            'O_00_44s_A', 'O_10_10_A', 'O_10_21c_A', 'O_10_30_A', 'O_10_32s_A', 'O_10_41c_A',
            'O_10_43c_A', 'O_11c_11c_A', 'O_11c_21s_A', 'O_11c_31c_A', 'O_11c_33c_A',
            'O_11c_41s_A', 'O_11c_43s_A', 'O_11s_11s_A', 'O_11s_22c_A', 'O_11s_31s_A',
            'O_11s_33s_A', 'O_11s_42c_A', 'O_11s_44c_A', 'O_20_20_A', 'O_20_22s_A',
            'O_20_32c_A', 'O_20_40_A', 'O_20_42s_A', 'O_20_44s_A', 'O_21c_21c_A', 'O_21c_30_A',
            'O_21c_32s_A', 'O_21c_41c_A', 'O_21c_43c_A', 'O_21s_21s_A', 'O_21s_31c_A',
            'O_21s_33c_A', 'O_21s_41s_A', 'O_21s_43s_A', 'O_22c_22c_A', 'O_22c_31s_A',
            'O_22c_33s_A', 'O_22c_42c_A', 'O_22c_44c_A', 'O_22s_22s_A', 'O_22s_32c_A',
            'O_22s_40_A', 'O_22s_42s_A', 'O_22s_44s_A', 'O_30_30_A', 'O_30_32s_A',
            'O_30_41c_A', 'O_30_43c_A', 'O_31c_31c_A', 'O_31c_33c_A', 'O_31c_41s_A',
            'O_31c_43s_A', 'O_31s_31s_A', 'O_31s_33s_A', 'O_31s_42c_A', 'O_31s_44c_A',
            'O_32c_32c_A', 'O_32c_40_A', 'O_32c_42s_A', 'O_32c_44s_A', 'O_32s_32s_A',
            'O_32s_41c_A', 'O_32s_43c_A', 'O_33c_33c_A', 'O_33c_41s_A', 'O_33c_43s_A',
            'O_33s_33s_A', 'O_33s_42c_A', 'O_33s_44c_A', 'O_40_40_A', 'O_40_42s_A',
            'O_40_44s_A', 'O_41c_41c_A', 'O_41c_43c_A', 'O_41s_41s_A', 'O_41s_43s_A',
            'O_42c_42c_A', 'O_42c_44c_A', 'O_42s_42s_A', 'O_42s_44s_A', 'O_43c_43c_A',
            'O_43s_43s_A', 'O_44c_44c_A', 'O_44s_44s_A', 'H1_00_00_A', 'H1_00_10_A',
            'H1_00_11c_A', 'H1_00_11s_A', 'H1_10_10_A', 'H1_10_11c_A', 'H1_10_11s_A',
            'H1_11c_11c_A', 'H1_11c_11s_A', 'H1_11s_11s_A',
        ),
        parameters=(
            8.60051425, -0.45260567, -0.16992188, 0.71090218, -0.74366234, 0.21346727,
            -0.01631084, 8.68154729, -0.60237376, -0.30176186, 0.48557414, -0.73410332,
            0.31068424, 8.84249723, -0.58882185, -0.22164151, 0.57478770, -0.70052724,
            0.35689971, 8.34045633, -0.54956715, -0.30049695, 0.88273202, -0.75524180,
            0.09615268, 7.77032629, -0.69849423, -0.16990014, 0.32225050, -0.74411097,
            0.52734242, 7.68194467, -0.55651373, -0.19935471, 0.78914028, -0.70886445,
            7.67592884, -0.57418892, -0.20256725, 0.72445879, -0.75962850, 7.68204563,
            -0.60352944, -0.29289231, 0.46293768, -0.71093000, 7.81568866, -0.49613089,
            -0.24610247, 0.82034623, -0.71097337, 7.77657224, -0.66018428, -0.28516173,
            0.60936743, 7.80836194, -0.61326138, -0.29883063, 0.60937227, 7.68162526,
            -0.54100524, -0.21288015, 0.82030874, 7.61732462, -0.66601726, -0.21289391,
            0.46288966, 7.65349693, -0.50583638, -0.29878961, 7.71670065, -0.50588716,
            -0.28513530, 7.79937971, -0.66601167, -0.24609382, 7.79885477, -0.54101513,
            -0.29296824, 7.71291480, -0.61328257, 7.65246537, -0.66016134, 7.61703785,
            -0.49609608, 7.68164628, -0.60351608, 7.80839329, 7.77548828, 7.76767882,
            7.67384276, 1.34740561, -0.02453898, -0.01968331, -0.16323325, 1.38180082,
            0.23225150, 0.00651610, 1.22878564, -0.03206459, 1.43617507,
        ),
    ),
    'damped': dict(
        data_rms=0.00031069, data_max_residual=0.00177831,
        labels=(
            'O_00_00_A', 'O_00_10_A', 'O_00_11c_A', 'O_00_11s_A', 'O_00_20_A', 'O_00_21c_A',
            'O_00_21s_A', 'O_00_22c_A', 'O_00_22s_A', 'O_10_10_A', 'O_10_11c_A', 'O_10_11s_A',
            'O_10_20_A', 'O_10_21c_A', 'O_10_21s_A', 'O_10_22c_A', 'O_10_22s_A', 'O_11c_11c_A',
            'O_11c_11s_A', 'O_11c_20_A', 'O_11c_21c_A', 'O_11c_21s_A', 'O_11c_22c_A',
            'O_11c_22s_A', 'O_11s_11s_A', 'O_11s_20_A', 'O_11s_21c_A', 'O_11s_21s_A',
            'O_11s_22c_A', 'O_11s_22s_A', 'O_20_20_A', 'O_20_21c_A', 'O_20_21s_A',
            'O_20_22c_A', 'O_20_22s_A', 'O_21c_21c_A', 'O_21c_21s_A', 'O_21c_22c_A',
            'O_21c_22s_A', 'O_21s_21s_A', 'O_21s_22c_A', 'O_21s_22s_A', 'O_22c_22c_A',
            'O_22c_22s_A', 'O_22s_22s_A', 'H1_00_00_A', 'H1_00_10_A', 'H1_00_11c_A',
            'H1_00_11s_A', 'H1_10_10_A', 'H1_10_11c_A', 'H1_10_11s_A', 'H1_11c_11c_A',
            'H1_11c_11s_A', 'H1_11s_11s_A',
        ),
        parameters=(
            2.90243528, -0.00058406, -0.24117990, -0.26803240, -0.03297205, 0.18382245,
            0.52365552, -0.29182850, -0.05942092, 2.71838370, 0.13297706, -0.03593885,
            -0.66514368, -0.13899384, -0.08314363, 0.62808546, 0.34524762, 2.82397143,
            0.27420951, 0.00669522, -0.18269583, -0.28344589, -0.00282527, 0.23147050,
            2.91555141, -0.35260348, -0.33334420, -0.63489934, -0.30595907, -0.29436913,
            2.87118269, 0.23245706, -0.05041399, -0.24160025, -0.34355095, 2.83004213,
            0.18579058, -0.20937941, -0.22657305, 2.70490883, -0.10846024, -0.11929516,
            2.73785782, 0.39134631, 2.77449037, 1.44226063, 0.00154433, -0.04188488,
            -0.13457563, 1.32154582, 0.19553924, -0.01894282, 1.26332719, -0.01900601,
            1.37240753,
        ),
    ),
}
