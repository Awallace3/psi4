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
import numpy as np
import pytest

from psi4 import core
from psi4.driver.procrouting import isapol_refine as R

pytestmark = [pytest.mark.smoke]

IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
#: H1's local frame is the C2v image of H2's, the signed permutation
#: diag(-1,-1,1).  This is what CamCASP's own ``Axes`` section builds for
#: ``H1 z global z x global -x``, and it is the frame that makes the two
#: hydrogens share one set of variables with no sign changes.
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
