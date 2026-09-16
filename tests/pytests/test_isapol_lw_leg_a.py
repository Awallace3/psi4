# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Leg A: the residual policy of `isa_localize_lw`, and what the recorded run measured.

Leg A is the localization step of the reference chain: nonlocal rank 0:4 site-pair
polarizabilities in, on-site local rank 1:3 polarizabilities out, in each site's
local frame.  In the reference chain that step is performed by a GPLv3 tool whose
source is not read, quoted or transliterated anywhere in Psi4.

*Why the acceptance comparison is not here.*  Localization couples every site and
every component, so the step cannot be run on a subset of its input: reproducing
the recorded output at even one matrix element requires all 9 x 25 x 25 supplied
numbers.  There is no reduced numerical slice of this stage, so the eleven-index
file-in/file-out comparison -- 7,425 entries, agreeing to 8.3e-13 -- lives in
`agent_scratch/pytests/test_isapol_lw_leg_a.py` with its two large fixtures.  What
that run *measured* is recorded below as literals, and the scratch guard re-derives
every one of them so a regenerated capture cannot leave stale numbers committed.

*What is here, and is stronger than the fixture version.*  The residual policy
itself -- which defects are reported, which are gated, and in which order the
input precondition and the output postcondition fire -- is tested on a synthetic
three-site model built to conserve charge exactly, into which a *known* defect is
then injected.  The recorded reference input violates the charge-flow sum rule by
an amount nobody chose; here the amount is chosen, so `input_sum_rule` can be
checked against the value it must report rather than against itself.
"""
from decimal import Decimal
import hashlib
import math

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

#: Unmodified production postcondition for everything the algorithm controls.
ALGORITHM_TOLERANCE = 1e-6
#: Output agreement floor used by the recorded comparison in `agent_scratch/`.
LEG_A_ATOL = 1e-11

#: The reference chain's declared quadrature, `Quad 10 / Beta 0.5`: the static
#: point plus ten dynamic nodes.  Both fixtures declare these frequencies, and
#: `test_leg_a_frequency_grid_is_the_declared_quadrature` shows they are exactly
#: our own grid's, which is what makes the recorded indices ours to reproduce.
QUAD, BETA = 10, 0.5
CASIMIR_OMEGAS = [0.0, 0.006609601596087073, 0.036174811998631026, 0.09544736369034741,
                  0.1976442118453102, 0.3704172128053662, 0.6749146404580318,
                  1.2648991724365144, 2.619244684547348, 6.91088595040828,
                  37.82376235021513]

#: Exact decimal charge-flow sum-rule defect of the SUPPLIED reference data,
#: indices 000..010, read from the printed seven-digit tokens with no float round
#: trip.  LW transports this defect exactly rather than repairing it, and the
#: reference chain localized the same imperfect data, so it is measured and
#: reported while the four algorithm-controlled residuals stay at 1e-6.  It is
#: above that 1e-6 at every one of the eleven indices, the smallest by threefold.
SUPPLIED_SUM_RULE = ['0.0007011', '0.0007011', '0.0007001', '0.0006940', '0.0006727',
                     '0.0006185', '0.00051040', '0.00034365', '0.00017116', '0.00006008',
                     '0.000002990']

#: What the recorded eleven-index comparison measured: worst absolute disagreement
#: with the reference's localized output, and the largest entry of that output, per
#: index.  The magnitudes fall by three orders of magnitude across the grid, so the
#: agreement is only meaningful as a ratio -- see the test below.
LEG_A_MAXABS = [6.252776074688882e-13, 8.242295734817162e-13, 5.684341886080801e-13,
                6.252776074688882e-13, 7.389644451905042e-13, 5.204725539442734e-13,
                6.536993168992922e-13, 5.186961971048731e-13, 5.226929999935237e-13,
                4.989342272665453e-13, 4.986939680651226e-13]
REFERENCE_SCALE = [220.666588197109, 220.636741690751, 219.776119734384, 214.51305638687,
                   195.857199661158, 153.073961018694, 102.585584108613, 55.395783335634,
                   21.036525673816, 4.101250616307, 0.151456947268]
#: Number of bond transfers LW performs on this three-site, two-bond graph at rank 4,
#: identical at every index.
LEG_A_TRANSFERS = 456

ORIGINS = [[0.0, 0.0, 0.0], [-1.45365196, 0.0, -1.12168732], [1.45365196, 0.0, -1.12168732]]
BONDS = [[0, 1], [0, 2]]
LABELS = ('O', 'H1', 'H2')
ALGORITHM_NAMES = ('off_site', 'reciprocity', 'molecular_sum', 'charge_sum_transport')

#: Charge column the synthetic defect is injected into: a rank-2 component, so the
#: defect is not confined to the charge-charge element that a rank-0 test would reach.
DEFECT_COMPONENT = 5


def _matrix(values):
    return psi4.core.Matrix.from_array(np.asarray(values, dtype=float))


def synthetic_supplied(defect=0.0, site=0, reciprocity_error=0.0):
    """A three-site rank-3 supplied model that conserves charge exactly, plus a defect.

    Deterministic and analytic: no fixture, no RNG.  Off-site blocks are laid down
    first and mirrored so the input is exactly reciprocal, `T[a,b] == T[b,a].T`;
    each on-site block's charge row and column are then set to minus the sum of that
    site's off-site ones, which is precisely the charge-flow sum rule.  `defect` is
    added to one on-site charge entry and its reciprocal partner, so the model's
    sum-rule violation is exactly `defect` and nothing else about it moves.

    The tensor is deliberately synthetic and must never be quoted as a polarizability.
    """
    n, nc = 3, 16
    i = np.arange(nc)
    tensors = np.zeros((n, n, nc, nc))
    for a in range(n):
        for b in range(n):
            if a != b:
                tensors[a, b] = np.cos(i[:, None] + 2.0*i[None, :] + 3.0*a + 5.0*b)*(1.0 + 0.1*(a + b))
    for a in range(n):
        for b in range(a + 1, n):
            tensors[b, a] = tensors[a, b].T
    for a in range(n):
        block = np.cos(0.7*i[:, None] + 1.3*i[None, :] + 2.0*a)
        tensors[a, a] = 0.5*(block + block.T)
        for k in range(nc):
            flow = sum(tensors[a, b][k, 0] for b in range(n) if b != a)
            tensors[a, a][k, 0] = tensors[a, a][0, k] = -flow
    tensors[site, site][DEFECT_COMPONENT, 0] += defect
    tensors[site, site][0, DEFECT_COMPONENT] += defect
    if reciprocity_error:
        tensors[0, 1][2, 3] += reciprocity_error
    return tensors


def localize(tensors, *args):
    return psi4.core.isa_localize_lw(
        _matrix(ORIGINS), [_matrix(tensors[a, b]) for a in range(3) for b in range(3)],
        0.0, BONDS, *args)


def test_leg_a_frequency_grid_is_the_declared_quadrature():
    """The eleven recorded indices are exactly our own `CasimirGrid(10, 0.5)` nodes.

    Both fixtures declare an `omega` per index; this is the check that those are
    our grid's and not merely near it, which is what makes the recorded output
    something our code can be held to index by index.  The nodes come in reciprocal
    pairs about `omega0` by construction of the `omega0 (1 -+ t)/(1 +- t)` map.
    """
    grid = psi4.core.CasimirGrid(QUAD, BETA)
    assert grid.n_freq() == QUAD and grid.omega0() == BETA
    assert [grid.omega(k) for k in range(QUAD + 1)] == CASIMIR_OMEGAS
    assert CASIMIR_OMEGAS[0] == 0.0
    for k in range(1, QUAD//2 + 1):
        assert CASIMIR_OMEGAS[k]*CASIMIR_OMEGAS[QUAD - k + 1] == pytest.approx(BETA**2, rel=1e-14)


def test_leg_a_supplied_defect_decays_and_exceeds_the_production_tolerance():
    """The supplied data really does violate the sum rule, by a decaying amount.

    Exact decimal arithmetic on the printed tokens, so this is a statement about
    the reference input and not about float rounding.  It stays above the
    unmodified 1e-6 production postcondition at every one of the eleven indices,
    which is why the recorded comparison needs the reporting policy rather than a
    loosened gate: there is no index at which the default would have passed.
    """
    defects = [Decimal(value) for value in SUPPLIED_SUM_RULE]
    assert len(defects) == QUAD + 1
    assert defects == sorted(defects, reverse=True)
    assert defects[0] == defects[1]  # the static point and the first node agree to print
    tolerance = Decimal('0.000001')
    assert all(defect > tolerance for defect in defects)
    assert defects[QUAD]/tolerance > 2  # the weakest index still clears it threefold
    assert defects[0]/defects[QUAD] > 200


def test_leg_a_recorded_agreement_is_at_double_precision_round_off():
    """The recorded agreement is not merely inside `LEG_A_ATOL`; it is round-off.

    The reference's localized entries fall by three orders of magnitude across the
    grid, so a single absolute floor would flatter the high nodes.  The absolute
    disagreement, by contrast, is flat at ~5e-13 -- it does not track the tensor
    magnitude at all, which is the signature of accumulated round-off rather than
    of a systematic difference in what the two codes compute.  Normalized on each
    index's own largest entry the worst of the eleven is 3.3e-12, at the smallest
    node, and 3.7e-15 at the static point.
    """
    assert len(LEG_A_MAXABS) == len(REFERENCE_SCALE) == QUAD + 1
    for index, (error, scale) in enumerate(zip(LEG_A_MAXABS, REFERENCE_SCALE)):
        assert error < LEG_A_ATOL/10, index
        assert error/scale < 1e-11, index
    # Flat in absolute terms across a grid whose tensors span three decades: the
    # spread of the errors is far smaller than the spread of what they measure.
    assert max(LEG_A_MAXABS)/min(LEG_A_MAXABS) < 2
    assert REFERENCE_SCALE[0]/REFERENCE_SCALE[QUAD] > 1000
    # Which means the tightest index is the smallest one, not the largest.
    assert np.argmax(np.array(LEG_A_MAXABS)/np.array(REFERENCE_SCALE)) == QUAD


def test_charge_conserving_supplied_model_passes_the_production_gate():
    """The synthetic baseline: every residual the algorithm owns, at 1e-6, by default."""
    tensors = synthetic_supplied()
    # The construction really is exactly reciprocal and exactly charge conserving.
    for a in range(3):
        for b in range(3):
            np.testing.assert_array_equal(tensors[a, b], tensors[b, a].T)
        for k in range(16):
            assert abs(sum(tensors[a, b][k, 0] for b in range(3))) < 1e-15

    result = localize(tensors)
    assert result.frequency == 0.0
    np.testing.assert_array_equal(result.positions, ORIGINS)
    residuals = {name: float(getattr(result.residuals, name)) for name in ALGORITHM_NAMES}
    assert max(residuals.values()) <= ALGORITHM_TOLERANCE
    # The bond transfers are antisymmetric in the site slots, so the charge-flow
    # sums are invariants; anything above rounding here would be a real defect.
    assert residuals['charge_sum_transport'] <= 1e-15
    assert float(result.residuals.input_sum_rule) <= 1e-15
    assert len(result.transfers) > 0


@pytest.mark.parametrize('defect', [0.25, 0.5])
def test_leg_a_defect_measurement_tracks_the_input(defect):
    """`input_sum_rule` reports the injected amount, and is not a constant.

    This is what the recorded fixture can only assert against itself: the defect
    here is chosen, so the reported number is checked against the value it must
    report.  Both legacy names reproduce it, because localization annihilates the
    off-site blocks that carried it.
    """
    result = localize(synthetic_supplied(defect), ALGORITHM_TOLERANCE, math.inf)
    residuals = result.residuals
    assert float(residuals.input_sum_rule) == pytest.approx(defect, abs=1e-14, rel=0)
    assert float(residuals.charge_sum) == pytest.approx(defect, abs=1e-14, rel=0)
    assert float(residuals.local_charge) == pytest.approx(defect, abs=1e-14, rel=0)
    # Reporting is not repairing: the defect is not reduced by localization ...
    assert float(residuals.local_charge) > ALGORITHM_TOLERANCE
    # ... and it is transported exactly, so the transport residual stays at rounding.
    assert float(residuals.charge_sum_transport) <= 1e-14
    # Everything the algorithm owns is untouched by the defect.
    for name in ('off_site', 'reciprocity', 'molecular_sum'):
        assert float(getattr(residuals, name)) <= 1e-12, name


@pytest.mark.parametrize('defect', [0.25, 0.5])
def test_leg_a_default_still_rejects_the_supplied_defect(defect):
    """The default combined gate is unchanged; nothing is silently accepted."""
    with pytest.raises(RuntimeError, match=r'postcondition exceeds residual tolerance '
                                           r'.*charge-sum=.*local-charge=.*input-sum-rule='):
        localize(synthetic_supplied(defect))


def test_leg_a_reporting_does_not_loosen_the_three_ordered_gates():
    """Reporting the input defect leaves every `residual_tolerance` gate in force.

    `residual_tolerance` is checked at three points, and this shows all three are
    live and which one wins when more than one would fire:

    1. the *input* precondition, on the supplied model's reciprocity, before any
       work is done;
    2. the graph solve, on the bond-flow linear system's own residual;
    3. the output postcondition, on the four algorithm-controlled residuals.

    An exactly reciprocal synthetic skips (1), so tightening past the solve's own
    accuracy reaches (2) and then (3).  Adding a deliberate 1e-12 input asymmetry
    makes (1) fire at the same tolerance that gave (2) before -- which is the proof
    that the input check runs first rather than being folded into the output one.
    """
    clean = synthetic_supplied()
    # (3): loose enough for the solve, tight enough to fail the output residuals.
    with pytest.raises(RuntimeError, match=r'postcondition exceeds residual tolerance'):
        localize(clean, 1e-13, math.inf)
    # (2): tighter than the solve can achieve, so it never reaches the output.
    with pytest.raises(RuntimeError, match=r'graph solve exceeds residual tolerance'):
        localize(clean, 1e-16, math.inf)
    # (1): the same 1e-16 now stops at the input instead, and so does 1e-14, which
    # on the clean model reached the solve.
    skewed = synthetic_supplied(reciprocity_error=1e-12)
    for tolerance in (1e-16, 1e-14):
        with pytest.raises(RuntimeError, match=r'input reciprocity exceeds residual tolerance'):
            localize(skewed, tolerance, math.inf)
    # The precondition is a real measurement, not a blanket refusal of asymmetry:
    # 1e-12 of it is admitted at 1e-6 and 1e-3 of it is not.
    assert localize(skewed, ALGORITHM_TOLERANCE, math.inf) is not None
    with pytest.raises(RuntimeError, match=r'input reciprocity exceeds residual tolerance'):
        localize(synthetic_supplied(reciprocity_error=1e-3), ALGORITHM_TOLERANCE, math.inf)


@pytest.mark.parametrize('tolerance', [float('nan'), 0.0])
def test_leg_a_rejects_meaningless_sum_rule_tolerance(tolerance):
    with pytest.raises(RuntimeError, match='input sum-rule tolerance'):
        localize(synthetic_supplied(), ALGORITHM_TOLERANCE, tolerance)


def test_leg_a_explicit_finite_sum_rule_tolerance():
    """A positive finite threshold gates the supplied defect on its own."""
    defect = 0.25
    tensors = synthetic_supplied(defect)
    assert localize(tensors, ALGORITHM_TOLERANCE, defect*2) is not None
    with pytest.raises(RuntimeError, match='input-sum-rule='):
        localize(tensors, ALGORITHM_TOLERANCE, defect/2)


def test_leg_a_policy_through_the_production_driver():
    """The same two policies through the supported driver, on the synthetic model.

    `reported_input_sum_rule` warns and lowers the combined flag while leaving the
    algorithm's own postcondition passing at 1e-6; `production` refuses outright.
    This is the policy half of the recorded eleven-frequency driver run, which
    itself stays in `agent_scratch/` with the fixtures it compares against.
    """
    from psi4.driver.procrouting import isapol_lw

    tensors = synthetic_supplied(0.25).reshape(1, 3, 3, 16, 16)
    common = dict(labels=list(LABELS), origins=ORIGINS, bonds=[tuple(b) for b in BONDS],
                  frequencies=[0.0], tensors=tensors, input_rank=3,
                  frames=np.array([np.eye(3)]*3),
                  provenance=isapol_lw.Provenance(
                      source_name='synthetic charge-flow defect, analytic',
                      source_sha256=hashlib.sha256(np.ascontiguousarray(tensors).tobytes()).hexdigest(),
                      producer='test_isapol_lw_leg_a.synthetic_supplied',
                      description='deterministic analytic model; never a polarizability'))

    model = isapol_lw.supplied_nonlocal_properties(residual_policy='reported_input_sum_rule', **common)
    assert model.metadata.residual_policy == 'reported_input_sum_rule'
    assert model.metadata.residual_tolerance == isapol_lw.PRODUCTION_TOLERANCE
    assert model.metadata.historical_fixture_sha256 is None
    # The supplied defect keeps the combined flag False and is warned about ...
    assert model.metadata.production_postcondition_passed is False
    assert all(d.production_postcondition_passed is False for d in model.frequency_diagnostics)
    # ... while everything the algorithm owns passed at 1e-6.
    assert all(d.algorithm_postcondition_passed is True for d in model.frequency_diagnostics)
    assert any('charge-flow sum-rule defect' in w for w in model.warnings)
    assert any('reported, not gated' in w for w in model.warnings)
    assert model.frequency_diagnostics[0].residuals.input_sum_rule == pytest.approx(0.25, abs=1e-14)

    with pytest.raises(RuntimeError, match='postcondition exceeds residual tolerance'):
        isapol_lw.supplied_nonlocal_properties(residual_policy='production', **common)
