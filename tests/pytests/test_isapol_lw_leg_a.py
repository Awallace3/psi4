# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""LW residual policy on charge-conserving synthetic tensors with injected defects.

Tests exercise gate ordering and defect transport, not archived molecular parity.
"""
import hashlib
import math

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]

#: Unmodified production postcondition for everything the algorithm controls.
ALGORITHM_TOLERANCE = 1e-6

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


def test_leg_a_default_still_rejects_the_supplied_defect():
    """The default combined gate is unchanged; nothing is silently accepted."""
    with pytest.raises(RuntimeError, match=r'postcondition exceeds residual tolerance '
                                           r'.*charge-sum=.*local-charge=.*input-sum-rule='):
        localize(synthetic_supplied(0.25))


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
    This is the policy half of the recorded eleven-frequency driver run.
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
