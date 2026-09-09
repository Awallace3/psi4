# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Leg A acceptance: the recorded distributed-to-local localization, all eleven indices.

Leg A is the localization step of the reference chain: nonlocal rank 0:4 site-pair
polarizabilities in, on-site local rank 1:3 polarizabilities out, in each site's local
frame. In the reference chain that step is performed by a GPLv3 tool whose source is
not read, quoted or transliterated anywhere in Psi4. This file gates our own
`isa_localize_lw` against that tool's recorded file-in/file-out pair, using only
committed literals: the supplied `H2O_NL4_{000..010}.pol` blocks and the recorded
`H2O_L3_{000..010}.pol` blocks, both already pinned by SHA256 in the two fixtures.

The acceptance criterion is an output comparison, not a sum-rule postcondition on the
input. The supplied pair data does not satisfy the charge-flow sum rule exactly (the
defect runs from 7.011e-04 at the static index down to 2.990e-06 at the highest node),
LW transports that defect exactly rather than repairing it, and the reference chain
localized the same imperfect data. So the input defect is measured and reported while
the four algorithm-controlled residuals are held to the unmodified production 1e-6.
That is a reclassification of two reported numbers, not a loosened tolerance: no
comparison tolerance is relaxed and the static-only 1e-3 waiver is not extended here.

Passing is localization-leg agreement on recorded reference input. It is not native
prediction, not dispersion-coefficient acceptance, and not external parity.
"""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import psi4

DATA = Path(__file__).parent / 'data_isapol/orient_local'
HERMETIC_SHA256 = 'b86d411e5fd81fc20358370ee211ba5b9bd997c57f8918c32ce0334c93ad7f49'
DYNAMIC_SHA256 = '132283408a5906e523231df9f99b1dcec2b88a29eb773b0c867e541a7eeced20'
# Output agreement floor. Measured worst over all eleven indices: 8.3e-13.
LEG_A_ATOL = 1e-11
# Unmodified production postcondition for everything the algorithm controls.
ALGORITHM_TOLERANCE = 1e-6
# Exact decimal charge-flow sum-rule defect of the SUPPLIED data, index 000..010.
# Read from the committed literals, never estimated; see test_leg_a_sum_rule_defect.
SUPPLIED_SUM_RULE = ['0.0007011', '0.0007011', '0.0007001', '0.0006940', '0.0006727',
                     '0.0006185', '0.00051040', '0.00034365', '0.00017116', '0.00006008',
                     '0.000002990']
ALGORITHM_NAMES = ('off_site', 'reciprocity', 'molecular_sum', 'charge_sum_transport')
LABELS = ('O', 'H1', 'H2')


@pytest.fixture(scope='module')
def chain():
    """The eleven recorded indices as (index, omega, supplied 9x25x25, expected 3x15x15)."""
    hermetic = (DATA / 'lw-hermetic-water.json').read_bytes()
    dynamic = (DATA / 'lw-dynamic-water.json').read_bytes()
    assert hashlib.sha256(hermetic).hexdigest() == HERMETIC_SHA256
    assert hashlib.sha256(dynamic).hexdigest() == DYNAMIC_SHA256
    h, d = json.loads(hermetic), json.loads(dynamic)
    # Both fixtures must describe the same molecule, graph and frame convention.
    assert h['sites'] == d['sites'] and h['bonds_zero_based'] == d['bonds_zero_based']
    assert h['frame_convention'] == d['frame_convention'] == 'local_to_global_columns'
    assert h['input_frame'] == d['input_frame'] == 'global'
    assert h['expected_frame'] == d['expected_frame'] == 'site_local'
    assert h['units'] == d['units'] == 'atomic' and h['geometry_units'] == d['geometry_units'] == 'bohr'
    grid = psi4.core.CasimirGrid(10, 0.5)
    records = []
    for index, node in [(0, h)] + [(n['index'], n) for n in d['nodes']]:
        supplied = np.array([s['values'] for s in node['distributed']['sections']], float)
        expected = np.array([s['values'] for s in node['expected_local']['sections']], float)
        assert supplied.shape == (9, 25, 25) and expected.shape == (3, 15, 15)
        omega = h['frequency'] if index == 0 else node['omega']
        assert omega == grid.omega(index)
        records.append({'index': index, 'omega': omega, 'supplied': supplied, 'expected': expected,
                        'sections': node['distributed']['sections'],
                        'supplied_sha256': node['distributed']['sha256'],
                        'expected_sha256': node['expected_local']['sha256']})
    assert [r['index'] for r in records] == list(range(11))
    assert records[0]['omega'] == 0.0
    return {'sites': h['sites'], 'bonds': h['bonds_zero_based'], 'records': records}


def _matrix(values):
    return psi4.core.Matrix.from_array(np.asarray(values, dtype=float))


def _rotation(frame):
    """Local-to-global column rotation on real Racah ranks 1..3."""
    return np.asarray(psi4.core.isa_multipole_rotation(3, frame))[1:16, 1:16].copy()


def _localize(chain, record, residual_tolerance=ALGORITHM_TOLERANCE,
              input_sum_rule_tolerance=math.inf, supplied=None):
    working = record['supplied'][:, :16, :16] if supplied is None else supplied[:, :16, :16]
    return psi4.core.isa_localize_lw(
        _matrix([s['origin'] for s in chain['sites']]),
        [_matrix(block) for block in working], record['omega'], chain['bonds'],
        residual_tolerance, input_sum_rule_tolerance)


@pytest.mark.parametrize('index', range(11))
def test_leg_a_recorded_localization(chain, index, record_property):
    """File-in/file-out agreement with the recorded reference localization."""
    record = chain['records'][index]
    result = _localize(chain, record)
    assert result.frequency == record['omega']
    np.testing.assert_array_equal(result.positions, [s['origin'] for s in chain['sites']])

    # Every residual LW is responsible for, at the unmodified production tolerance.
    algorithm = {name: float(getattr(result.residuals, name)) for name in ALGORITHM_NAMES}
    assert max(algorithm.values()) <= ALGORITHM_TOLERANCE
    # The bond transfers are antisymmetric in the site slots, so the charge-flow
    # sums are invariants; anything above rounding here would be a real defect.
    assert algorithm['charge_sum_transport'] <= 1e-15

    # Rank 0 is dropped on output, and the comparison is in each site's local frame.
    actual_global = np.array([np.asarray(block) for block in result.local])
    assert actual_global.shape == (3, 15, 15)
    rotations = [_rotation(s['frame']) for s in chain['sites']]
    actual = np.array([d.T @ a @ d for d, a in zip(rotations, actual_global)])
    errors = np.abs(actual - record['expected'])
    assert np.isfinite(errors).all() and errors.size == 675

    record_property('leg_a_index', index)
    record_property('leg_a_frequency', record['omega'])
    record_property('leg_a_supplied_sha256', record['supplied_sha256'])
    record_property('leg_a_expected_sha256', record['expected_sha256'])
    record_property('leg_a_maxabs', float(errors.max()))
    record_property('leg_a_site_maxabs', json.dumps(dict(zip(LABELS, errors.max(axis=(1, 2)).tolist()))))
    record_property('leg_a_algorithm_residuals', json.dumps(algorithm, sort_keys=True))
    record_property('leg_a_supplied_sum_rule', float(result.residuals.input_sum_rule))
    record_property('leg_a_transfers', len(result.transfers))
    np.testing.assert_allclose(actual, record['expected'], atol=LEG_A_ATOL, rtol=0)

    # Negative frame control: the unrotated H1 block is nowhere near the recorded one,
    # so the agreement above is not an artefact of a trivial rotation. Scale-free,
    # because the tensor magnitudes fall by four orders of magnitude across the grid.
    unrotated = np.max(np.abs(actual_global[1] - record['expected'][1]))
    assert unrotated > 0.01 * np.max(np.abs(record['expected'][1]))
    assert unrotated > 1.0e6 * LEG_A_ATOL


@pytest.mark.parametrize('index', range(11))
def test_leg_a_sum_rule_defect(chain, index):
    """The supplied defect is measured exactly, reported, and transported unrepaired."""
    record = chain['records'][index]
    values = [s['values'] for s in record['sections']]
    # Exact decimal arithmetic on the committed seven-digit literals: no float rounding.
    sums = [sum(Decimal(values[3*a+b][k][0]) for b in range(3)) for a in range(3) for k in range(16)]
    worst = max(sums, key=abs)
    assert abs(worst) == Decimal(SUPPLIED_SUM_RULE[index])
    assert abs(worst) > Decimal('0.000001')  # the supplied data really does violate it

    result = _localize(chain, record)
    residuals = result.residuals
    # The reported defect is the supplied one, to the precision of the printed input.
    assert float(residuals.input_sum_rule) == pytest.approx(float(abs(worst)), abs=1e-12, rel=0)
    # Off-site blocks are annihilated, so both legacy names reproduce that same defect.
    assert float(residuals.charge_sum) == pytest.approx(float(residuals.input_sum_rule), abs=1e-15, rel=0)
    assert float(residuals.local_charge) == pytest.approx(float(residuals.input_sum_rule), abs=1e-15, rel=0)
    # Reporting is not repairing: the defect is not reduced by localization.
    assert float(residuals.local_charge) > ALGORITHM_TOLERANCE if index < 10 else True
    assert float(residuals.charge_sum_transport) <= 1e-15


def test_leg_a_defect_measurement_tracks_the_input(chain):
    """Negative control: input_sum_rule is measured from the input, not a constant."""
    record = chain['records'][0]
    baseline = _localize(chain, record).residuals
    injected = 0.25
    perturbed = record['supplied'].copy()
    # Add a known amount to one charge-flow entry of the O-O block, and to its
    # reciprocal partner so the separate input-reciprocity precondition still holds.
    perturbed[0, 5, 0] += injected
    perturbed[0, 0, 5] += injected
    # Predict the new defect from the perturbed input rather than assuming a value.
    blocks = perturbed[:, :16, :16].reshape(3, 3, 16, 16)
    predicted = max(np.abs(blocks.sum(axis=1)[:, :, 0]).max(),
                    np.abs(blocks.sum(axis=0)[:, 0, :]).max())
    assert predicted > float(baseline.input_sum_rule)

    moved = _localize(chain, record, supplied=perturbed).residuals
    assert float(moved.input_sum_rule) == pytest.approx(predicted, abs=1e-12, rel=0)
    assert float(moved.input_sum_rule) == pytest.approx(injected, abs=1e-3, rel=0)
    # Still transported exactly, and still reproduced by the two legacy names.
    assert float(moved.charge_sum_transport) <= 1e-14
    assert float(moved.charge_sum) == pytest.approx(predicted, abs=1e-12, rel=0)


@pytest.mark.parametrize('index', range(11))
def test_leg_a_default_still_rejects_the_supplied_defect(chain, index):
    """The default combined gate is unchanged; nothing is silently accepted."""
    record = chain['records'][index]
    with pytest.raises(RuntimeError, match=r'postcondition exceeds residual tolerance '
                                           r'.*charge-sum=.*local-charge=.*input-sum-rule='):
        psi4.core.isa_localize_lw(
            _matrix([s['origin'] for s in chain['sites']]),
            [_matrix(block) for block in record['supplied'][:, :16, :16]],
            record['omega'], chain['bonds'])


def test_leg_a_reporting_does_not_loosen_the_algorithm_gate(chain):
    """Reporting the input defect leaves residual_tolerance in force."""
    record = chain['records'][0]
    # At the static index the input reciprocity is 5.77e-12 and off_site is 2.20e-11,
    # so a 1e-11 request clears the input precondition and must then be refused by the
    # postcondition, even though the input sum rule is reported rather than gated.
    with pytest.raises(RuntimeError, match=r'postcondition exceeds residual tolerance'):
        _localize(chain, record, residual_tolerance=1e-11)
    # A tolerance below the supplied input's own reciprocity error is refused earlier,
    # by the input precondition, which the reporting policy also leaves in force.
    with pytest.raises(RuntimeError, match=r'input reciprocity exceeds residual tolerance'):
        _localize(chain, record, residual_tolerance=1e-14)
    # ... and the same call at the production tolerance is admitted.
    assert _localize(chain, record, residual_tolerance=ALGORITHM_TOLERANCE) is not None


@pytest.mark.parametrize('tolerance', [float('nan'), 0.0])
def test_leg_a_rejects_meaningless_sum_rule_tolerance(chain, tolerance):
    with pytest.raises(RuntimeError, match='input sum-rule tolerance'):
        _localize(chain, chain['records'][0], input_sum_rule_tolerance=tolerance)


def test_leg_a_explicit_finite_sum_rule_tolerance(chain):
    """A positive finite threshold gates the supplied defect on its own."""
    record = chain['records'][0]
    defect = float(Decimal(SUPPLIED_SUM_RULE[0]))
    assert _localize(chain, record, input_sum_rule_tolerance=defect * 2) is not None
    with pytest.raises(RuntimeError, match='input-sum-rule='):
        _localize(chain, record, input_sum_rule_tolerance=defect / 2)


def test_leg_a_through_the_production_driver(chain, record_property):
    """The whole eleven-frequency chain through the supported driver policy."""
    from psi4.driver.procrouting import isapol_lw

    sites, records = chain['sites'], chain['records']
    tensors = np.array([r['supplied'].reshape(3, 3, 25, 25) for r in records])
    assert tensors.shape == (11, 3, 3, 25, 25)
    model = isapol_lw.supplied_nonlocal_properties(
        labels=[s['label'] for s in sites],
        origins=[s['origin'] for s in sites],
        bonds=[tuple(b) for b in chain['bonds']],
        frequencies=[r['omega'] for r in records],
        tensors=tensors, input_rank=4, truncation=isapol_lw.TRUNCATE_RANK4,
        frames=[s['frame'] for s in sites],
        residual_policy='reported_input_sum_rule',
        provenance=isapol_lw.Provenance(
            source_name='recorded reference distributed polarizabilities, indices 000-010',
            source_sha256=records[0]['supplied_sha256'], producer='reference chain (recorded literals)',
            description='committed fixture literals only; no external path or executable is used'))
    assert model.metadata.residual_policy == 'reported_input_sum_rule'
    assert model.metadata.residual_tolerance == isapol_lw.PRODUCTION_TOLERANCE
    assert model.metadata.historical_fixture_sha256 is None
    # The supplied defect keeps the combined flag False and is warned about ...
    assert model.metadata.production_postcondition_passed is False
    assert all(d.production_postcondition_passed is False for d in model.frequency_diagnostics)
    # ... while everything the algorithm owns passed at 1e-6 at every frequency.
    assert all(d.algorithm_postcondition_passed is True for d in model.frequency_diagnostics)
    assert any('charge-flow sum-rule defect' in w for w in model.warnings)
    assert any('reported, not gated' in w for w in model.warnings)

    local = model.raw_local.array
    expected = np.array([r['expected'] for r in records])
    assert local.shape == expected.shape == (11, 3, 15, 15)
    errors = np.abs(local - expected)
    record_property('leg_a_driver_maxabs', float(errors.max()))
    record_property('leg_a_driver_entries', int(errors.size))
    record_property('leg_a_driver_sum_rule',
                    json.dumps([d.residuals.input_sum_rule for d in model.frequency_diagnostics]))
    print('leg A driver chain', json.dumps({'entries': int(errors.size), 'maxabs': float(errors.max()),
                                            'worst_index': int(np.argmax(errors.max(axis=(1, 2, 3))))}))
    assert errors.size == 7425
    np.testing.assert_allclose(local, expected, atol=LEG_A_ATOL, rtol=0)


def test_leg_a_production_policy_is_unreachable_for_this_input(chain):
    """The strict single-gate policy still refuses the recorded reference input."""
    from psi4.driver.procrouting import isapol_lw

    sites, records = chain['sites'], chain['records'][:1]
    with pytest.raises(RuntimeError, match='postcondition exceeds residual tolerance'):
        isapol_lw.supplied_nonlocal_properties(
            labels=[s['label'] for s in sites], origins=[s['origin'] for s in sites],
            bonds=[tuple(b) for b in chain['bonds']], frequencies=[records[0]['omega']],
            tensors=np.array([records[0]['supplied'].reshape(3, 3, 25, 25)]),
            input_rank=4, truncation=isapol_lw.TRUNCATE_RANK4,
            frames=[s['frame'] for s in sites], residual_policy='production',
            provenance=isapol_lw.Provenance(
                source_name='recorded reference distributed polarizabilities, index 000',
                source_sha256=records[0]['supplied_sha256'], producer='reference chain (recorded literals)',
                description='committed fixture literals only'))
