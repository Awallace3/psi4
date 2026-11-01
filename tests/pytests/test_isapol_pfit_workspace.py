"""Finite PFIT admission and tiny synthetic algebra; no molecular SCF."""
import json

import numpy as np
import pytest

from psi4.driver.procrouting import isapol_pfit_plan as plan


@pytest.mark.parametrize('value', [True, 1., float('inf'), 0, -1, '1024',
                                  plan.MAX_NATIVE_WORK_BYTES+1])
def test_native_workspace_requires_representable_positive_integer(value):
    with pytest.raises(ValueError, match='workspace budget'):
        plan.native_work_limit(value)


def test_legacy_default_and_large_native_base_are_not_silently_raised():
    assert plan.native_work_limit() == 256*1024**2
    for count, expected, cap in ((432, 688_747_264, 1024**3),
                                  (612, 1_363_037_344, 2*1024**3)):
        with pytest.raises(ValueError, match='native PFIT workspace budget'):
            plan.native_row_plan(2000, count, 11)
        explicit = plan.native_row_plan(2000, count, 11, maximum_work_bytes=cap)
        assert explicit['native_base_bytes'] == expected
        assert explicit['native_work_bytes'] == cap
        with pytest.raises(ValueError, match='below kernel base'):
            plan.native_row_plan(2000, count, 11, maximum_work_bytes=expected-1)
        # Base-only acceptance is conditional: LAPACK may need more at runtime.
        assert plan.native_row_plan(2000, count, 11,
            maximum_work_bytes=expected)['native_base_bytes'] == expected


def test_combined_plan_reserves_entire_cap_beside_python_and_live_ledger():
    from psi4.driver.procrouting.isapol_bounded_response import BoundedResources, _Ledger
    fit = plan.combined_fit_plan(2000, 108, 432, 11, maximum_work_bytes=1024**3)
    python, work = plan.fit_envelope(2000, 108, 432, 11)
    assert fit['complete_refinement_bytes'] == python+1024**3
    assert fit['complete_refinement_work'] == work
    numeric = fit['complete_refinement_bytes']
    reserved = 12345
    exact = _Ledger(BoundedResources(numeric+reserved, work+7, 1024**3, capacity='large'),
                    reserved=reserved)
    exact.admit('earlier stage', 0, 7)
    exact.admit('complete-cloud refinement', numeric, work)
    assert exact.peak == numeric+reserved and exact.work == work+7
    for byte_short, work_short in ((1, 0), (0, 1)):
        short = _Ledger(BoundedResources(numeric+reserved-byte_short, work+7-work_short,
                                        1024**3, capacity='large'), reserved=reserved)
        short.admit('earlier stage', 0, 7)
        with pytest.raises(ValueError, match='resource limit'):
            short.admit('complete-cloud refinement', numeric, work)
        assert short.work == 7


def test_generic_native_base_includes_unclamped_blocks_qr_copy_and_predictions():
    n, k, t, b, q = 2, 3, 2, 4096, 17
    rows = n*(n+1)//2
    source = 8*(b*(k+t)+k)
    base = 8*(t*(40*k*k+64*k+16)+b*(k+2*t))+source
    normal = plan.native_row_plan(n, k, t, block_rows=b)
    assert normal['native_base_bytes'] == base
    assert normal['native_block_rows'] == b > rows
    extra = plan.native_row_plan(n, k, t, block_rows=b, solver='StreamingQR',
                                 qr_chunk_rows=q, retain_pair_predictions=True)
    assert extra['native_base_bytes']-base == 8*(t*rows+q*(k+t)+q*(k+1))


def _synthetic_model(sites=1):
    from psi4.driver.procrouting import isapol_refine as refine
    frame = tuple(map(tuple, np.eye(3)))
    atoms = tuple(refine.RefinementSite(f'H{i}', f'H{i}', (float(i), 0., 0.), frame,
                                       1 if sites == 1 else 2) for i in range(sites))
    anchor = np.ones((4 if sites == 1 else 9,)*2)
    anchor[0, :] = anchor[:, 0] = 0.
    return refine.refinement_model(atoms, [anchor.copy() for _ in atoms], cutoff=1e-4,
                                   weight_type=4, weight_coefficient=1e-3,
                                   provenance='synthetic workspace regression only')


def test_large_stream_refuses_before_targets_fields_rows_or_native_dispatch(monkeypatch):
    from psi4 import core
    from psi4.driver.procrouting import isapol_pfit_stream as stream
    model = _synthetic_model(12)
    assert model.parameter_count == 432
    points = np.zeros((2000, 3))
    def forbidden(*args, **kwargs):
        raise AssertionError('large fit must refuse before target/field/design construction')
    class Targets:
        def __array__(self, *args, **kwargs):
            forbidden()
    monkeypatch.setattr(stream._refine, 'channel_fields', forbidden)
    monkeypatch.setattr(stream, 'PackedDesignRows', forbidden)
    monkeypatch.setattr(core, 'isa_pfit_solve_rows_multi', forbidden)
    with pytest.raises(ValueError, match='native PFIT workspace budget'):
        stream.refine_streamed((model,)*11, points, Targets(),
            target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
            source_id='synthetic', generation_record='no molecular inputs')


@pytest.mark.parametrize('solver', ['NormalEquationsDSYSV', 'StreamingQR'])
def test_tiny_real_native_fit_forwards_exact_cap_and_snapshots_options(monkeypatch, solver):
    from psi4 import core
    from psi4.driver.procrouting import isapol_pfit_stream as stream
    model = _synthetic_model()
    points = np.zeros((5, 3))
    fields = np.random.default_rng(23).normal(size=(5, model.channel_count))
    targets = np.random.default_rng(24).normal(size=(15, 1))
    options = core.IsaPfitOptions()
    options.solver = getattr(core.IsaPfitSolver, solver)
    options.maximum_work_bytes = 1024**2
    kwargs = dict(target_origin=core.IsaPfitTargetOrigin.SyntheticAnalyticTest,
                  source_id='tiny synthetic', generation_record='no molecular inputs',
                  fields=fields, block_rows=4, options=options)
    expected = stream.refine_streamed([model], points, targets, **kwargs)
    native = core.isa_pfit_solve_rows_multi
    seen = []
    def dispatch(problems, blocks, copied):
        seen.append(copied.maximum_work_bytes)
        options.maximum_work_bytes = 1
        options.retain_pair_predictions = True
        return native(problems, blocks, copied)
    monkeypatch.setattr(core, 'isa_pfit_solve_rows_multi', dispatch)
    actual = stream.refine_streamed([model], points, targets, **kwargs)
    assert seen == [1024**2]  # one dispatch, no cap escalation/retry
    np.testing.assert_array_equal(actual[0].parameters, expected[0].parameters)
    assert not actual[0].diagnostics.native_verified


def test_version3_explicit_cap_roundtrip_and_legacy_field_set():
    from psi4.driver.procrouting.isapol_molecular import MolecularDeclaration, PROFILE
    from psi4.driver.procrouting.isapol_schema import NativeMonomerRequest
    declaration = MolecularDeclaration(PROFILE, (np.eye(3),)*12, ())
    data = dict(schema='native-monomer-request/3', point_group='c1',
                declaration=declaration.as_dict(),
                resources=dict(max_bytes=64*1024**3, max_work=100_000_000_000_000,
                               max_io_bytes=64*1024**3, capacity='large'),
                max_replay_work=10**12, pfit_maximum_work_bytes=1024**3,
                implementation_sha256=dict(core='0'*64, isapol_python='1'*64))
    assert NativeMonomerRequest.from_dict(json.loads(json.dumps(data))).as_dict() == data
    with pytest.raises(ValueError, match='versioned'):
        NativeMonomerRequest.from_dict(dict(data, schema='native-monomer-request/2'))
    missing = {k: v for k, v in data.items() if k != 'pfit_maximum_work_bytes'}
    with pytest.raises(ValueError, match='versioned'):
        NativeMonomerRequest.from_dict(missing)
    with pytest.raises(ValueError, match='workspace budget'):
        NativeMonomerRequest.from_dict(dict(missing, schema='native-monomer-request/2'))
    for value in (None, True, float('inf'), -1, plan.MAX_NATIVE_WORK_BYTES+1):
        with pytest.raises(ValueError):
            NativeMonomerRequest.from_dict(dict(data, pfit_maximum_work_bytes=value))


def test_bounded_gate_precedes_packed_allocation_targets_and_later_solves(monkeypatch):
    from types import SimpleNamespace as NS
    from psi4 import core
    from psi4.driver.procrouting import isapol_bounded as bounded
    from psi4.driver.procrouting.isapol_bounded_response import BoundedResources, _Ledger
    from psi4.driver.procrouting.isapol_logging import StageLog
    model = _synthetic_model(12)
    resources = BoundedResources(64*1024**3, 100*10**12, 64*1024**3, capacity='large')
    calls = []
    class Runner:
        def __init__(self, *args, **kwargs):
            self.ledger, self.auxiliary = _Ledger(resources), NS(nfunction=1)
            self.state_sha256 = self.auxiliary_sha256 = self.grid_sha256 = 'synthetic'
            self.model, self.correction = 'synthetic only', None
            self.lw_residual_policy = self.lw_disclosure = 'synthetic only'
            self.partition, self.provenance, self.stability = None, None, None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def prepare(self, *args):
            pass
        def solve(self, omega):
            calls.append(omega)
            assert len(calls) == 1, 'insufficient cap must prevent subsequent solves'
            return NS(target_response=np.zeros((1, 1)), nonlocal_response=np.zeros((300, 300)),
                      residual=0., diagnostics={})
    class Cloud:
        def __init__(self, *args):
            pass
        def x(self): return np.zeros(2000)
        y = z = x
    local = NS(raw_local=NS(array=np.zeros((1, 12, 24, 24))),
               frequency_diagnostics=[NS(residuals=NS(maximum=0.))])
    def forbidden(*args, **kwargs):
        raise AssertionError('insufficient native workspace must precede target allocation/generation')
    native_empty = np.empty
    def guarded_empty(shape, *args, **kwargs):
        if shape == (2_001_000, 11):
            forbidden()
        return native_empty(shape, *args, **kwargs)
    monkeypatch.setattr(bounded, 'BoundedResponse', Runner)
    monkeypatch.setattr(core, 'FitPoints', Cloud)
    monkeypatch.setattr(bounded.lw, 'supplied_nonlocal_properties', lambda **kwargs: local)
    monkeypatch.setattr(bounded, '_refinement_node',
                        lambda *args, **kwargs: (model, model.parameter_labels, (), None))
    monkeypatch.setattr(bounded, 'fitted_point_targets', forbidden)
    monkeypatch.setattr(np, 'empty', guarded_empty)
    lattice = core.FitPointsOptions()
    lattice.npoints = 2000
    wfn = NS(molecule=lambda: NS(clone=lambda: None))
    quadrature = NS(frequencies=tuple(float(i) for i in range(11)), cp_weights=(1.,)*11)
    with pytest.raises(ValueError, match='native PFIT workspace budget'):
        bounded.bounded_properties(wfn, None, caller_converged=True,
            sites=model.sites, bonds=(), quadrature=quadrature, response_grid=np.zeros((1, 4)),
            smoothing=None, shell_cutoff=1e-8, charge_penalty=1000.,
            anchor_metric_damping=.0005, lattice_options=lattice, localization_rank_limit=2,
            weight_type=4, weight_coefficient=.001, cutoff=1e-4, resources=resources,
            scf_correction='NONE', max_parameters=432, refinement_capacity='molecular',
            max_replay_work=10**12, log=StageLog(0))
    assert calls == [0.]
